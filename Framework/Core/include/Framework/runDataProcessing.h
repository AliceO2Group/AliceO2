// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_RUN_DATA_PROCESSING_H
#define FRAMEWORK_RUN_DATA_PROCESSING_H

#include "Framework/ChannelConfigurationPolicy.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataAllocator.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigContext.h"
#include "Framework/BoostOptionsRetriever.h"
#include "Framework/CustomWorkflowTerminationHook.h"
#include "Framework/CommonServices.h"
#include "Framework/Logger.h"

#include <unistd.h>
#include <vector>
#include <cstring>
#include <exception>

namespace boost
{
class exception;
}

namespace o2::framework
{
using Inputs = std::vector<InputSpec>;
using Outputs = std::vector<OutputSpec>;
using Options = std::vector<ConfigParamSpec>;
} // namespace o2::framework

/// To be implemented by the user to specify one or more DataProcessorSpec.
///
/// Use the ConfigContext @a context in input to get the value of global configuration
/// properties like command line options, number of available CPUs or whatever
/// can affect the creation of the actual workflow.
///
/// @returns a std::vector of DataProcessorSpec which represents the actual workflow
///         to be executed
o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& context);

// This template magic allow users to customize the behavior of the process
// by (optionally) implementing a `configure` method which modifies one of the
// objects in question.
//
// For example it can be optionally implemented by the user to specify the
// channel policies for your setup. Use this if you want to customize the way
// your devices communicate between themself, e.g. if you want to use REQ/REP
// in place of PUB/SUB.
//
// The advantage of this approach is that we do not need to expose the
// configurability / configuration object to the user, unless he really wants to
// modify it. The drawback is that we need to declare the `customize` method
// before include this file.

// By default we leave the channel policies unchanged. Notice that the default still include
// a "match all" policy which uses pub / sub
// FIXME: add a debug statement saying that the default policy was used?
void defaultConfiguration(std::vector<o2::framework::ChannelConfigurationPolicy>& channelPolicies) {}
void defaultConfiguration(std::vector<o2::framework::ConfigParamSpec>& globalWorkflowOptions) {}
void defaultConfiguration(std::vector<o2::framework::CompletionPolicy>& completionPolicies) {}
void defaultConfiguration(std::vector<o2::framework::DispatchPolicy>& dispatchPolicies) {}
void defaultConfiguration(std::vector<o2::framework::ServiceSpec>& services)
{
  services = o2::framework::CommonServices::defaultServices();
}

void defaultConfiguration(o2::framework::OnWorkflowTerminationHook& hook)
{
  hook = [](const char*) {};
}

struct UserCustomizationsHelper {
  template <typename T>
  static auto userDefinedCustomization(T& something, int preferUser) -> decltype(customize(something), void())
  {
    customize(something);
  }

  template <typename T>
  static auto userDefinedCustomization(T& something, long preferUser)
    -> decltype(defaultConfiguration(something), void())
  {
    defaultConfiguration(something);
  }
};

namespace o2::framework
{
class ConfigContext;
}
/// Helper used to customize a workflow pipelining options
void overridePipeline(o2::framework::ConfigContext& ctx, std::vector<o2::framework::DataProcessorSpec>& workflow);

/// Helper used to customize a workflow via a template data processor
void overrideCloning(o2::framework::ConfigContext& ctx, std::vector<o2::framework::DataProcessorSpec>& workflow);

// This comes from the framework itself. This way we avoid code duplication.
int doMain(int argc, char** argv, o2::framework::WorkflowSpec const& specs,
           std::vector<o2::framework::ChannelConfigurationPolicy> const& channelPolicies,
           std::vector<o2::framework::CompletionPolicy> const& completionPolicies,
           std::vector<o2::framework::DispatchPolicy> const& dispatchPolicies,
           std::vector<o2::framework::ConfigParamSpec> const& workflowOptions,
           o2::framework::ConfigContext& configContext);

void doBoostException(boost::exception& e);

int main(int argc, char** argv)
{
  using namespace o2::framework;
  using namespace boost::program_options;

  int result = 1;
  try {
    // The 0 here is an int, therefore having the template matching in the
    // SFINAE expression above fit better the version which invokes user code over
    // the default one.
    // The default policy is a catch all pub/sub setup to be consistent with the past.
    std::vector<o2::framework::ConfigParamSpec> workflowOptions;
    UserCustomizationsHelper::userDefinedCustomization(workflowOptions, 0);
    workflowOptions.push_back(ConfigParamSpec{"readers", VariantType::Int64, 1ll, {"number of parallel readers to use"}});
    workflowOptions.push_back(ConfigParamSpec{"pipeline", VariantType::String, "", {"override default pipeline size"}});
    workflowOptions.push_back(ConfigParamSpec{"clone", VariantType::String, "", {"clone processors from a template"}});

    // options for AOD rate limiting
    workflowOptions.push_back(ConfigParamSpec{"aod-memory-rate-limit", VariantType::Int64, 0LL, {"Rate limit AOD processing based on memory"}});

    // options for AOD writer
    workflowOptions.push_back(ConfigParamSpec{"aod-writer-json", VariantType::String, "", {"Name of the json configuration file"}});
    workflowOptions.push_back(ConfigParamSpec{"aod-writer-resfile", VariantType::String, "", {"Default name of the output file"}});
    workflowOptions.push_back(ConfigParamSpec{"aod-writer-resmode", VariantType::String, "RECREATE", {"Creation mode of the result files: NEW, CREATE, RECREATE, UPDATE"}});
    workflowOptions.push_back(ConfigParamSpec{"aod-writer-ntfmerge", VariantType::Int, -1, {"Number of time frames to merge into one file"}});
    workflowOptions.push_back(ConfigParamSpec{"aod-writer-keep", VariantType::String, "", {"Comma separated list of ORIGIN/DESCRIPTION/SUBSPECIFICATION:treename:col1/col2/..:filename"}});

    workflowOptions.push_back(ConfigParamSpec{"fairmq-rate-logging", VariantType::Int, 60, {"Rate logging for FairMQ channels"}});

    workflowOptions.push_back(ConfigParamSpec{"forwarding-policy",
                                              VariantType::String,
                                              "dangling",
                                              {"Which messages to forward."
                                               " dangling: dangling outputs,"
                                               " all: all messages"}});
    workflowOptions.push_back(ConfigParamSpec{"forwarding-destination",
                                              VariantType::String,
                                              "file",
                                              {"Destination for forwarded messages."
                                               " file: write to file,"
                                               " fairmq: send to output proxy"}});

    std::vector<CompletionPolicy> completionPolicies;
    UserCustomizationsHelper::userDefinedCustomization(completionPolicies, 0);
    auto defaultCompletionPolicies = CompletionPolicy::createDefaultPolicies();
    completionPolicies.insert(std::end(completionPolicies), std::begin(defaultCompletionPolicies), std::end(defaultCompletionPolicies));

    std::vector<DispatchPolicy> dispatchPolicies;
    UserCustomizationsHelper::userDefinedCustomization(dispatchPolicies, 0);
    auto defaultDispatchPolicies = DispatchPolicy::createDefaultPolicies();
    dispatchPolicies.insert(std::end(dispatchPolicies), std::begin(defaultDispatchPolicies), std::end(defaultDispatchPolicies));

    std::vector<std::unique_ptr<ParamRetriever>> retrievers;
    std::unique_ptr<ParamRetriever> retriever{new BoostOptionsRetriever(true, argc, argv)};
    retrievers.emplace_back(std::move(retriever));
    auto workflowOptionsStore = std::make_unique<ConfigParamStore>(workflowOptions, std::move(retrievers));
    workflowOptionsStore->preload();
    workflowOptionsStore->activate();
    ConfigParamRegistry workflowOptionsRegistry(std::move(workflowOptionsStore));
    ConfigContext configContext(workflowOptionsRegistry, argc, argv);
    o2::framework::WorkflowSpec specs = defineDataProcessing(configContext);
    overridePipeline(configContext, specs);
    overrideCloning(configContext, specs);
    for (auto& spec : specs) {
      UserCustomizationsHelper::userDefinedCustomization(spec.requiredServices, 0);
    }
    std::vector<ChannelConfigurationPolicy> channelPolicies;
    UserCustomizationsHelper::userDefinedCustomization(channelPolicies, 0);
    auto defaultChannelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(configContext);
    channelPolicies.insert(std::end(channelPolicies), std::begin(defaultChannelPolicies), std::end(defaultChannelPolicies));
    result = doMain(argc, argv, specs, channelPolicies, completionPolicies, dispatchPolicies, workflowOptions, configContext);
  } catch (boost::exception& e) {
    doBoostException(e);
  } catch (std::exception const& error) {
    LOG(ERROR) << "error while setting up workflow: " << error.what();
  } catch (...) {
    LOG(ERROR) << "Unknown error while setting up workflow.";
  }

  char* idstring = nullptr;
  for (int argi = 0; argi < argc; argi++) {
    if (strcmp(argv[argi], "--id") == 0 && argi + 1 < argc) {
      idstring = argv[argi + 1];
      break;
    }
  }
  o2::framework::OnWorkflowTerminationHook onWorkflowTerminationHook;
  UserCustomizationsHelper::userDefinedCustomization(onWorkflowTerminationHook, 0);
  onWorkflowTerminationHook(idstring);
  LOG(INFO) << "Process " << getpid() << " is exiting.";
  return result;
}

#endif
