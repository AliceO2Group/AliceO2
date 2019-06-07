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
#include "Framework/ConfigParamsHelper.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigContext.h"
#include "Framework/BoostOptionsRetriever.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include <unistd.h>
#include <vector>
#include <cstring>
#include <exception>

namespace o2
{
namespace framework
{
using Inputs = std::vector<InputSpec>;
using Outputs = std::vector<OutputSpec>;
using Options = std::vector<ConfigParamSpec>;

} // namespace framework
} // namespace o2

/// To be implemented by the user to specify one or more DataProcessorSpec.
/// 
/// Use the ConfigContext @a context in input to get the value of global configuration
/// properties like command line options, number of available CPUs or whatever
/// can affect the creation of the actual workflow.
///
/// @returns a std::vector of DataProcessorSpec which represents the actual workflow
///         to be executed
o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const&context);

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
void defaultConfiguration(std::vector<o2::framework::ConfigParamSpec> &globalWorkflowOptions) {}
void defaultConfiguration(std::vector<o2::framework::CompletionPolicy> &completionPolicies) {}

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

// This comes from the framework itself. This way we avoid code duplication.
int doMain(int argc, char** argv, o2::framework::WorkflowSpec const& specs,
           std::vector<o2::framework::ChannelConfigurationPolicy> const& channelPolicies,
           std::vector<o2::framework::CompletionPolicy> const &completionPolicies,
           std::vector<o2::framework::ConfigParamSpec> const &workflowOptions,
           o2::framework::ConfigContext &configContext);

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
    std::vector<ChannelConfigurationPolicy> channelPolicies;
    UserCustomizationsHelper::userDefinedCustomization(channelPolicies, 0);
    auto defaultChannelPolicies = ChannelConfigurationPolicy::createDefaultPolicies();
    channelPolicies.insert(std::end(channelPolicies), std::begin(defaultChannelPolicies), std::end(defaultChannelPolicies));

    std::vector<CompletionPolicy> completionPolicies;
    UserCustomizationsHelper::userDefinedCustomization(completionPolicies, 0);
    auto defaultCompletionPolicies = CompletionPolicy::createDefaultPolicies();
    completionPolicies.insert(std::end(completionPolicies), std::begin(defaultCompletionPolicies), std::end(defaultCompletionPolicies));

    std::unique_ptr<ParamRetriever> retriever{ new BoostOptionsRetriever(workflowOptions, true, argc, argv) };
    ConfigParamRegistry workflowOptionsRegistry(std::move(retriever));
    ConfigContext configContext{ workflowOptionsRegistry };
    o2::framework::WorkflowSpec specs = defineDataProcessing(configContext);
    result = doMain(argc, argv, specs, channelPolicies, completionPolicies, workflowOptions, configContext);
  } catch (std::exception const& error) {
    LOG(ERROR) << "error while setting up workflow: " << error.what();
  } catch (...) {
    LOG(ERROR) << "Unknown error while setting up workflow.";
  }

  LOG(INFO) << "Process " << getpid() << " is exiting.";
  return result;
}

#endif
