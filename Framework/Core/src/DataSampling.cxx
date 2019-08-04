// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSampling.cxx
/// \brief Implementation of O2 Data Sampling, v1.0
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/DataSampling.h"
#include "Framework/Dispatcher.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"

#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>

using namespace o2::configuration;
using SubSpecificationType = o2::header::DataHeader::SubSpecificationType;

namespace o2
{
namespace framework
{

std::string DataSampling::createDispatcherName()
{
  return std::string("Dispatcher"); //_") + getenv("HOSTNAME");
}

void DataSampling::GenerateInfrastructure(WorkflowSpec& workflow, const std::string& policiesSource, size_t threads)
{
  LOG(DEBUG) << "Generating Data Sampling infrastructure...";
  Dispatcher dispatcher(createDispatcherName(), policiesSource);
  Options options;

  std::unique_ptr<ConfigurationInterface> cfg = ConfigurationFactory::getConfiguration(policiesSource);
  auto policiesTree = cfg->getRecursive("dataSamplingPolicies");

  for (auto&& policyConfig : policiesTree) {

    std::unique_ptr<DataSamplingPolicy> policy;

    // We don't want the Dispatcher to exit due to one faulty Policy
    try {
      policy = std::make_unique<DataSamplingPolicy>(policyConfig.second);
    } catch (const std::exception& ex) {
      LOG(WARN) << "Could not load the Data Sampling Policy '"
                << policyConfig.second.get_optional<std::string>("id").value_or("") << "', because: " << ex.what();
      continue;
    } catch (...) {
      LOG(WARN) << "Could not load the Data Sampling Policy '"
                << policyConfig.second.get_optional<std::string>("id").value_or("") << "'";
      continue;
    }

    for (const auto& path : policy->getPathMap()) {
      dispatcher.registerPath({path.first, path.second});
    }

    if (!policy->getFairMQOutputChannel().empty()) {
      options.push_back({"channel-config", VariantType::String, policy->getFairMQOutputChannel().c_str(), {"Out-of-band channel config"}});
      LOG(DEBUG) << " - registering output FairMQ channel '" << policy->getFairMQOutputChannel() << "'";
    }
  }

  if (dispatcher.getInputSpecs().size() > 0) {
    DataProcessorSpec spec;
    spec.name = dispatcher.getName();
    spec.inputs = dispatcher.getInputSpecs();
    spec.outputs = dispatcher.getOutputSpecs();
    spec.algorithm = adaptFromTask<Dispatcher>(std::move(dispatcher));
    spec.maxInputTimeslices = threads;
    spec.labels = {{"DataSampling"}, {"Dispatcher"}};
    spec.options = options;

    workflow.emplace_back(std::move(spec));
  } else {
    LOG(DEBUG) << "No input to this dispatcher, it won't be added to the workflow.";
  }
}

void DataSampling::CustomizeInfrastructure(std::vector<CompletionPolicy>& policies)
{
  CompletionPolicy dispatcherConsumesASAP = CompletionPolicyHelpers::defineByName(createDispatcherName(), CompletionPolicy::CompletionOp::Consume);
  policies.push_back(dispatcherConsumesASAP);
}

void DataSampling::CustomizeInfrastructure(std::vector<ChannelConfigurationPolicy>& policies)
{
  // todo: add push-pull for channels that require blocking
  // now it cannot be done, since matching is possible only using data processors names
}

std::vector<InputSpec> DataSampling::InputSpecsForPolicy(const std::string& policiesSource, const std::string& policyName)
{
  std::unique_ptr<ConfigurationInterface> config = ConfigurationFactory::getConfiguration(policiesSource);
  return InputSpecsForPolicy(config.get(), policyName);
}

std::vector<InputSpec> DataSampling::InputSpecsForPolicy(ConfigurationInterface* const config, const std::string& policyName)
{
  std::vector<InputSpec> inputs;
  auto policiesTree = config->getRecursive("dataSamplingPolicies");

  for (auto&& policyConfig : policiesTree) {
    if (policyConfig.second.get<std::string>("id") == policyName) {
      DataSamplingPolicy policy(policyConfig.second);
      for (const auto& path : policy.getPathMap()) {
        InputSpec input = DataSpecUtils::matchingInput(path.second);
        inputs.push_back(input);
      }
      break;
    }
  }
  return inputs;
}

} // namespace framework
} // namespace o2
