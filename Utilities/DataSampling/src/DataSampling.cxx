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

#include "DataSampling/DataSampling.h"
#include "DataSampling/DataSamplingPolicy.h"
#include "DataSampling/Dispatcher.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"

#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>

using namespace o2::configuration;
using namespace o2::framework;
using SubSpecificationType = o2::header::DataHeader::SubSpecificationType;

namespace o2::utilities
{

std::string DataSampling::createDispatcherName()
{
  return std::string("Dispatcher"); //_") + getenv("HOSTNAME");
}

void DataSampling::GenerateInfrastructure(WorkflowSpec& workflow, const std::string& policiesSource, size_t threads)
{
  std::unique_ptr<ConfigurationInterface> cfg = ConfigurationFactory::getConfiguration(policiesSource);
  if (cfg->getRecursive("").count("dataSamplingPolicies") == 0) {
    LOG(WARN) << "No \"dataSamplingPolicies\" structure found in the config file. If no Data Sampling is expected, then it is completely fine.";
    return;
  }
  auto policiesTree = cfg->getRecursive("dataSamplingPolicies");
  Dispatcher dispatcher(createDispatcherName(), policiesSource);
  DataSampling::DoGenerateInfrastructure(dispatcher, workflow, policiesTree, threads);
}

void DataSampling::GenerateInfrastructure(WorkflowSpec& workflow, const boost::property_tree::ptree& policiesTree, size_t threads)
{
  Dispatcher dispatcher(createDispatcherName(), "");
  DataSampling::DoGenerateInfrastructure(dispatcher, workflow, policiesTree, threads);
}

void DataSampling::DoGenerateInfrastructure(Dispatcher& dispatcher, WorkflowSpec& workflow, const boost::property_tree::ptree& policiesTree, size_t threads)
{
  LOG(DEBUG) << "Generating Data Sampling infrastructure...";

  for (auto&& policyConfig : policiesTree) {

    std::unique_ptr<DataSamplingPolicy> policy;

    // We don't want the Dispatcher to exit due to one faulty Policy
    try {
      dispatcher.registerPolicy(std::make_unique<DataSamplingPolicy>(DataSamplingPolicy::fromConfiguration(policyConfig.second)));
    } catch (const std::exception& ex) {
      LOG(WARN) << "Could not load the Data Sampling Policy '"
                << policyConfig.second.get_optional<std::string>("id").value_or("") << "', because: " << ex.what();
      continue;
    } catch (...) {
      LOG(WARN) << "Could not load the Data Sampling Policy '"
                << policyConfig.second.get_optional<std::string>("id").value_or("") << "'";
      continue;
    }
  }

  if (dispatcher.numberOfPolicies() > 0) {
    DataProcessorSpec spec;
    spec.name = dispatcher.getName();
    spec.inputs = dispatcher.getInputSpecs();
    spec.outputs = dispatcher.getOutputSpecs();
    spec.maxInputTimeslices = threads;
    spec.labels = {{"DataSampling"}, {"Dispatcher"}};
    spec.options = dispatcher.getOptions();
    spec.algorithm = adaptFromTask<Dispatcher>(std::move(dispatcher));

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
      auto policy = DataSamplingPolicy::fromConfiguration(policyConfig.second);
      for (const auto& path : policy.getPathMap()) {
        InputSpec input = DataSpecUtils::matchingInput(path.second);
        inputs.push_back(input);
      }
      break;
    }
  }
  return inputs;
}

std::vector<InputSpec> DataSampling::InputSpecsForPolicy(std::shared_ptr<configuration::ConfigurationInterface> config, const std::string& policyName)
{
  std::vector<InputSpec> inputs;
  auto policiesTree = config->getRecursive("dataSamplingPolicies");

  for (auto&& policyConfig : policiesTree) {
    if (policyConfig.second.get<std::string>("id") == policyName) {
      auto policy = DataSamplingPolicy::fromConfiguration(policyConfig.second);
      for (const auto& path : policy.getPathMap()) {
        InputSpec input = DataSpecUtils::matchingInput(path.second);
        inputs.push_back(input);
      }
      break;
    }
  }
  return inputs;
}

std::vector<OutputSpec> DataSampling::OutputSpecsForPolicy(const std::string& policiesSource, const std::string& policyName)
{
  std::unique_ptr<ConfigurationInterface> config = ConfigurationFactory::getConfiguration(policiesSource);
  return OutputSpecsForPolicy(config.get(), policyName);
}

std::vector<OutputSpec> DataSampling::OutputSpecsForPolicy(ConfigurationInterface* const config, const std::string& policyName)
{
  std::vector<OutputSpec> outputs;
  auto policiesTree = config->getRecursive("dataSamplingPolicies");

  for (auto&& policyConfig : policiesTree) {
    if (policyConfig.second.get<std::string>("id") == policyName) {
      auto policy = DataSamplingPolicy::fromConfiguration(policyConfig.second);
      for (const auto& path : policy.getPathMap()) {
        outputs.push_back(path.second);
      }
      break;
    }
  }
  return outputs;
}

uint16_t DataSampling::PortForPolicy(configuration::ConfigurationInterface* const config, const std::string& policyName)
{
  auto policiesTree = config->getRecursive("dataSamplingPolicies");
  for (auto&& policyConfig : policiesTree) {
    if (policyConfig.second.get<std::string>("id") == policyName) {
      return policyConfig.second.get<uint16_t>("port");
    }
  }
  throw std::runtime_error("Could not find the policy '" + policyName + "'");
}

uint16_t DataSampling::PortForPolicy(const std::string& policiesSource, const std::string& policyName)
{
  std::unique_ptr<ConfigurationInterface> config = ConfigurationFactory::getConfiguration(policiesSource);
  return PortForPolicy(config.get(), policyName);
}

std::vector<std::string> DataSampling::MachinesForPolicy(configuration::ConfigurationInterface* const config, const std::string& policyName)
{
  std::vector<std::string> machines;
  auto policiesTree = config->getRecursive("dataSamplingPolicies");
  for (auto&& policyConfig : policiesTree) {
    if (policyConfig.second.get<std::string>("id") == policyName) {
      for (const auto& machine : policyConfig.second.get_child("machines")) {
        machines.emplace_back(machine.second.get<std::string>(""));
      }
      return machines;
    }
  }
  throw std::runtime_error("Could not find the policy '" + policyName + "'");
}

std::vector<std::string> DataSampling::MachinesForPolicy(const std::string& policiesSource, const std::string& policyName)
{
  std::unique_ptr<ConfigurationInterface> config = ConfigurationFactory::getConfiguration(policiesSource);
  return MachinesForPolicy(config.get(), policyName);
}

} // namespace o2::utilities
