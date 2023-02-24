// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
#include <set>

using namespace o2::configuration;
using namespace o2::framework;
using SubSpecificationType = o2::header::DataHeader::SubSpecificationType;

namespace o2::utilities
{

std::string DataSampling::createDispatcherName()
{
  return std::string("Dispatcher"); //_") + getenv("HOSTNAME");
}

void DataSampling::GenerateInfrastructure(WorkflowSpec& workflow, const std::string& policiesSource, size_t threads, const std::string& host)
{
  std::unique_ptr<ConfigurationInterface> cfg = ConfigurationFactory::getConfiguration(policiesSource);
  if (cfg->getRecursive("").count("dataSamplingPolicies") == 0) {
    LOG(warn) << "No \"dataSamplingPolicies\" structure found in the config file. If no Data Sampling is expected, then it is completely fine.";
    return;
  }
  auto policiesTree = cfg->getRecursive("dataSamplingPolicies");
  Dispatcher dispatcher(createDispatcherName(), policiesSource);
  DataSampling::DoGenerateInfrastructure(dispatcher, workflow, policiesTree, threads, host);
}

void DataSampling::GenerateInfrastructure(WorkflowSpec& workflow, const boost::property_tree::ptree& policiesTree, size_t threads, const std::string& host)
{
  Dispatcher dispatcher(createDispatcherName(), "");
  DataSampling::DoGenerateInfrastructure(dispatcher, workflow, policiesTree, threads, host);
}

void DataSampling::DoGenerateInfrastructure(Dispatcher& dispatcher, WorkflowSpec& workflow, const boost::property_tree::ptree& policiesTree, size_t threads, const std::string& host)
{
  LOG(debug) << "Generating Data Sampling infrastructure...";
  std::set<std::string> ids; // keep track of the ids we have met so far

  for (auto&& policyConfig : policiesTree) {

    // We don't want the Dispatcher to exit due to one faulty Policy
    try {
      auto policy = DataSamplingPolicy::fromConfiguration(policyConfig.second);
      if (ids.count(policy.getName()) == 1) {
        LOG(error) << "A policy with the same id has already been encountered (" + policy.getName() + ")";
      }
      ids.insert(policy.getName());
      std::vector<std::string> machines;
      if (policyConfig.second.count("machines") > 0) {
        for (const auto& machine : policyConfig.second.get_child("machines")) {
          machines.emplace_back(machine.second.get<std::string>(""));
        }
      }
      if (host.empty() || machines.empty() || std::find(machines.begin(), machines.end(), host) != machines.end()) {
        dispatcher.registerPolicy(std::make_unique<DataSamplingPolicy>(std::move(policy)));
      }
    } catch (const std::exception& ex) {
      LOG(warn) << "Could not load the Data Sampling Policy '"
                << policyConfig.second.get_optional<std::string>("id").value_or("") << "', because: " << ex.what();
      continue;
    } catch (...) {
      LOG(warn) << "Could not load the Data Sampling Policy '"
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
    LOG(debug) << "No input to this dispatcher, it won't be added to the workflow.";
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

std::vector<framework::InputSpec> DataSampling::InputSpecsForPolicy(const boost::property_tree::ptree& policiesTree, const std::string& policyName)
{
  std::vector<InputSpec> inputs;
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

std::vector<framework::OutputSpec> DataSampling::OutputSpecsForPolicy(const boost::property_tree::ptree& policiesTree, const std::string& policyName)
{
  std::vector<OutputSpec> outputs;
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

std::optional<uint16_t> DataSampling::PortForPolicy(const boost::property_tree::ptree& policiesTree, const std::string& policyName)
{
  for (auto&& policyConfig : policiesTree) {
    if (policyConfig.second.get<std::string>("id") == policyName) {
      auto boostOptionalPort = policyConfig.second.get_optional<uint16_t>("port");
      return boostOptionalPort.has_value() ? std::optional<uint16_t>(boostOptionalPort.value()) : std::nullopt;
    }
  }
  throw std::runtime_error("Could not find the policy '" + policyName + "'");
}

std::vector<std::string> DataSampling::MachinesForPolicy(const boost::property_tree::ptree& policiesTree, const std::string& policyName)
{
  std::vector<std::string> machines;
  for (auto&& policyConfig : policiesTree) {
    if (policyConfig.second.get<std::string>("id") == policyName) {
      if (policyConfig.second.count("machines") > 0) {
        for (const auto& machine : policyConfig.second.get_child("machines")) {
          machines.emplace_back(machine.second.get<std::string>(""));
        }
      }
      return machines;
    }
  }
  throw std::runtime_error("Could not find the policy '" + policyName + "'");
}

std::string DataSampling::BindLocationForPolicy(const boost::property_tree::ptree& policiesTree, const std::string& policyName)
{
  for (auto&& policyConfig : policiesTree) {
    if (policyConfig.second.get<std::string>("id") == policyName) {
      return policyConfig.second.get_optional<std::string>("bindLocation").value_or("remote");
    }
  }
  throw std::runtime_error("Could not find the policy '" + policyName + "'");
}

} // namespace o2::utilities
