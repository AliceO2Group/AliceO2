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

/// \file DataSamplingPolicy.cxx
/// \brief Implementation of O2 Data Sampling Policy
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "DataSampling/DataSamplingPolicy.h"
#include "DataSampling/DataSamplingHeader.h"
#include "DataSampling/DataSamplingConditionFactory.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/Logger.h"

#include <boost/property_tree/ptree.hpp>

using namespace o2::framework;

namespace o2::utilities
{

using boost::property_tree::ptree;

DataSamplingPolicy::DataSamplingPolicy(std::string name) : mName(std::move(name))
{
}

void DataSamplingPolicy::registerPath(const InputSpec& inputSpec, const OutputSpec& outputSpec)
{
  mPaths.emplace_back(inputSpec, outputSpec);
}

void DataSamplingPolicy::registerCondition(std::unique_ptr<DataSamplingCondition>&& condition)
{
  mConditions.emplace_back(std::move(condition));
}

void DataSamplingPolicy::setFairMQOutputChannel(std::string channel)
{
  mFairMQOutputChannel = std::move(channel);
}

DataSamplingPolicy DataSamplingPolicy::fromConfiguration(const ptree& config)
{
  auto name = config.get<std::string>("id");
  DataSamplingPolicy policy(name);

  size_t outputId = 0;
  std::vector<InputSpec> inputSpecs = DataDescriptorQueryBuilder::parse(config.get<std::string>("query").c_str());
  std::vector<OutputSpec> outputSpecs;
  // Optionally user can specify the outputs,
  if (auto outputsQuery = config.get<std::string>("outputs", ""); !outputsQuery.empty()) {
    std::vector<InputSpec> outputsAsInputSpecs = DataDescriptorQueryBuilder::parse(outputsQuery.c_str());
    if (outputsAsInputSpecs.size() != inputSpecs.size()) {
      throw std::runtime_error(
        "The number of outputs should match the number of inputs (queries),"
        " which is not the case for the policy '" +
        name + "'(" +
        std::to_string(inputSpecs.size()) + " inputs vs. " + std::to_string(outputsAsInputSpecs.size()) + " outputs).");
    }
    for (const auto& outputAsInputSpec : outputsAsInputSpecs) {
      outputSpecs.emplace_back(DataSpecUtils::asOutputSpec(outputAsInputSpec));
      outputSpecs.back().lifetime = Lifetime::QA;
    }
  } else { // otherwise default format will be used
    for (const auto& inputSpec : inputSpecs) {
      if (DataSpecUtils::getOptionalSubSpec(inputSpec).has_value()) {
        outputSpecs.emplace_back(OutputSpec{
          {inputSpec.binding},
          createPolicyDataOrigin(),
          createPolicyDataDescription(name, outputId++),
          DataSpecUtils::getOptionalSubSpec(inputSpec).value(),
          Lifetime::QA});
      } else {
        outputSpecs.emplace_back(OutputSpec{
          {inputSpec.binding},
          {createPolicyDataOrigin(), createPolicyDataDescription(name, outputId++)},
          Lifetime::QA});
      }
    }
  }
  assert(inputSpecs.size() == outputSpecs.size());
  for (size_t i = 0; i < inputSpecs.size(); i++) {
    policy.registerPath(inputSpecs[i], outputSpecs[i]);
  }

  for (const auto& conditionConfig : config.get_child("samplingConditions")) {
    auto condition = DataSamplingConditionFactory::create(conditionConfig.second.get<std::string>("condition"));
    condition->configure(conditionConfig.second);
    policy.registerCondition(std::move(condition));
  }

  policy.setFairMQOutputChannel(config.get_optional<std::string>("fairMQOutput").value_or(""));

  return policy;
}

const framework::OutputSpec* DataSamplingPolicy::match(const ConcreteDataMatcher& input) const
{
  const auto it = mPaths.find(input);
  return it != mPaths.end() ? &(it->second) : nullptr;
}

bool DataSamplingPolicy::decide(const o2::framework::DataRef& dataRef)
{
  bool decision = std::all_of(mConditions.begin(), mConditions.end(),
                              [dataRef](std::unique_ptr<DataSamplingCondition>& condition) {
                                return condition->decide(dataRef);
                              });

  mTotalAcceptedMessages += decision;
  mTotalEvaluatedMessages++;

  return decision;
}

Output DataSamplingPolicy::prepareOutput(const ConcreteDataMatcher& input, Lifetime lifetime) const
{
  auto result = mPaths.find(input);
  if (result != mPaths.end()) {
    auto dataType = DataSpecUtils::asConcreteDataTypeMatcher(result->second);
    return Output{dataType.origin, dataType.description, input.subSpec};
  } else {
    return Output{header::gDataOriginInvalid, header::gDataDescriptionInvalid};
  }
}

const std::string& DataSamplingPolicy::getName() const
{
  return mName;
}

const DataSamplingPolicy::PathMap& DataSamplingPolicy::getPathMap() const
{
  return mPaths;
}

const std::string& DataSamplingPolicy::getFairMQOutputChannel() const
{
  return mFairMQOutputChannel;
}

std::string DataSamplingPolicy::getFairMQOutputChannelName() const
{
  size_t nameBegin = mFairMQOutputChannel.find("name=") + sizeof("name=") - 1;
  size_t nameEnd = mFairMQOutputChannel.find_first_of(',', nameBegin);
  std::string name = mFairMQOutputChannel.substr(nameBegin, nameEnd - nameBegin);
  return name;
}

uint32_t DataSamplingPolicy::getTotalAcceptedMessages() const
{
  return mTotalAcceptedMessages;
}
uint32_t DataSamplingPolicy::getTotalEvaluatedMessages() const
{
  return mTotalEvaluatedMessages;
}

header::DataOrigin DataSamplingPolicy::createPolicyDataOrigin()
{
  return header::DataOrigin("DS");
}

header::DataDescription DataSamplingPolicy::createPolicyDataDescription(std::string policyName, size_t id)
{
  if (id > 99) {
    throw std::runtime_error("Maximum 100 inputs in DataSamplingPolicy are supported. Call the developers if you really need more.");
  }

  if (policyName.size() > 14) {
    LOG(warning) << "DataSamplingPolicy name '" << policyName << "' is longer than 14 characters, we have to trim it. "
                 << "Use a shorter policy name to avoid potential output name conflicts.";
    policyName.resize(14);
  }

  header::DataDescription outputDescription;
  outputDescription.runtimeInit((policyName + std::to_string(id)).c_str());
  return outputDescription;
}

} // namespace o2::utilities
