// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSamplingPolicy.cxx
/// \brief Implementation of O2 Data Sampling Policy
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/DataSamplingPolicy.h"
#include "Framework/DataSamplingHeader.h"
#include "Framework/DataSamplingConditionFactory.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/Logger.h"

#include <boost/property_tree/ptree.hpp>

namespace o2::framework
{

using boost::property_tree::ptree;

DataSamplingPolicy::DataSamplingPolicy() = default;

DataSamplingPolicy::DataSamplingPolicy(const ptree& config)
{
  configure(config);
}

DataSamplingPolicy::~DataSamplingPolicy() = default;

void DataSamplingPolicy::configure(const ptree& config)
{
  mName = config.get<std::string>("id");
  if (mName.size() > 14) {
    LOG(WARNING) << "DataSamplingPolicy name '" << mName << "' is longer than 14 characters, we have to trim it. "
                 << "Use a shorter policy name to avoid potential output name conflicts.";
    mName.resize(14);
  }

  auto subSpecString = config.get_optional<std::string>("subSpec").value_or("*");
  auto subSpec = subSpecString.find_first_of("-*") != std::string::npos ? -1 : std::strtoull(subSpecString.c_str(), nullptr, 10);

  mPaths.clear();
  size_t outputId = 0;
  std::vector<InputSpec> inputSpecs = DataDescriptorQueryBuilder::parse(config.get<std::string>("query").c_str());

  for (const auto& inputSpec : inputSpecs) {

    if (DataSpecUtils::getOptionalSubSpec(inputSpec).has_value()) {
      OutputSpec outputSpec{
        {inputSpec.binding},
        createPolicyDataOrigin(),
        createPolicyDataDescription(mName, outputId++),
        DataSpecUtils::getOptionalSubSpec(inputSpec).value(),
        inputSpec.lifetime};

      mPaths.push_back({inputSpec, outputSpec});

    } else {
      OutputSpec outputSpec{
        {inputSpec.binding},
        {createPolicyDataOrigin(), createPolicyDataDescription(mName, outputId++)},
        inputSpec.lifetime};

      mPaths.push_back({inputSpec, outputSpec});
    }
  }

  mConditions.clear();
  for (const auto& conditionConfig : config.get_child("samplingConditions")) {
    mConditions.push_back(DataSamplingConditionFactory::create(conditionConfig.second.get<std::string>("condition")));
    mConditions.back()->configure(conditionConfig.second);
  }

  mFairMQOutputChannel = config.get_optional<std::string>("fairMQOutput").value_or("");
}

bool DataSamplingPolicy::match(const ConcreteDataMatcher& input) const
{
  return mPaths.find(input) != mPaths.end();
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

const Output DataSamplingPolicy::prepareOutput(const ConcreteDataMatcher& input, Lifetime lifetime) const
{
  auto result = mPaths.find(input);
  if (result != mPaths.end()) {
    auto dataType = DataSpecUtils::asConcreteDataTypeMatcher(result->second);
    return Output{dataType.origin, dataType.description, input.subSpec, lifetime};
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
    LOG(WARNING) << "DataSamplingPolicy name '" << policyName << "' is longer than 14 characters, we have to trim it. "
                 << "Use a shorter policy name to avoid potential output name conflicts.";
    policyName.resize(14);
  }

  header::DataDescription outputDescription;
  outputDescription.runtimeInit((policyName + std::to_string(id)).c_str());
  return outputDescription;
}

} // namespace o2::framework