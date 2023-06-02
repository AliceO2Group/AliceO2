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

/// \file MergerBuilder.cxx
/// \brief Definition of MergerBuilder for O2 Mergers
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include <Framework/CompletionPolicyHelpers.h>
#include <Framework/CompletionPolicy.h>
#include <Monitoring/Monitoring.h>
#include <Mergers/FullHistoryMerger.h>

#include "Mergers/MergerBuilder.h"
#include "Mergers/IntegratingMerger.h"
#include "Framework/TimerParamSpec.h"

using namespace o2::framework;

namespace o2::mergers
{

MergerBuilder::MergerBuilder() : mName("INVALID"),
                                 mInputSpecs{},
                                 mOutputSpec{header::gDataOriginInvalid, header::gDataDescriptionInvalid},
                                 mConfig{}
{
}

void MergerBuilder::setName(std::string name)
{
  mName = name;
}

void MergerBuilder::setTopologyPosition(size_t layer, size_t id)
{
  mLayer = layer;
  mId = id;
}

void MergerBuilder::setTimePipeline(size_t timepipeline)
{
  mTimePipeline = timepipeline;
}

void MergerBuilder::setInputSpecs(const framework::Inputs& inputs)
{
  mInputSpecs = inputs;
}

void MergerBuilder::setOutputSpec(const framework::OutputSpec& output)
{
  mOutputSpec = output;
  mOutputSpec.binding = {MergerBuilder::mergerOutputBinding()};
}

void MergerBuilder::setConfig(MergerConfig config)
{
  mConfig = config;
}

framework::DataProcessorSpec MergerBuilder::buildSpec()
{
  framework::DataProcessorSpec merger;

  merger.name = mConfig.detectorName + "-" + mergerIdString() + "-" + mName + std::to_string(mLayer) + "l-" + std::to_string(mId);

  merger.inputs = mInputSpecs;

  merger.outputs.push_back(mOutputSpec);
  framework::DataAllocator::SubSpecificationType subSpec = DataSpecUtils::getOptionalSubSpec(mOutputSpec).value();
  if (DataSpecUtils::validate(mOutputSpec) == false) {
    // inner layer => generate output spec according to scheme
    subSpec = mergerSubSpec(mLayer, mId);
    merger.outputs[0] = OutputSpec{{mergerOutputBinding()},
                                   mergerDataOrigin(),
                                   mergerDataDescription(mName),
                                   subSpec}; // it servers as a unique merger output ID
  } else {
    // last layer
    merger.outputs[0].binding = {mergerOutputBinding()};
  }

  if (mConfig.inputObjectTimespan.value == InputObjectsTimespan::LastDifference) {
    merger.algorithm = framework::adaptFromTask<IntegratingMerger>(mConfig, subSpec);
  } else {
    merger.algorithm = framework::adaptFromTask<FullHistoryMerger>(mConfig, subSpec);
  }

  // Create the TimerSpec for cycleDurations
  std::vector<o2::framework::TimerSpec> timers;
  for (auto& [cycleDuration, validity] : mConfig.publicationDecision.param.decision) {
    timers.push_back({cycleDuration * 1000000000 /*µs*/, validity});
  }

  merger.inputs.push_back({"timer-publish", "TMR", mergerDataDescription(mName), mergerSubSpec(mLayer, mId), framework::Lifetime::Timer, timerSpecs(timers)});
  merger.labels.push_back(mergerLabel());
  if (mConfig.expendable) {
    framework::DataProcessorLabel expendableLabel = {"expendable"};
    merger.labels.emplace_back(expendableLabel);
  }
  merger.maxInputTimeslices = mTimePipeline;

  return std::move(merger);
}

void MergerBuilder::customizeInfrastructure(std::vector<framework::CompletionPolicy>& policies)
{
  auto matcher = [label = mergerLabel()](framework::DeviceSpec const& device) {
    return std::find(device.labels.begin(), device.labels.end(), label) != device.labels.end();
  };
  // each merger's name contains the common label and should always consume
  policies.emplace_back(CompletionPolicyHelpers::consumeWhenAny("MergerCompletionPolicy", matcher));
}

} // namespace o2::mergers
