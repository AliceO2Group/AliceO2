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

/// \file IntegratingMerger.cxx
/// \brief Implementation of O2 Mergers, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/IntegratingMerger.h"

#include "Mergers/MergerAlgorithm.h"
#include "Mergers/MergerBuilder.h"

#include <InfoLogger/InfoLogger.hxx>

#include <Monitoring/MonitoringFactory.h>

#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"

using namespace o2::framework;

namespace o2::mergers
{

IntegratingMerger::IntegratingMerger(const MergerConfig& config, const header::DataHeader::SubSpecificationType& subSpec)
  : mConfig(config),
    mSubSpec(subSpec)
{
}

void IntegratingMerger::init(framework::InitContext& ictx)
{
  mCyclesSinceReset = 0;
  mCollector = monitoring::MonitoringFactory::Get(mConfig.monitoringUrl);
  mCollector->addGlobalTag(monitoring::tags::Key::Subsystem, monitoring::tags::Value::Mergers);

  // set detector field in infologger
  AliceO2::InfoLogger::InfoLoggerContext* ilContext = nullptr;
  try {
    ilContext = &ictx.services().get<AliceO2::InfoLogger::InfoLoggerContext>();
  } catch (const RuntimeErrorRef& err) {
    LOG(warn) << "Could not find the DPL InfoLogger Context.";
  }
  ilContext->setField(AliceO2::InfoLogger::InfoLoggerContext::FieldName::Detector, mConfig.detectorName);
}

void IntegratingMerger::run(framework::ProcessingContext& ctx)
{
  // we have to avoid mistaking the timer input with data inputs.
  auto* timerHeader = ctx.inputs().get("timer-publish").header;

  for (const DataRef& ref : InputRecordWalker(ctx.inputs())) {
    if (ref.header != timerHeader) {
      auto other = object_store_helpers::extractObjectFrom(ref);
      if (std::holds_alternative<std::monostate>(mMergedObject)) {
        LOG(debug) << "Received the first input object in the run or after the last moving window reset";
        mMergedObject = std::move(object_store_helpers::extractObjectFrom(ref));
      } else if (std::holds_alternative<TObjectPtr>(mMergedObject)) {
        // We expect that if the first object was TObject, then all should.
        auto targetAsTObject = std::get<TObjectPtr>(mMergedObject);
        auto otherAsTObject = std::get<TObjectPtr>(other);
        algorithm::merge(targetAsTObject.get(), otherAsTObject.get());
      } else if (std::holds_alternative<MergeInterfacePtr>(mMergedObject)) {
        // We expect that if the first object inherited MergeInterface, then all should.
        auto otherAsMergeInterface = std::get<MergeInterfacePtr>(other);
        std::get<MergeInterfacePtr>(mMergedObject)->merge(otherAsMergeInterface.get());
      } else {
        throw std::runtime_error("mMergedObject' variant has no value.");
      }
      mDeltasMerged++;
    }
  }

  if (ctx.inputs().isValid("timer-publish")) {
    mCyclesSinceReset++;
    publish(ctx.outputs());

    if (mConfig.mergedObjectTimespan.value == MergedObjectTimespan::LastDifference ||
        mConfig.mergedObjectTimespan.value == MergedObjectTimespan::NCycles && mConfig.mergedObjectTimespan.param == mCyclesSinceReset) {
      clear();
    }
  }
}

// I am not calling it reset(), because it does not have to be performed during the FairMQs reset.
void IntegratingMerger::clear()
{
  mMergedObject = std::monostate{};
  mCyclesSinceReset = 0;
  mTotalDeltasMerged = 0;
  mDeltasMerged = 0;
}

void IntegratingMerger::publish(framework::DataAllocator& allocator)
{
  mTotalDeltasMerged += mDeltasMerged;

  if (std::holds_alternative<std::monostate>(mMergedObject)) {
    LOG(info) << "No objects received since start or reset, nothing to publish";
  } else if (std::holds_alternative<MergeInterfacePtr>(mMergedObject)) {
    allocator.snapshot(framework::OutputRef{MergerBuilder::mergerOutputBinding(), mSubSpec},
                       *std::get<MergeInterfacePtr>(mMergedObject));
    LOG(info) << "Published the merged object with " << mTotalDeltasMerged << " deltas in total,"
              << " including " << mDeltasMerged << " in the last cycle.";
  } else if (std::holds_alternative<TObjectPtr>(mMergedObject)) {
    allocator.snapshot(framework::OutputRef{MergerBuilder::mergerOutputBinding(), mSubSpec},
                       *std::get<TObjectPtr>(mMergedObject));
    LOG(info) << "Published the merged object with " << mTotalDeltasMerged << " deltas in total,"
              << " including " << mDeltasMerged << " in the last cycle.";
  } else {
    throw std::runtime_error("mMergedObject' variant has no value.");
  }

  mCollector->send({mTotalDeltasMerged, "total_deltas_merged"}, monitoring::DerivedMetricMode::RATE);
  mCollector->send({mDeltasMerged, "deltas_merged_since_last_publication"});
  mCollector->send({mCyclesSinceReset, "cycles_since_reset"});
  mDeltasMerged = 0;
}

} // namespace o2::mergers
