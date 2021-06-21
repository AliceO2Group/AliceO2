// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  mCollector = monitoring::MonitoringFactory::Get(mConfig.monitoringUrl);
  mCollector->addGlobalTag(monitoring::tags::Key::Subsystem, monitoring::tags::Value::Mergers);
}

void IntegratingMerger::run(framework::ProcessingContext& ctx)
{
  // we have to avoid mistaking the timer input with data inputs.
  auto* timerHeader = ctx.inputs().get("timer-publish").header;

  for (const DataRef& ref : InputRecordWalker(ctx.inputs())) {
    if (ref.header != timerHeader) {
      if (std::holds_alternative<std::monostate>(mMergedObject)) {
        mMergedObject = object_store_helpers::extractObjectFrom(ref);

      } else if (std::holds_alternative<TObjectPtr>(mMergedObject)) {
        // We expect that if the first object was TObject, then all should.
        auto other = TObjectPtr(framework::DataRefUtils::as<TObject>(ref).release(), algorithm::deleteTCollections);
        auto target = std::get<TObjectPtr>(mMergedObject);
        algorithm::merge(target.get(), other.get());

      } else if (std::holds_alternative<MergeInterfacePtr>(mMergedObject)) {
        // We expect that if the first object inherited MergeInterface, then all should.
        auto other = framework::DataRefUtils::as<MergeInterface>(ref);
        std::get<MergeInterfacePtr>(mMergedObject)->merge(other.get());
      } else {
        throw std::runtime_error("mMergedObject' variant has no value.");
      }
      mDeltasMerged++;
    }
  }

  if (ctx.inputs().isValid("timer-publish")) {

    publish(ctx.outputs());

    if (mConfig.mergedObjectTimespan.value == MergedObjectTimespan::LastDifference) {
      mMergedObject = std::monostate{};
    }
  }
}

void IntegratingMerger::publish(framework::DataAllocator& allocator)
{
  mTotalDeltasMerged += mDeltasMerged;

  if (std::holds_alternative<std::monostate>(mMergedObject)) {
    LOG(INFO) << "Nothing to publish yet";
  } else if (std::holds_alternative<MergeInterfacePtr>(mMergedObject)) {
    allocator.snapshot(framework::OutputRef{MergerBuilder::mergerOutputBinding(), mSubSpec},
                       *std::get<MergeInterfacePtr>(mMergedObject));
    LOG(INFO) << "Published the merged object with " << mTotalDeltasMerged << " deltas in total,"
              << " including " << mDeltasMerged << " in the last cycle.";
  } else if (std::holds_alternative<TObjectPtr>(mMergedObject)) {
    allocator.snapshot(framework::OutputRef{MergerBuilder::mergerOutputBinding(), mSubSpec},
                       *std::get<TObjectPtr>(mMergedObject));
    LOG(INFO) << "Published the merged object with " << mTotalDeltasMerged << " deltas in total,"
              << " including " << mDeltasMerged << " in the last cycle.";
  } else {
    throw std::runtime_error("mMergedObject' variant has no value.");
  }

  mCollector->send({mTotalDeltasMerged, "total_deltas_merged"}, monitoring::DerivedMetricMode::RATE);
  mCollector->send({mDeltasMerged, "deltas_merged_since_last_publication"});
  mDeltasMerged = 0;
}

} // namespace o2::mergers
