// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FullHistoryMerger.cxx
/// \brief Implementation of O2 Mergers, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/FullHistoryMerger.h"
#include "Mergers/MergerAlgorithm.h"
#include "Mergers/MergerBuilder.h"
#include "Mergers/MergeInterface.h"

#include "Headers/DataHeader.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include <Monitoring/MonitoringFactory.h>

using namespace o2::header;
using namespace o2::framework;
using namespace std::chrono;

namespace o2::mergers
{

FullHistoryMerger::FullHistoryMerger(const MergerConfig& config, const header::DataHeader::SubSpecificationType& subSpec)
  : mConfig(config),
    mSubSpec(subSpec)
{
  mCollector = monitoring::MonitoringFactory::Get("infologger:///debug?mergers");
}

FullHistoryMerger::~FullHistoryMerger()
{
  delete mFirstObjectSerialized.second.header;
  delete mFirstObjectSerialized.second.payload;
  delete mFirstObjectSerialized.second.spec;
}

void FullHistoryMerger::init(framework::InitContext& ictx)
{
}

void FullHistoryMerger::run(framework::ProcessingContext& ctx)
{
  // we have to avoid mistaking the timer input with data inputs.
  auto* timerHeader = ctx.inputs().get("timer-publish").header;

  for (const DataRef& ref : InputRecordWalker(ctx.inputs())) {
    if (ref.header != timerHeader) {
      updateCache(ref);
      mUpdatesReceived++;
    }
  }

  if (ctx.inputs().isValid("timer-publish") && !mFirstObjectSerialized.first.empty()) {
    mergeCache();
    publish(ctx.outputs());
  }
}

void FullHistoryMerger::updateCache(const DataRef& ref)
{
  auto* dh = get<DataHeader*>(ref.header);
  std::string sourceID = std::string(dh->dataOrigin.str) + "/" + std::string(dh->dataDescription.str) + "/" + std::to_string(dh->subSpecification);

  // I am not sure if ref.spec is always a concrete spec and not a broader matcher. Comparing it this way should be safer.
  if (mFirstObjectSerialized.first.empty() || mFirstObjectSerialized.first == sourceID) {

    // We store one object in the serialized form, so we can take it as the first object to be merged (multiple times).
    // If we kept it deserialized, we would need to require implementing a clone() method in MergeInterface.

    // Clang-Tidy: 'if' statement is unnecessary; deleting null pointer has no effect
    delete mFirstObjectSerialized.second.spec;
    delete mFirstObjectSerialized.second.header;
    delete mFirstObjectSerialized.second.payload;

    mFirstObjectSerialized.first = sourceID;
    mFirstObjectSerialized.second.spec = new InputSpec(*ref.spec);
    mFirstObjectSerialized.second.header = new char[Stack::headerStackSize(reinterpret_cast<std::byte const*>(dh))];
    memcpy((void*)mFirstObjectSerialized.second.header, ref.header, dh->headerSize);
    mFirstObjectSerialized.second.payload = new char[dh->payloadSize];
    memcpy((void*)mFirstObjectSerialized.second.payload, ref.payload, dh->payloadSize);

  } else {
    mCache[sourceID] = object_store_helpers::extractObjectFrom(ref);
  }
}

void FullHistoryMerger::mergeCache()
{
  LOG(INFO) << "Merging " << mCache.size() + 1 << " objects.";

  mMergedObject = object_store_helpers::extractObjectFrom(mFirstObjectSerialized.second);
  assert(!std::holds_alternative<std::monostate>(mMergedObject));
  mObjectsMerged++;

  // We expect that all the objects use the same kind of interface
  if (std::holds_alternative<TObjectPtr>(mMergedObject)) {

    auto target = std::get<TObjectPtr>(mMergedObject);
    for (auto& [name, entry] : mCache) {
      (void)name;
      auto other = std::get<TObjectPtr>(entry);
      algorithm::merge(target.get(), other.get());
      mObjectsMerged++;
    }

  } else if (std::holds_alternative<MergeInterfacePtr>(mMergedObject)) {
    auto target = std::get<MergeInterfacePtr>(mMergedObject);
    for (auto& [name, entry] : mCache) {
      (void)name;
      auto other = std::get<MergeInterfacePtr>(entry);
      target->merge(other.get());
      mObjectsMerged++;
    }
  }
}

void FullHistoryMerger::publish(framework::DataAllocator& allocator)
{
  // todo see if std::visit is faster here
  if (std::holds_alternative<std::monostate>(mMergedObject)) {
    LOG(INFO) << "Nothing to publish yet";
  } else if (std::holds_alternative<MergeInterfacePtr>(mMergedObject)) {
    allocator.snapshot(framework::OutputRef{MergerBuilder::mergerOutputBinding(), mSubSpec},
                       *std::get<MergeInterfacePtr>(mMergedObject));
  } else if (std::holds_alternative<TObjectPtr>(mMergedObject)) {
    allocator.snapshot(framework::OutputRef{MergerBuilder::mergerOutputBinding(), mSubSpec},
                       *std::get<TObjectPtr>(mMergedObject));
  } else {
    throw std::runtime_error("mMergedObject' variant has no value.");
  }

  mTotalObjectsMerged += mObjectsMerged;
  mTotalUpdatesReceived += mUpdatesReceived;
  mCollector->send({mTotalObjectsMerged, "total_objects_merged"}, monitoring::DerivedMetricMode::RATE);
  mCollector->send({mObjectsMerged, "objects_merged_since_last_publication"});
  mCollector->send({mTotalUpdatesReceived, "total_updates_received"}, monitoring::DerivedMetricMode::RATE);
  mCollector->send({mUpdatesReceived, "updates_received_since_last_publication"});
  mObjectsMerged = 0;
  mUpdatesReceived = 0;
}

} // namespace o2::mergers
