// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <gsl/span>

#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "Framework/InputSpec.h"
#include "Framework/Logger.h"
#include "EMCALWorkflow/EventBuilderSpec.h"

using namespace o2::emcal;

void EventBuilderSpec::init(framework::InitContext& ctx)
{
}

void EventBuilderSpec::run(framework::ProcessingContext& ctx)
{
  auto dataorigin = o2::header::gDataOriginEMC;
  std::unordered_map<o2::header::DataHeader::SubSpecificationType, gsl::span<const TriggerRecord>> triggers;
  std::unordered_map<o2::header::DataHeader::SubSpecificationType, gsl::span<const Cell>> cells;

  std::vector<Cell> outputCells;
  std::vector<TriggerRecord> outputTriggers;

  auto posCells = ctx.inputs().getPos("inputcells"),
       posTriggerRecords = ctx.inputs().getPos("inputtriggers");
  auto numSlotsCells = ctx.inputs().getNofParts(posCells),
       numSlotsTriggerRecords = ctx.inputs().getNofParts(posTriggerRecords);
  for (decltype(numSlotsCells) islot = 0; islot < numSlotsCells; islot++) {
    auto celldata = ctx.inputs().getByPos(posCells, islot);
    auto subspecification = framework::DataRefUtils::getHeader<header::DataHeader*>(celldata)->subSpecification;
    // Discard message if it is a deadbeaf message (empty timeframe)
    if (subspecification == 0xDEADBEEF) {
      continue;
    }
    cells[subspecification] = ctx.inputs().get<gsl::span<o2::emcal::Cell>>(celldata);
  }
  for (decltype(numSlotsTriggerRecords) islot = 0; islot < numSlotsTriggerRecords; islot++) {
    auto trgrecorddata = ctx.inputs().getByPos(posTriggerRecords, islot);
    auto subspecification = framework::DataRefUtils::getHeader<header::DataHeader*>(trgrecorddata)->subSpecification;
    // Discard message if it is a deadbeaf message (empty timeframe)
    if (subspecification == 0xDEADBEEF) {
      continue;
    }
    triggers[subspecification] = ctx.inputs().get<gsl::span<o2::emcal::TriggerRecord>>(trgrecorddata);
  }

  for (auto interaction : connectRangesFromSubtimeframes(triggers)) {
    int start = outputCells.size(),
        ncellsEvent = 0;
    for (auto [subspec, triggerrecord] : interaction.mRangesSubtimeframe) {
      LOG(DEBUG) << interaction.mInteractionRecord.bc << " / " << interaction.mInteractionRecord.orbit << ": Receiving " << triggerrecord.getEntries() << " digits for subspec " << subspec;
      auto cellcont = cells.find(subspec);
      if (cellcont != cells.end()) {
        for (auto& cell : gsl::span<const Cell>(cellcont->second.data() + triggerrecord.getFirstEntry(), triggerrecord.getEntries())) {
          outputCells.emplace_back(cell);
        }
      } else {
        LOG(ERROR) << "No cell container found for subspec " << subspec;
      }
    }
    if (ncellsEvent) {
      std::sort(outputCells.begin() + start, outputCells.begin() + start + ncellsEvent, [](const Cell& lhs, const Cell& rhs) { return lhs.getTower() < rhs.getTower(); });
    }
    outputTriggers.emplace_back(interaction.mInteractionRecord, interaction.mTriggerType, start, ncellsEvent);
  }

  ctx.outputs().snapshot(framework::Output{dataorigin, "CELLS", 0, framework::Lifetime::Timeframe}, outputCells);
  ctx.outputs().snapshot(framework::Output{dataorigin, "CELLSTRGR", 0, framework::Lifetime::Timeframe}, outputTriggers);
}

std::set<EventBuilderSpec::RangeCollection> EventBuilderSpec::connectRangesFromSubtimeframes(const std::unordered_map<o2::header::DataHeader::SubSpecificationType, gsl::span<const TriggerRecord>>& triggerrecords) const
{
  std::set<RangeCollection> events;

  // Search interaction records from all subevents
  // build also a map with indices of the trigger record in gsl span for each subspecification
  std::unordered_map<o2::InteractionRecord, std::unordered_map<o2::header::DataHeader::SubSpecificationType, int>> allInteractions;
  for (auto& [subspecification, trgrec] : triggerrecords) {
    for (decltype(trgrec.size()) trgPos = 0; trgPos < trgrec.size(); trgPos++) {
      auto eventIR = trgrec[trgPos].getBCData();
      auto interactionPtr = allInteractions.find(eventIR);
      if (interactionPtr == allInteractions.end()) {
        allInteractions[eventIR] = {{subspecification, trgPos}};
      } else {
        interactionPtr->second.insert({subspecification, trgPos});
      }
    }
  }

  // iterate over all subevents for all bunch crossings
  for (const auto& [collisionID, subevents] : allInteractions) {
    RangeCollection nextevent;
    nextevent.mInteractionRecord = collisionID;
    bool first = true;
    for (auto& [subspecification, indexInSubevent] : subevents) {
      auto& eventTR = triggerrecords.find(subspecification)->second[indexInSubevent];
      if (first) {
        nextevent.mTriggerType = eventTR.getTriggerBits();
        first = false;
      }
      nextevent.mRangesSubtimeframe.push_back({subspecification, o2::dataformats::RangeReference(eventTR.getFirstEntry(), eventTR.getNumberOfObjects())});
    }
    events.insert(nextevent);
  }
  return events;
}

o2::framework::DataProcessorSpec o2::emcal::getEventBuilderSpec(std::vector<unsigned int> subspecifications)
{
  auto dataorigin = o2::header::gDataOriginEMC;
  o2::framework::Inputs inputs;
  for (auto& spec : subspecifications) {
    inputs.push_back({"inputcells", dataorigin, "CELLS", spec, o2::framework::Lifetime::Timeframe});
    inputs.push_back({"inputtriggers", dataorigin, "CELLSTRGR", spec, o2::framework::Lifetime::Timeframe});
  }
  o2::framework::Outputs outputs{
    {dataorigin, "CELLS", 0, o2::framework::Lifetime::Timeframe},
    {dataorigin, "CELLSTRGR", 0, o2::framework::Lifetime::Timeframe}};

  return o2::framework::DataProcessorSpec{"EMCALEventBuilder",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<EventBuilderSpec>(),
                                          o2::framework::Options{}};
}