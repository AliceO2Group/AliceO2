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
#include <fairlogger/Logger.h>

#include "DataFormatsPHOS/Cell.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "PHOSWorkflow/EventBuilderSpec.h"
#include "Framework/ControlService.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <vector>

using namespace o2::phos;

void EventBuilderSpec::init(framework::InitContext& ctx)
{
}

void EventBuilderSpec::run(framework::ProcessingContext& ctx)
{
  // Merge subevents from two FLPs. FLPs send messages with non-zero subspecifications
  // take TrigRecs with same BC stamps and copy corresponding cells
  // sells are sorted, so find subspec with smaller absId and copy it first (exclude/handle trigger cells!)
  // check if all halves of events were found, otherwise send warning

  std::vector<o2::phos::Cell> outputCells;
  std::vector<o2::phos::TriggerRecord> outputTriggers;
  int nOutCells = 0;

  int posCells = ctx.inputs().getPos("inputcells");
  int posTriggerRecords = ctx.inputs().getPos("inputtriggers");
  int numSlotsCells = ctx.inputs().getNofParts(posCells);
  int numSlotsTriggerRecords = ctx.inputs().getNofParts(posTriggerRecords);

  std::vector<SubspecSet> subspecs;

  // Combine pairs (trigRec,cells) from different subspecs
  for (int islot = 0; islot < numSlotsTriggerRecords; islot++) {
    auto trgrecorddata = ctx.inputs().getByPos(posTriggerRecords, islot);
    auto subspecification = framework::DataRefUtils::getHeader<header::DataHeader*>(trgrecorddata)->subSpecification;
    // Discard message if it is a deadbeaf message (empty timeframe)
    if (subspecification == 0xDEADBEEF) {
      continue;
    }
    auto celldata = ctx.inputs().getByPos(posCells, islot);
    // check if cells from the same source?
    // if not try to find proper slot
    if (framework::DataRefUtils::getHeader<header::DataHeader*>(celldata)->subSpecification != subspecification) {
      for (int islotCell = 0; islotCell < numSlotsCells; islotCell++) {
        celldata = ctx.inputs().getByPos(posCells, islotCell);
        if (framework::DataRefUtils::getHeader<header::DataHeader*>(celldata)->subSpecification == subspecification) {
          break;
        }
      }
      // if still not found pair
      if (framework::DataRefUtils::getHeader<header::DataHeader*>(celldata)->subSpecification != subspecification) {
        LOG(error) << "Can not find cells for TrigRecords with subspecification " << subspecification;
        continue;
      }
    }
    subspecs.emplace_back(ctx.inputs().get<gsl::span<o2::phos::TriggerRecord>>(trgrecorddata),
                          ctx.inputs().get<gsl::span<o2::phos::Cell>>(celldata));
  }

  // Cells in vectors are sorted. Copy in correct order to preserve sorting
  int first1 = 0;             // which subspec should go first: 0: undefined yet, 1:[0],2:[1],
  if (subspecs.size() == 1) { // one input, just copy to output
    for (const Cell& c : subspecs[0].cellSpan) {
      outputCells.emplace_back(c);
    }
    for (const TriggerRecord& r : subspecs[0].trSpan) {
      outputTriggers.emplace_back(r);
    }

  } else {
    if (subspecs.size() > 2) { // error
      LOG(error) << " Too many subspecs for event building:" << subspecs.size();
    } else { // combine two subspecs
      auto tr1 = subspecs[0].trSpan.begin();
      auto tr2 = subspecs[1].trSpan.begin();
      while (tr1 != subspecs[0].trSpan.end() && tr2 != subspecs[1].trSpan.end()) {
        if (tr1->getBCData() == tr2->getBCData()) { // OK, copy
          if (first1 == 0) {                        // order of subspecs not yet defined
            if (subspecs[0].cellSpan.size() > 0) {
              short absId = subspecs[0].cellSpan[0].getAbsId();
              if (absId > 0) { // this is readout cell
                first1 = 1 + (absId - 1) / (2 * 64 * 56);
              } else { // TRU cell
                first1 = 1 + (subspecs[0].cellSpan[0].getTRUId() - 1) / (8 * 28 * 7);
              }
            } else {
              if (subspecs[1].cellSpan.size() > 0) {
                short absId = subspecs[1].cellSpan[0].getAbsId();
                if (absId > 0) { // this is readout cell
                  first1 = 2 - (absId - 1) / (2 * 64 * 56);
                } else { // TRU cell
                  first1 = 2 - (subspecs[1].cellSpan[0].getTRUId() - 1) / (8 * 28 * 7);
                }
              }
              // if both lists are empty, keep first1 zero and try next time
            }
          }

          gsl::details::span_iterator<const o2::phos::Cell> itC1, end1, itC2, end2;
          if (first1 > 1) { // copy first set1 then set2
            itC1 = subspecs[0].cellSpan.begin() + tr1->getFirstEntry();
            end1 = subspecs[0].cellSpan.begin() + tr1->getFirstEntry() + tr1->getNumberOfObjects();
            itC2 = subspecs[1].cellSpan.begin() + tr2->getFirstEntry();
            end2 = subspecs[1].cellSpan.begin() + tr2->getFirstEntry() + tr2->getNumberOfObjects();
          } else {
            itC1 = subspecs[1].cellSpan.begin() + tr2->getFirstEntry();
            end1 = subspecs[1].cellSpan.begin() + tr2->getFirstEntry() + tr2->getNumberOfObjects();
            itC2 = subspecs[0].cellSpan.begin() + tr1->getFirstEntry();
            end2 = subspecs[0].cellSpan.begin() + tr1->getFirstEntry() + tr1->getNumberOfObjects();
          }
          // first copy readout Cells from both events, then trigger
          while (itC1 != end1) {
            if (itC1->getAbsId() == 0) { // TRU cells further
              break;
            }
            outputCells.emplace_back(*itC1);
            ++itC1;
          }
          while (itC2 != end2) {
            if (itC2->getAbsId() == 0) {
              break;
            }
            outputCells.emplace_back(*itC2);
            ++itC2;
          }
          // Copy trigger
          while (itC1 != end1) {
            outputCells.emplace_back(*itC1);
            ++itC1;
          }
          while (itC2 != end2) {
            outputCells.emplace_back(*itC2);
            ++itC2;
          }

          outputTriggers.emplace_back(tr1->getBCData(), nOutCells, outputCells.size() - nOutCells);
          nOutCells = outputCells.size();
          ++tr1;
          ++tr2;
        } else { // inconsistent BCs
          // find smaller BC and copy one part and mark as missing second part
          if (tr1->getBCData() < tr2->getBCData()) {
            auto itC1 = subspecs[0].cellSpan.begin() + tr1->getFirstEntry();
            auto end1 = subspecs[0].cellSpan.begin() + tr1->getFirstEntry() + tr1->getNumberOfObjects();
            while (itC1 != end1) {
              outputCells.emplace_back(*itC1);
            }
            outputTriggers.emplace_back(tr1->getBCData(), nOutCells, outputCells.size() - nOutCells);
            nOutCells = outputCells.size();
            ++tr1;
          } else {
            auto itC1 = subspecs[1].cellSpan.begin() + tr2->getFirstEntry();
            auto end1 = subspecs[1].cellSpan.begin() + tr2->getFirstEntry() + tr2->getNumberOfObjects();
            while (itC1 != end1) {
              outputCells.emplace_back(*itC1);
            }
            outputTriggers.emplace_back(tr2->getBCData(), nOutCells, outputCells.size() - nOutCells);
            nOutCells = outputCells.size();
            ++tr2;
          }
        }
      }
      if (tr1 != subspecs[0].trSpan.end() || tr2 != subspecs[1].trSpan.end()) {
        LOG(error) << "Inconsistent number of TriggerRecords in subsec 1 and 2: " << subspecs[0].trSpan.size() << " and " << subspecs[1].trSpan.size();
      }
    }
  }

  ctx.outputs().snapshot(framework::Output{o2::header::gDataOriginPHS, "CELLS", 0, framework::Lifetime::Timeframe}, outputCells);
  ctx.outputs().snapshot(framework::Output{o2::header::gDataOriginPHS, "CELLTRIGREC", 0, framework::Lifetime::Timeframe}, outputTriggers);
}

o2::framework::DataProcessorSpec o2::phos::getEventBuilderSpec()
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("inputcells", "PHS", "CELLS", 1, o2::framework::Lifetime::Timeframe);          // Input subspec 1, output subspec 0
  inputs.emplace_back("inputtriggers", "PHS", "CELLTRIGREC", 1, o2::framework::Lifetime::Timeframe); // Input subspec 1, output subspec 0
  // output should be anyway generated by raw converter. Or not?
  //  inputs.emplace_back("STFDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);

  o2::framework::Outputs outputs{
    {o2::header::gDataOriginPHS, "CELLS", 0, o2::framework::Lifetime::Timeframe}, // output is zero subspec
    {o2::header::gDataOriginPHS, "CELLTRIGREC", 0, o2::framework::Lifetime::Timeframe}};

  return o2::framework::DataProcessorSpec{"PHOSEventBuilder",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::phos::EventBuilderSpec>(),
                                          o2::framework::Options{}};
}
