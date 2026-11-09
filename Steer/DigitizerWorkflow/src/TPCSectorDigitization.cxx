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

#include "TPCSectorDigitization.h"
#include <sstream>
#include <string>
#include <TChain.h>
#include "DetectorsRaw/HBFUtils.h"
#include "Framework/Logger.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "TPCBase/Sector.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/Digitizer.h"

namespace o2::tpc
{

namespace
{
std::string getBranchNameLeft(int sector)
{
  std::stringstream branchnamestreamleft;
  branchnamestreamleft << "TPCHitsShiftedSector" << int(o2::tpc::Sector::getLeft(o2::tpc::Sector(sector)));
  return branchnamestreamleft.str();
}

std::string getBranchNameRight(int sector)
{
  std::stringstream branchnamestreamright;
  branchnamestreamright << "TPCHitsShiftedSector" << sector;
  return branchnamestreamright.str();
}
} // namespace

std::vector<dataformats::RangeReference<int, int>> digitizeSector(Digitizer& digitizer, const steer::DigitizationContext& context,
                                                                  const std::vector<TChain*>& simChains, int sector, ULong64_t timeframe,
                                                                  SectorFlushBuffers& buffers, const std::function<void()>& onFlush,
                                                                  std::mutex* cdbMutex)
{
  // locks cdbMutex, if given, for calls that read calibrations through the CDBInterface
  auto cdbLock = [cdbMutex]() { return cdbMutex ? std::unique_lock<std::mutex>(*cdbMutex) : std::unique_lock<std::mutex>(); };

  std::vector<dataformats::RangeReference<int, int>> events;
  const bool isContinuous = digitizer.isContinuousReadout();
  const auto& irecords = context.getEventRecords();
  const auto& eventParts = context.getEventParts();

  digitizer.setSector(sector);
  digitizer.setSeedingTimeframe(timeframe);
  {
    auto lock = cdbLock();
    digitizer.init();
  }

  size_t digitCounter = 0;
  auto flush = [&](bool finalFlush = false) {
    buffers.digits.clear();
    buffers.labels.clear();
    buffers.commonMode.clear();
    digitizer.flush(buffers.digits, buffers.labels, buffers.commonMode, finalFlush);
    LOG(info) << "TPC: Flushed " << buffers.digits.size() << " digits, " << buffers.labels.getNElements() << " labels and " << buffers.commonMode.size() << " common mode entries";
    onFlush();
    digitCounter += buffers.digits.size();
  };

  if (isContinuous) {
    auto& hbfu = o2::raw::HBFUtils::Instance();
    double time = hbfu.getFirstIRofTF(o2::InteractionRecord(0, hbfu.orbitFirstSampled)).bc2ns() / 1000.;
    digitizer.setOutputDigitTimeOffset(time);
    if (!irecords.empty()) {
      auto lock = cdbLock();
      digitizer.setStartTime(irecords[0].getTimeNS() / 1000.f);
    }
  }

  // loop over all composite collisions given from context
  // (aka loop over all the interaction records)
  for (int collID = 0; collID < irecords.size(); ++collID) {
    const double eventTime = irecords[collID].getTimeNS() / 1000.f;
    LOG(info) << "TPC: Event time " << eventTime << " us";
    digitizer.setEventTime(eventTime);
    if (!isContinuous) {
      auto lock = cdbLock();
      digitizer.setStartTime(eventTime);
    }
    size_t startSize = digitCounter;

    // for each collision, loop over the constituents event and source IDs
    // (background signal merging is basically taking place here)
    for (auto& part : eventParts[collID]) {
      const int eventID = part.entryID;
      const int sourceID = part.sourceID;

      // get the hits for this event and this source
      std::vector<o2::tpc::HitGroup> hitsLeft;
      std::vector<o2::tpc::HitGroup> hitsRight;
      context.retrieveHits(simChains, getBranchNameLeft(sector).c_str(), part.sourceID, part.entryID, &hitsLeft);
      context.retrieveHits(simChains, getBranchNameRight(sector).c_str(), part.sourceID, part.entryID, &hitsRight);
      LOG(debug) << "TPC: Found " << hitsLeft.size() << " hit groups left and " << hitsRight.size() << " hit groups right in collision " << collID << " eventID " << part.entryID;

      digitizer.process(hitsLeft, eventID, sourceID);
      digitizer.process(hitsRight, eventID, sourceID);

      flush();

      if (!isContinuous) {
        events.emplace_back(startSize, buffers.digits.size());
      }
    }
  }

  // final flushing step; getting everything not yet written out
  if (isContinuous) {
    LOG(info) << "TPC: Final flush";
    flush(true);
    events.emplace_back(0, digitCounter); // all digits are grouped to 1 super-event pseudo-triggered mode
  }
  return events;
}

} // namespace o2::tpc
