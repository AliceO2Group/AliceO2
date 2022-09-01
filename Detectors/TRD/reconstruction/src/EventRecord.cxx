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

#include <string>

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Digit.h"
#include "TRDReconstruction/EventRecord.h"
#include "DataFormatsTRD/Constants.h"

#include "Framework/Output.h"
#include "Framework/ProcessingContext.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/InputRecordWalker.h"

#include "DataFormatsTRD/Constants.h"

#include <cassert>
#include <array>
#include <string>
#include <bitset>
#include <vector>
#include <gsl/span>
#include <typeinfo>

namespace o2::trd
{

void EventRecord::sortTrackletsByHCID()
{
  // sort the tracklets by HCID
  std::stable_sort(std::begin(mTracklets), std::end(mTracklets), [this](const Tracklet64& trackleta, const Tracklet64& trackletb) { return trackleta.getHCID() < trackletb.getHCID(); });
}

void EventRecordContainer::sendData(o2::framework::ProcessingContext& pc, bool generatestats)
{
  //at this point we know the total number of tracklets and digits and triggers.
  auto dataReadStart = std::chrono::high_resolution_clock::now();

  size_t digitcount = 0;
  size_t trackletcount = 0;
  std::vector<Tracklet64> tracklets;
  std::vector<Digit> digits;
  std::vector<TriggerRecord> triggers;
  for (auto& event : mEventRecords) {
    event.sortTrackletsByHCID();
    tracklets.insert(tracklets.end(), event.getTracklets().begin(), event.getTracklets().end());
    digits.insert(digits.end(), event.getDigits().begin(), event.getDigits().end());
    triggers.emplace_back(event.getBCData(), digitcount, event.getDigits().size(), trackletcount, event.getTracklets().size());
    digitcount += event.getDigits().size();
    trackletcount += event.getTracklets().size();
  }

  pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTRD, "DIGITS", 0, o2::framework::Lifetime::Timeframe}, digits);
  pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTRD, "TRACKLETS", 0, o2::framework::Lifetime::Timeframe}, tracklets);
  pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTRD, "TRKTRGRD", 0, o2::framework::Lifetime::Timeframe}, triggers);
  if (generatestats) {
    accumulateStats();
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTRD, "RAWSTATS", 0, o2::framework::Lifetime::Timeframe}, mTFStats);
  }

  std::chrono::duration<double, std::micro> dataReadTime = std::chrono::high_resolution_clock::now() - dataReadStart;
  LOG(debug) << "Preparing for sending and sending data took  " << std::chrono::duration_cast<std::chrono::milliseconds>(dataReadTime).count() << "ms";
}

void EventRecordContainer::accumulateStats()
{
  int eventcount = mEventRecords.size();
  int sumtracklets = 0;
  int sumdigits = 0;
  int sumdigittime = 0;
  int sumtracklettime = 0;
  uint64_t sumwordsrejected = 0;
  uint64_t sumwordsread = 0;
  for (auto event : mEventRecords) {
    sumtracklets += event.getEventStats().mTrackletsFound;
    sumdigits += event.getEventStats().mDigitsFound;
    sumtracklettime += event.getEventStats().mTimeTakenForTracklets;
    sumdigittime += event.getEventStats().mTimeTakenForDigits;
    // OS: the two counters below are not even used anymore, are they needed?
    sumwordsrejected += event.getEventStats().mWordsRejected;
    sumwordsread += event.getEventStats().mWordsRead;
  }
  if (eventcount != 0) {
    mTFStats.mTrackletsPerEvent = sumtracklets / eventcount;
    mTFStats.mDigitsPerEvent = sumdigits / eventcount;
    mTFStats.mTimeTakenForTracklets = sumtracklettime;
    mTFStats.mTimeTakenForDigits = sumdigittime;
  }
}

void EventRecordContainer::setCurrentEventRecord(const InteractionRecord& ir)
{
  // check if we already have an EventRecord for given interaction
  bool foundEventRecord = false;
  for (int idx = 0; idx < mEventRecords.size(); ++idx) {
    if (mEventRecords.at(idx).getBCData() == ir) {
      mCurrEventRecord = idx;
      foundEventRecord = true;
    }
  }
  if (!foundEventRecord) {
    // we add a new EventRecord
    mEventRecords.emplace_back(ir);
    mCurrEventRecord = mEventRecords.size() - 1;
  }
}

void EventRecordContainer::resetCounters()
{
  mTFStats.clear();
}
} // namespace o2::trd
