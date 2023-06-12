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

void EventRecord::sortTrackletsByDetector()
{
  // sort the tracklets by detector ID
  std::stable_sort(std::begin(mTracklets), std::end(mTracklets), [this](const Tracklet64& trackleta, const Tracklet64& trackletb) { return trackleta.getDetector() < trackletb.getDetector(); });
}

void EventRecordContainer::sendData(o2::framework::ProcessingContext& pc, bool generatestats)
{
  // at this point we know the total number of tracklets and digits and triggers.
  size_t digitcount = 0;
  size_t trackletcount = 0;
  std::vector<Tracklet64> tracklets;
  std::vector<Digit> digits;
  std::vector<TriggerRecord> triggers;
  for (auto& event : mEventRecords) {
    event.sortTrackletsByDetector();
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
  if (mConfigEventPresent) {
    LOGP(info, "ZZZ Sending config event with size of {}", mConfigEventData.size());
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTRD, "CONFEVT", 0, o2::framework::Lifetime::Condition}, mConfigEventData);
  }
}

void EventRecordContainer::accumulateStats()
{
  int eventcount = mEventRecords.size();
  int sumtracklets = 0;
  int sumdigits = 0;
  double sumdigittime = 0;
  double sumtracklettime = 0;
  double sumtime = 0;
  uint64_t sumwordsrejected = 0;
  uint64_t sumwordsread = 0;
  for (auto event : mEventRecords) {
    sumtracklets += event.getEventStats().mTrackletsFound;
    sumdigits += event.getEventStats().mDigitsFound;
    sumtracklettime += event.getEventStats().mTimeTakenForTracklets;
    sumdigittime += event.getEventStats().mTimeTakenForDigits;
    sumtime += event.getEventStats().mTimeTaken;
  }
  if (eventcount != 0) {
    mTFStats.mTrackletsPerEvent = sumtracklets / eventcount;
    mTFStats.mDigitsPerEvent = sumdigits / eventcount;
    mTFStats.mTimeTakenForTracklets = sumtracklettime;
    mTFStats.mTimeTakenForDigits = sumdigittime;
    mTFStats.mTimeTaken = sumtime;
  }
}

void EventRecordContainer::setCurrentEventRecord(const InteractionRecord& ir)
{
  // check if we already have an EventRecord for given interaction
  bool foundEventRecord = false;
  for (int idx = 0; idx < (int)mEventRecords.size(); ++idx) {
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

void EventRecordContainer::reset()
{
  mEventRecords.clear();
  mTFStats.clear();
  mConfigEventPresent = false;
  mConfigEventData.clear();
}

void EventRecordContainer::addConfigEvent(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>& data, uint32_t start, uint32_t end, uint32_t configeventlength, DigitHCHeaderAll& digithcheaders, InteractionRecord& ir)
{
  // copy the config event into the outgoing message.
  // Ir, digitheaders, event payload data.
  mConfigEventPresent = true;
  // append the new incoming config event to the end of the last one.
  mConfigEventData.push_back(end - start); //(uint32_t)ir.bc);
  mConfigEventData.push_back(ir.orbit);
  for (int dcheadercount = 0; dcheadercount < 4; ++dcheadercount) {
    mConfigEventData.push_back(digithcheaders.getHeader(dcheadercount));
  }
  uint32_t length = end - start;
  mConfigEventData.push_back(length);

  for (uint32_t datapos = start; datapos < end; ++datapos) {
    mConfigEventData.push_back(data[datapos]);
  }
  mConfigEventData.push_back(constants::CONFIGEVENTENDA);
  mConfigEventData.push_back(constants::CONFIGEVENTENDB);
}

} // namespace o2::trd
