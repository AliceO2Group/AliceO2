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

//  Event Record                                                              //
//  Store the tracklets and digits for a single trigger
//  used temporarily for raw data

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

//Digit information
std::vector<Digit>& EventRecord::getDigits() { return mDigits; }
void EventRecord::addDigits(Digit& digit) { mDigits.push_back(digit); }
void EventRecord::addDigits(std::vector<Digit>::iterator& start, std::vector<Digit>::iterator& end) { mDigits.insert(std::end(mDigits), start, end); }

//tracklet information
std::vector<Tracklet64>& EventRecord::getTracklets() { return mTracklets; }
void EventRecord::addTracklet(Tracklet64& tracklet) { mTracklets.push_back(tracklet); }
void EventRecord::addTracklets(std::vector<Tracklet64>::iterator& start, std::vector<Tracklet64>::iterator& end)
{
  mTracklets.insert(std::end(mTracklets), start, end);
}
void EventRecord::addTracklets(std::vector<Tracklet64>& tracklets)
{
  for (auto tracklet : tracklets) {
    mTracklets.push_back(tracklet);
  }
  //mTracklets.insert(mTracklets.back(), tracklets.begin(),tracklets.back());
}

void EventRecord::sortByHCID()
{
  // sort the tracklets by HCID
  std::stable_sort(std::begin(mTracklets), std::end(mTracklets), [this](const Tracklet64& trackleta, const Tracklet64& trackletb) { return trackleta.getHCID() < trackletb.getHCID(); });
}
// now for event storage
void EventStorage::addDigits(InteractionRecord& ir, Digit& digit)
{
  bool added = false;
  for (int count = 0; count < mEventRecords.size(); ++count) {
    if (ir == mEventRecords[count].getBCData()) {
      //TODO replace this with a hash/map not a vector
      mEventRecords[count].addDigits(digit);
      added = true;
    }
  }
  if (!added) {
    // unseen ir so add it
    mEventRecords.push_back(ir);
    mEventRecords.back().addDigits(digit);
  }
}
void EventStorage::addDigits(InteractionRecord& ir, std::vector<Digit>::iterator start, std::vector<Digit>::iterator end)
{
  bool added = false;
  for (int count = 0; count < mEventRecords.size(); ++count) {
    if (ir == mEventRecords[count].getBCData()) {
      //TODO replace this with a hash/map not a vector
      mEventRecords[count].addDigits(start, end);
      added = true;
    }
  }
  if (!added) {
    // unseen ir so add it
    mEventRecords.push_back(ir);
    mEventRecords.back().addDigits(start, end);
  }
}
void EventStorage::addTracklet(InteractionRecord& ir, Tracklet64& tracklet)
{
  bool added = false;
  for (int count = 0; count < mEventRecords.size(); ++count) {
    if (ir == mEventRecords[count].getBCData()) {
      //TODO replace this with a hash/map not a vector
      mEventRecords[count].addTracklet(tracklet);
      added = true;
    }
  }
  if (!added) {
    // unseen ir so add it
    mEventRecords.push_back(ir);
    mEventRecords.back().addTracklet(tracklet);
  }
}
void EventStorage::addTracklets(InteractionRecord& ir, std::vector<Tracklet64>& tracklets)
{
  bool added = false;
  int count = 0;
  for (int count = 0; count < mEventRecords.size(); ++count) {
    if (ir == mEventRecords[count].getBCData()) {
      //TODO replace this with a hash/map not a vector
      mEventRecords[count].addTracklets(tracklets); //mTracklets.insert(mTracklets.back(),start,end);
      added = true;
    }
  }
  if (!added) {
    // unseen ir so add it
    mEventRecords.push_back(ir);
    mEventRecords.back().addTracklets(tracklets);
  }
}
void EventStorage::addTracklets(InteractionRecord& ir, std::vector<Tracklet64>::iterator& start, std::vector<Tracklet64>::iterator& end)
{
  bool added = false;
  for (int count = 0; count < mEventRecords.size(); ++count) {
    if (ir == mEventRecords[count].getBCData()) {
      //TODO replace this with a hash/map not a vector
      mEventRecords[count].addTracklets(start, end); //mTracklets.insert(mTracklets.back(),start,end);
      added = true;
    }
  }
  if (!added) {
    // unseen ir so add it
    mEventRecords.push_back(ir);
    mEventRecords.back().addTracklets(start, end);
    //  LOG(info) << "x unknown ir adding " << std::distance(start,end)<< " tracklets";
  }
}
void EventStorage::unpackData(std::vector<TriggerRecord>& triggers, std::vector<Tracklet64>& tracklets, std::vector<Digit>& digits)
{
  int digitcount = 0;
  int trackletcount = 0;
  for (auto& event : mEventRecords) {
    tracklets.insert(std::end(tracklets), std::begin(event.getTracklets()), std::end(event.getTracklets()));
    digits.insert(std::end(digits), std::begin(event.getDigits()), std::end(event.getDigits()));
    triggers.emplace_back(event.getBCData(), digitcount, event.getDigits().size(), trackletcount, event.getTracklets().size());
    digitcount += event.getDigits().size();
    trackletcount += event.getTracklets().size();
  }
}

void EventStorage::sendData(o2::framework::ProcessingContext& pc, bool displaytracklets)
{
  //at this point we know the total number of tracklets and digits and triggers.
  uint64_t trackletcount = 0;
  uint64_t digitcount = 0;
  uint64_t triggercount = 0;
  sumTrackletsDigitsTriggers(trackletcount, digitcount, triggercount);
  std::vector<Tracklet64> tracklets;
  tracklets.reserve(trackletcount);
  std::vector<Digit> digits;
  digits.reserve(digitcount);
  std::vector<TriggerRecord> triggers;
  triggers.reserve(triggercount);
  for (auto& event : mEventRecords) {
    //sort tracklets
    event.sortByHCID();
    //TODO do this sort in parallel over the events
    tracklets.insert(std::end(tracklets), std::begin(event.getTracklets()), std::end(event.getTracklets()));
    digits.insert(std::end(digits), std::begin(event.getDigits()), std::end(event.getDigits()));
    triggers.emplace_back(event.getBCData(), digitcount, event.getDigits().size(), trackletcount, event.getTracklets().size());
    digitcount += event.getDigits().size();
    trackletcount += event.getTracklets().size();
  }
  if (displaytracklets) { // this is purely for trivial debugging purposes
    for (auto& event : mEventRecords) {
      LOG(info) << "***** Event : " << event.getBCData() << " digit count : " << event.getDigits().size() << " tracklet count : " << event.getTracklets().size();
      auto& trackletv = event.getTracklets();
      for (auto& tracklt : trackletv) {
        LOG(info) << "        " << std::distance(trackletv.begin(), trackletv.end()) << " " << tracklt;
      }
      LOG(info) << "*****";
    }
  }
  LOG(info) << "Sending data onwards with " << digits.size() << " Digits and " << tracklets.size() << " Tracklets and " << triggers.size() << " Triggers";
  pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTRD, "DIGITS", 0, o2::framework::Lifetime::Timeframe}, digits);
  pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTRD, "TRACKLETS", 0, o2::framework::Lifetime::Timeframe}, tracklets);
  pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginTRD, "TRKTRGRD", 0, o2::framework::Lifetime::Timeframe}, triggers);
}

int EventStorage::sumTracklets()
{
  int sum = 0;
  for (auto event : mEventRecords) {
    sum += event.getTracklets().size();
  }
  return sum;
}
int EventStorage::sumDigits()
{
  int sum = 0;
  for (auto event : mEventRecords) {
    sum += event.getDigits().size();
  }
  return sum;
}
void EventStorage::sumTrackletsDigitsTriggers(uint64_t& tracklets, uint64_t& digits, uint64_t& triggers)
{
  int digitsum = 0;
  int trackletsum = 0;
  int triggersum = 0;
  for (auto event : mEventRecords) {
    digitsum += event.getDigits().size();
    trackletsum += event.getTracklets().size();
    triggersum++;
  }
}

std::vector<Tracklet64>& EventStorage::getTracklets(InteractionRecord& ir)
{
  bool found = false;
  for (int count = 0; count < mEventRecords.size(); ++count) {
    if (ir == mEventRecords[count].getBCData()) {
      found = true;
      return mEventRecords[count].getTracklets();
    }
  }
  LOG(warn) << "attempted to get tracklets from IR: " << ir << " total tracklets of:" << sumTracklets();
  printIR();
  return mDummyTracklets;
}
std::vector<Digit>& EventStorage::getDigits(InteractionRecord& ir)
{
  bool found = false;
  for (int count = 0; count < mEventRecords.size(); ++count) {
    if (ir == mEventRecords[count].getBCData()) {
      found = true;
      return mEventRecords[count].getDigits();
    }
  }
  LOG(warn) << "attempted to get digits from IR: " << ir << " total digits of:" << sumDigits();
  printIR();
  return mDummyDigits;
}

void EventStorage::printIR()
{
  for (int count = 0; count < mEventRecords.size(); ++count) {
    LOG(info) << "[" << count << "]" << mEventRecords[count].getBCData() << " ";
  }
}

} // namespace o2::trd
