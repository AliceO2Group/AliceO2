// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "DataFormatsTRD/EventRecord.h"
#include "DataFormatsTRD/CompressedDigit.h"
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
std::vector<CompressedDigit>& EventRecord::getCompressedDigits() { return mCompressedDigits; }
void EventRecord::addDigits(Digit& digit) { mDigits.push_back(digit); }
void EventRecord::addCompressedDigits(CompressedDigit& digit) { mCompressedDigits.push_back(digit); }
void EventRecord::addDigits(std::vector<Digit>::iterator& start, std::vector<Digit>::iterator& end) { mDigits.insert(std::end(mDigits), start, end); }
void EventRecord::addCompressedDigits(std::vector<CompressedDigit>::iterator& start, std::vector<CompressedDigit>::iterator& end) { mCompressedDigits.insert(std::end(mCompressedDigits), start, end); }

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

// now for event storage
void EventStorage::addDigits(InteractionRecord& ir, Digit& digit)
{
  bool added = false;
  for (auto eventrecord : mEventRecords) {
    if (ir == eventrecord.getBCData()) {
      //TODO replace this with a hash/map not a vector
      eventrecord.addDigits(digit);
      added = true;
    }
  }
  if (!added) {
    // unseen ir so add it
    mEventRecords.push_back(ir);
    mEventRecords.back().addDigits(digit);
  }
}
void EventStorage::addCompressedDigits(InteractionRecord& ir, CompressedDigit& digit)
{
  bool added = false;
  int count = 0;
  for (auto eventrecord : mEventRecords) {
    if (ir == eventrecord.getBCData()) {
      //TODO replace this with a hash/map not a vector
      mEventRecords[count].addCompressedDigits(digit);
      added = true;
    }
    count++;
  }
  if (!added) {
    // unseen ir so add it
    mEventRecords.push_back(ir);
    mEventRecords.back().addCompressedDigits(digit);
  }
}
void EventStorage::addDigits(InteractionRecord& ir, std::vector<Digit>::iterator start, std::vector<Digit>::iterator end)
{
  bool added = false;
  int count = 0;
  for (auto eventrecord : mEventRecords) {
    if (ir == eventrecord.getBCData()) {
      //TODO replace this with a hash/map not a vector
      mEventRecords[count].addDigits(start, end);
      added = true;
    }
    count++;
  }
  if (!added) {
    // unseen ir so add it
    mEventRecords.push_back(ir);
    mEventRecords.back().addDigits(start, end);
  }
}
void EventStorage::addCompressedDigits(InteractionRecord& ir, std::vector<CompressedDigit>::iterator start, std::vector<CompressedDigit>::iterator end)
{
  bool added = false;
  int count = 0;
  for (auto eventrecord : mEventRecords) {
    if (ir == eventrecord.getBCData()) {
      //TODO replace this with a hash/map not a vector
      mEventRecords[count].addCompressedDigits(start, end);
      added = true;
    }
    count++;
  }
  if (!added) {
    // unseen ir so add it
    mEventRecords.push_back(ir);
    mEventRecords.back().addCompressedDigits(start, end);
  }
}
void EventStorage::addTracklet(InteractionRecord& ir, Tracklet64& tracklet)
{
  bool added = false;
  int count = 0;
  for (auto eventrecord : mEventRecords) {
    if (ir == eventrecord.getBCData()) {
      //TODO replace this with a hash/map not a vector
      mEventRecords[count].addTracklet(tracklet);
      added = true;
    }
    count++;
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
  for (auto eventrecord : mEventRecords) {
    if (ir == eventrecord.getBCData()) {
      //TODO replace this with a hash/map not a vector
      mEventRecords[count].addTracklets(tracklets); //mTracklets.insert(mTracklets.back(),start,end);
                                                    // LOG(info) << "adding " << tracklets.size()  << " tracklets and tracklet sum:  " << sumTracklets() << " IR count : "<< mEventRecords.size();;
      added = true;
    }
    count++;
  }
  if (!added) {
    // unseen ir so add it
    mEventRecords.push_back(ir);
    mEventRecords.back().addTracklets(tracklets);
    // hLOG(info) << "unknown ir adding " << tracklets.size()  << " tracklets and sum of : "<< sumTracklets() << " IR count : "<< mEventRecords.size();
  }
}
void EventStorage::addTracklets(InteractionRecord& ir, std::vector<Tracklet64>::iterator& start, std::vector<Tracklet64>::iterator& end)
{
  bool added = false;
  int count = 0;
  for (auto eventrecord : mEventRecords) {
    if (ir == eventrecord.getBCData()) {
      //TODO replace this with a hash/map not a vector
      mEventRecords[count].addTracklets(start, end); //mTracklets.insert(mTracklets.back(),start,end);
      //  LOG(info) << "x iknown ir adding " << std::distance(start,end)<< " tracklets";
      added = true;
    }
    count++;
  }
  if (!added) {
    // unseen ir so add it
    mEventRecords.push_back(ir);
    mEventRecords.back().addTracklets(start, end);
    //  LOG(info) << "x unknown ir adding " << std::distance(start,end)<< " tracklets";
  }
}
void EventStorage::unpackDataForSending(std::vector<TriggerRecord>& triggers, std::vector<Tracklet64>& tracklets, std::vector<Digit>& digits)
{
  int digitcount = 0;
  int trackletcount = 0;
  for (auto event : mEventRecords) {
    tracklets.insert(std::end(tracklets), std::begin(event.getTracklets()), std::end(event.getTracklets()));
    digits.insert(std::end(digits), std::begin(event.getDigits()), std::end(event.getDigits()));
    triggers.emplace_back(event.getBCData(), digitcount, event.getDigits().size(), trackletcount, event.getTracklets().size());
    digitcount += event.getDigits().size();
    trackletcount += event.getTracklets().size();
  }
}
void EventStorage::unpackDataForSending(std::vector<TriggerRecord>& triggers, std::vector<Tracklet64>& tracklets, std::vector<CompressedDigit>& digits)
{
  int digitcount = 0;
  int trackletcount = 0;
  for (auto event : mEventRecords) {
    tracklets.insert(std::end(tracklets), std::begin(event.getTracklets()), std::end(event.getTracklets()));
    digits.insert(std::end(digits), std::begin(event.getCompressedDigits()), std::end(event.getCompressedDigits()));
    triggers.emplace_back(event.getBCData(), digitcount, event.getDigits().size(), trackletcount, event.getTracklets().size());
    digitcount += event.getDigits().size();
    trackletcount += event.getTracklets().size();
    //LOG(info) << "For IR::" << event.getBCData();
    //LOG(info) << "tracklets to add : " << event.getTracklets().size() << " trackletcout:" << trackletcount;
    //LOG(info) << "digits to add : " << event.getDigits().size() << " digitcount " << digitcount;
  }
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
std::vector<Tracklet64>& EventStorage::getTracklets(InteractionRecord& ir)
{
  bool found = false;
  for (int i = 0; i < mEventRecords.size(); ++i) {
    if (ir == mEventRecords[i].getBCData()) {
      found = true;
      return mEventRecords[i].getTracklets();
    }
  }
  LOG(warn) << "attempted to get tracklets from IR: " << ir << " total tracklets of:" << sumTracklets();
  printIR();
  return mDummyTracklets;
}
std::vector<Digit>& EventStorage::getDigits(InteractionRecord& ir)
{
  bool found = false;
  for (auto event : mEventRecords) {
    if (ir == event.getBCData()) {
      found = true;
      return event.getDigits();
    }
  }
  LOG(warn) << "attempted to get digits from IR: " << ir << " total digits of:" << sumDigits();
  printIR();
  return mDummyDigits;
}

std::vector<CompressedDigit>& EventStorage::getCompressedDigits(InteractionRecord& ir)
{
  bool found = false;
  for (auto event : mEventRecords) {
    if (ir == event.getBCData()) {
      found = true;
      return event.getCompressedDigits();
    }
  }
  LOG(warn) << "attempted to get digits from IR: " << ir << " total digits of:" << sumDigits();
  printIR();
  return mDummyCompressedDigits;
}

void EventStorage::printIR()
{
  std::string records;
  int count = 0;
  for (auto event : mEventRecords) {
    LOG(info) << "[" << count << "]" << event.getBCData() << " ";
    count++;
  }
}

} // namespace o2::trd
