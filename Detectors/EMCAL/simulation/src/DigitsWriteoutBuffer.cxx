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

#include <unordered_map>
#include <vector>
#include <list>
#include <deque>
#include <iostream>
#include <gsl/span>
#include "EMCALSimulation/LabeledDigit.h"
#include "EMCALSimulation/DigitsWriteoutBuffer.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "TMath.h"

using namespace o2::emcal;

DigitsWriteoutBuffer::DigitsWriteoutBuffer(unsigned int nTimeBins) : mBufferSize(nTimeBins)
{
  for (int itime = 0; itime < nTimeBins; itime++) {
    mTimedDigitsFuture.push_back(o2::emcal::DigitTimebin());
  }
}

void DigitsWriteoutBuffer::init()
{
  const SimParam* simParam = &(o2::emcal::SimParam::Instance());

  mLiveTime = simParam->getLiveTime();
  mBusyTime = simParam->getBusyTime();
  mPreTriggerTime = simParam->getPreTriggerTime();
  mSwapPhase = simParam->getBCPhaseSwap();
  mDigitStream.init();
}

void DigitsWriteoutBuffer::clear()
{
  for (auto& iNode : mTimedDigitsFuture) {
    iNode.mRecordMode = false;
    iNode.mEndWindow = false;
    iNode.mDigitMap->clear();
  }
  mTimedDigitsPast.clear();
}

void DigitsWriteoutBuffer::reserve()
{
  if (mTimedDigitsFuture.size() < mBufferSize - 1) {
    int originalSize = mTimedDigitsFuture.size();
    for (int itime = 0; itime < (mBufferSize - originalSize); itime++) {
      mTimedDigitsFuture.push_back(o2::emcal::DigitTimebin());
    }
  }
}

// Add digits to the buffer
void DigitsWriteoutBuffer::addDigits(unsigned int towerID, std::vector<LabeledDigit>& digList)
{

  for (int ientry = 0; ientry < digList.size(); ientry++) {
    auto& buffEntry = mTimedDigitsFuture[ientry];
    auto& dig = digList.at(ientry);

    auto towerEntry = buffEntry.mDigitMap->find(towerID);
    if (towerEntry == buffEntry.mDigitMap->end()) {
      towerEntry = buffEntry.mDigitMap->insert(std::pair<int, std::list<o2::emcal::LabeledDigit>>(towerID, std::list<o2::emcal::LabeledDigit>())).first;
    }
    towerEntry->second.push_back(dig);
  }
}

// When the current time is forwarded (every 100 ns) the entry 0 of the future dequeue
// is removed and pushed at the end of the past dequeue. At the same time a new entry in
// the future dequeue is pushed at the end, and - in case the past dequeue reached 15 entries
// the first entry of the past dequeue is removed.
void DigitsWriteoutBuffer::forwardMarker(o2::InteractionTimeRecord record)
{

  unsigned long eventTime = record.getTimeNS();

  // To see how much the marker will forwarded, or see how much of the future buffer will be written into the past past buffer
  int sampleDifference = (eventTime - mTriggerTime) / 100 - (mLastEventTime - mTriggerTime) / 100;

  for (int idel = 0; idel < sampleDifference; idel++) {

    // Stop reading if record mode is false to save memory
    if (!mTimedDigitsFuture.front().mRecordMode) {
      break;
    }

    // with sampleDifference, the future buffer will written into the past buffer
    // the added entries will be removed the future, and the same number will added as empty bins
    mTimedDigitsPast.push_back(mTimedDigitsFuture.front());
    mTimedDigitsFuture.pop_front();
    mTimedDigitsFuture.push_back(o2::emcal::DigitTimebin());

    if (mTimedDigitsPast.size() > mBufferSize) {
      mTimedDigitsPast.pop_front();
    }

    // If it is the end of the readout window write all the digits, labels, and trigger record into the streamer
    if (mTimedDigitsPast.back().mEndWindow) {

      // Find the trigger time bin
      o2::emcal::DigitTimebin triggerNode;
      for (auto& digitTimebin : mTimedDigitsPast) {
        if (digitTimebin.mTriggerColl) {
          triggerNode = digitTimebin;
          break;
        }
      }
      setSampledDigitsTime();
      mDigitStream.fill(mTimedDigitsPast, triggerNode.mInterRecord.value());
      clear();
    }
  }

  // If we have a trigger, all the time bins in the future buffer will be set to record mode
  // the last time bin will the end of the readout window since it will be mTriggerTime + 1500 ns
  if ((eventTime - mTriggerTime) >= (mLiveTime + mBusyTime) || mFirstEvent) {
    mTriggerTime = eventTime;
    mTimedDigitsFuture.front().mTriggerColl = true;
    mTimedDigitsFuture.front().mInterRecord = record;
    mTimedDigitsFuture[(mLiveTime / 100) - 1].mEndWindow = true;

    long timeStamp = (eventTime / 100) * 100; /// This is to make the event time multiple of 100s
    for (auto& iNode : mTimedDigitsFuture) {
      long diff = (timeStamp - eventTime);
      if (TMath::Abs(diff) > mLiveTime) {
        break;
      }
      iNode.mRecordMode = true;
      timeStamp += 100;
    }
  }

  // If we have a pre-trigger collision, all the time bins in the future buffer will be set to record mode
  if ((eventTime - mTriggerTime) >= (mLiveTime + mBusyTime - mPreTriggerTime)) {
    std::cout << "Pre-trigger collision\n";
    long timeStamp = (eventTime / 100) * 100; /// This is to make the event time multiple of 100s
    for (auto& iNode : mTimedDigitsFuture) {
      long diff = (timeStamp - eventTime);
      if (TMath::Abs(diff) > mLiveTime) {
        break;
      }
      iNode.mRecordMode = true;
      timeStamp += 100;
    }
  }

  mLastEventTime = eventTime;
  // mPhase = ((int)(std::fmod(mLastEventTime, 100) / 25));
  mPhase = (record.bc + mSwapPhase) % 4;

  if (mFirstEvent) {
    mFirstEvent = false;
  }
}

/// For the end of the run only write all the remaining digits into the streamer
void DigitsWriteoutBuffer::finish()
{
  for (unsigned int ibin = 0; ibin < mBufferSize; ibin++) {

    if (!mTimedDigitsFuture.front().mRecordMode || mTimedDigitsFuture.front().mDigitMap->empty()) {
      break;
    }

    mTimedDigitsPast.push_back(mTimedDigitsFuture.front());
    mTimedDigitsFuture.pop_front();

    if (mTimedDigitsPast.size() > mBufferSize) {
      mTimedDigitsPast.pop_front();
    }

    if (mTimedDigitsPast.back().mEndWindow) {

      // Find the trigger time bin
      o2::emcal::DigitTimebin triggerNode;
      for (auto& digitTimebin : mTimedDigitsPast) {
        if (digitTimebin.mTriggerColl) {
          triggerNode = digitTimebin;
          break;
        }
      }
      setSampledDigitsTime();
      mDigitStream.fill(mTimedDigitsPast, triggerNode.mInterRecord.value());
      clear();
      break;
    }
  }
}

void DigitsWriteoutBuffer::setSampledDigitsTime()
{

  // Setting the digit time
  // If we have a delay the first digit in the buffer will start from mDelay,
  // If we also have digits coming from pre-trigger collisions, also their time will start from mDelay,
  // so, to shift the digits back to zero, their time has to be subtracted by the extra digits (with time [0,mDelay])
  int timeStamp = mLiveTime - (mTimedDigitsPast.size() * 100);
  for (auto& buffEntry : mTimedDigitsPast) {
    for (auto& [tower, digitList] : *buffEntry.mDigitMap) {
      for (auto& digit : digitList) {
        digit.setTimeStamp(digit.getTimeStamp() + timeStamp);
      }
    }
    timeStamp += 100;
  }
}
