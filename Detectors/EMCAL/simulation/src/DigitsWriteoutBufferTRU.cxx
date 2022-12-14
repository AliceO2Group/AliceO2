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
#include "EMCALSimulation/DigitsWriteoutBufferTRU.h"
#include "EMCALSimulation/DigitsVectorStream.h"
#include "EMCALSimulation/LZEROElectronics.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "EMCALBase/TriggerMappingV2.h"
#include "TMath.h"

using namespace o2::emcal;

//_____________________________________________________________________
//
void DigitsWriteoutBufferTRU::fillOutputContainer(bool& isEndOfTimeFrame, bool& isStartOfTimeFrame)
{
  int eventTimeBin = 13;
  bool needsEmptyTimeBins = false;
  int nProcessedTimeBins = 0;
  int timeBin = mFirstTimeBin;
  o2::InteractionRecord nextInteractionRecord;

  std::deque<o2::emcal::DigitTimebin>
    mDequeTime;

  // If end of run or end of timeframe read out what we have
  bool isEnd = mEndOfRun || isEndOfTimeFrame;
  // Checking the Interaction Record for a new event
  // If the time is different from the one that is currently
  // saved, this means there is a new collision and we
  // can read out the previous time bins
  // markerTimeBin indicates how many steps we can
  // put our marker forward
  int markerTimeBin = 0;
  for (auto& time : mTimeBins) {
    // The time bins between the last event and the timing of this event are uncorrelated and can be written out
    // This is true up to 13 samples, which is the pulse of the TRU
    // So either a new collision in less than 13, or 13 samples are read together
    if (!(markerTimeBin < eventTimeBin)) {
      break;
    }

    /// End of Run
    if (isEnd) {
      break;
    }

    // check if minterrecord which is optional exists.
    // if it doesn't, keep the previously assigned interrecord value
    if (time.mInterRecord.has_value()) {
      auto interactionrecordsavedtmp = time.mInterRecord.value();
      nextInteractionRecord = interactionrecordsavedtmp;
      auto nextTime = nextInteractionRecord.bc2ns(nextInteractionRecord.bc, nextInteractionRecord.orbit);
      auto currentTime = mCurrentInteractionRecord.bc2ns(mCurrentInteractionRecord.bc, mCurrentInteractionRecord.orbit);
      // If the new IR and the old IR have different times, this means
      // we have different collisions and we can read out
      if (nextTime != currentTime) {
        break;
      }
    }

    // increasing the marker for the read out, this tells us
    // how many time bins we can move forward
    ++markerTimeBin;
  }

  // Now filling the vector stream
  for (auto& time : mTimeBins) {
    /// the time bins between the last event and the timing of this event are uncorrelated and can be written out
    if (!(nProcessedTimeBins + mFirstTimeBin < markerTimeBin)) {
      break;
    }

    mDequeTime.push_back(time);
    mDigitStream.fill(mDequeTime, mCurrentInteractionRecord);

    ++nProcessedTimeBins;
    ++timeBin;
  }

  mCurrentInteractionRecord = nextInteractionRecord;

  if (nProcessedTimeBins > 0) {
    mFirstTimeBin += nProcessedTimeBins;
    while (nProcessedTimeBins--) {
      mTimeBins.pop_front();
    }
  }
}
//_____________________________________________________________________
//
void DigitsWriteoutBufferTRU::fillOutputContainer()
{
  int eventTimeBin = 13;
  bool needsEmptyTimeBins = false;
  int nProcessedTimeBins = 0;
  int timeBin = mFirstTimeBin;
  o2::InteractionRecord interactionrecordsaved;

  std::deque<o2::emcal::DigitTimebin>
    mDequeTime;

  for (auto& time : mTimeBins) {
    /// the time bins between the last event and the timing of this event are uncorrelated and can be written out
    if (!(nProcessedTimeBins + mFirstTimeBin < eventTimeBin)) {
      break;
    }

    /// End of Run
    if (mEndOfRun) {
      break;
    }

    mDequeTime.push_back(time);

    // check if minterrecord whcih is optional exists.
    // if it doesn't, keep the previously assigned interrecrod value
    if (time.mInterRecord.has_value()) {
      auto interactionrecordsavedtmp = time.mInterRecord.value();
      interactionrecordsaved = interactionrecordsavedtmp;
    }
    mDigitStream.fill(mDequeTime, interactionrecordsaved);

    ++nProcessedTimeBins;
    ++timeBin;
  }

  if (nProcessedTimeBins > 0) {
    mFirstTimeBin += nProcessedTimeBins;
    while (nProcessedTimeBins--) {
      mTimeBins.pop_front();
    }
  }
}
//________________________________________________________
// Constructor: reserves space to keep always a minimum buffer size
DigitsWriteoutBufferTRU::DigitsWriteoutBufferTRU(unsigned int nTimeBins) : mBufferSize(nTimeBins)
{
  for (int itime = 0; itime < nTimeBins; itime++) {
    mTimeBins.push_back(o2::emcal::DigitTimebin());
  }
}
//________________________________________________________
// Reserve space to keep always a minimum buffer size (put it in the cpp)
void DigitsWriteoutBufferTRU::reserve(int eventTimeBin)
{
  const auto space = mBufferSize + eventTimeBin - mFirstTimeBin;
  if (mTimeBins.size() < space) {
    mTimeBins.resize(space);
  }
}
//________________________________________________________
void DigitsWriteoutBufferTRU::init()
{
  const SimParam* simParam = &(o2::emcal::SimParam::Instance());

  mLiveTime = simParam->getLiveTime();
  mBusyTime = simParam->getBusyTime();
  mPreTriggerTime = simParam->getPreTriggerTime();
  mEndOfRun = 0;

  mDigitStream.init();
}
//________________________________________________________
void DigitsWriteoutBufferTRU::clear()
{
  mTimeBins.clear();
  mEndOfRun = 0;
}
//________________________________________________________
void DigitsWriteoutBufferTRU::finish()
{
  mEndOfRun = 1;
}
//________________________________________________________
// Add digits to the buffer
void DigitsWriteoutBufferTRU::addDigits(unsigned int towerID, std::vector<LabeledDigit>& digList)
{

  // mTimeBin has to have the absolute time information
  for (int ientry = 0; ientry < digList.size(); ientry++) {
    auto& buffEntry = mTimeBins[ientry];
    auto& dig = digList.at(ientry);

    auto towerEntry = buffEntry.mDigitMap->find(towerID);
    if (towerEntry == buffEntry.mDigitMap->end()) {
      towerEntry = buffEntry.mDigitMap->insert(std::pair<int, std::list<o2::emcal::LabeledDigit>>(towerID, std::list<o2::emcal::LabeledDigit>())).first;
    }
    towerEntry->second.push_back(dig);
  }
}
