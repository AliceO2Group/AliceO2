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
#include "EMCALSimulation/DigitsWriteoutBufferTRU.h"
#include "EMCALSimulation/LZEROElectronics.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <fairlogger/Logger.h> // for LOG
#include "EMCALBase/TriggerMappingV2.h"

using namespace o2::emcal;

//_____________________________________________________________________
//
void DigitsWriteoutBufferTRU::fillOutputContainer(bool isEndOfTimeFrame, InteractionRecord& nextInteractionRecord, std::vector<TRUElectronics>& patchesFromAllTRUs, LZEROElectronics& LZERO)
{
  int eventTimeBin = 13;
  bool needsEmptyTimeBins = false;
  int nProcessedTimeBins = 0;

  // auto geom = o2::emcal::Geometry::GetInstance("EMCAL_COMPLETE12SMV1_DCAL_8SM", "Geant4", "EMV-EMCAL");
  // TriggerMappingV2  mTriggerMap(geom);
  // for (auto& digitsTimeBin : mTimeBins) {
  //   for (auto& [fastor, digitsList] : *digitsTimeBin.mDigitMap) {
  //     // Digit loop
  //     // The peak finding algorithm is run after getting out of the loop!
  //     auto whichTRU2 = std::get<0>(mTriggerMap.getTRUFromAbsFastORIndex(fastor));
  //     auto whichFastOrTRU2 = std::get<1>(mTriggerMap.getTRUFromAbsFastORIndex(fastor));
  //     LOG(info) << "DIG SIMONE fillOutputContainer in DigitsWriteoutBufferTRU: in loop whichFastOr = " << whichFastOrTRU2 << ", whichTRU = " << whichTRU2 << ", AbsFastOr = " << fastor;
  //   }
  // }

  std::deque<o2::emcal::DigitTimebinTRU>
    mDequeTime;

  // If end of run or end of timeframe read out what we have
  bool isEnd = mEndOfRun || isEndOfTimeFrame;
  // Checking the Interaction Record for the time of the next event in BC units
  // Difference becomes the new marker if the collision happens before 13 samples
  auto difference = nextInteractionRecord.toLong() - mCurrentInteractionRecord.toLong();
  // LOG(info) << "DIG SIMONE fillOutputContainer in DigitsWriteoutBufferTRU: beginning";
  if (mNoPileupMode || difference >= 13 || isEnd) {
    // Next collision happening way after the current one, just
    // send out and clear entire buffer
    // Simplification and optimization at software level - hardware would continuosly sample, but here we don't need to allocate memory dynamically which we are never going to use
    // Also in a dedicated no-pileup mode we always write out after each collision.
    for (auto& time : mTimeBins) {
      // LOG(info) << "DIG SIMONE fillOutputContainer in DigitsWriteoutBufferTRU: before mDequeTime.push_back";
      mDequeTime.push_back(time);
    }
    // mDigitStream.fill(mDequeTime, mCurrentInteractionRecord);
    // LOG(info) << "DIG SIMONE fillOutputContainer in DigitsWriteoutBufferTRU: before LZERO.fill";
    LZERO.fill(mDequeTime, mCurrentInteractionRecord, patchesFromAllTRUs);
    auto TriggerInputs = LZERO.getTriggerInputs();
    auto TriggerInputsPatches = LZERO.getTriggerInputsPatches();
    int nIter = TriggerInputs.size();
    LOG(info) << "DIG SIMONE fillOutputContainer in DigitsWriteoutBufferTRU: size of TriggerInputs = " << nIter;
    // for (auto& patches : patchesFromAllTRUs) {
    //   LZERO.updatePatchesADC(patches);
    //   LZERO.peakFinderOnAllPatches(patches);
    // }

    mCurrentInteractionRecord = nextInteractionRecord;
    clear();

  } else {
    // Next collsions happing in the tail of the current one,
    // Copy out up to difference and pop and push this amount
    // of samples

    // Now filling the vector stream
    for (auto& time : mTimeBins) {
      if (!(nProcessedTimeBins + mFirstTimeBin < difference)) {
        break;
      }

      // LOG(info) << "DIG SIMONE fillOutputContainer in DigitsWriteoutBufferTRU: before mDequeTime.push_back";
      mDequeTime.push_back(time);
      // mDigitStream.fill(mDequeTime, mCurrentInteractionRecord);

      ++nProcessedTimeBins;
    }

    // mDigitStream.fill(mDequeTime, mCurrentInteractionRecord);
    // LOG(info) << "DIG SIMONE fillOutputContainer in DigitsWriteoutBufferTRU: before LZERO.fill";
    LZERO.fill(mDequeTime, mCurrentInteractionRecord, patchesFromAllTRUs);
    auto TriggerInputs = LZERO.getTriggerInputs();
    auto TriggerInputsPatches = LZERO.getTriggerInputsPatches();
    int nIter = TriggerInputs.size();
    LOG(info) << "DIG SIMONE fillOutputContainer in DigitsWriteoutBufferTRU: size of TriggerInputs = " << nIter;
    // for (auto& patches : patchesFromAllTRUs) {
    //   LZERO.updatePatchesADC(patches);
    //   LZERO.peakFinderOnAllPatches(patches);
    // }
    mCurrentInteractionRecord = nextInteractionRecord;

    if (nProcessedTimeBins > 0) {
      mFirstTimeBin += nProcessedTimeBins;
      while (nProcessedTimeBins--) {
        mTimeBins.pop_front(); // deque
      }
    }
    reserve(15);
  }
  // LOG(info) << "DIG SIMONE fillOutputContainer in DigitsWriteoutBufferTRU: end";
}
//________________________________________________________
// Constructor: reserves space to keep always a minimum buffer size
DigitsWriteoutBufferTRU::DigitsWriteoutBufferTRU(unsigned int nTimeBins) : mBufferSize(nTimeBins)
{
  for (int itime = 0; itime < nTimeBins; itime++) {
    mTimeBins.push_back(o2::emcal::DigitTimebinTRU());
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
  // mPreTriggerTime = simParam->getPreTriggerTime();
  mNoPileupMode = simParam->isDisablePileup();
  mEndOfRun = 0;

  // mDigitStream.init();
}
//________________________________________________________
void DigitsWriteoutBufferTRU::clear()
{
  mTimeBins.clear();
  reserve(15);
  // mEndOfRun = 0;
}
//________________________________________________________
void DigitsWriteoutBufferTRU::finish()
{
  mEndOfRun = 1;
}
//________________________________________________________
// Add digits to the buffer
void DigitsWriteoutBufferTRU::addDigits(unsigned int towerID, std::vector<Digit>& digList)
{
  // mTimeBin has to have the absolute time information
  for (int ientry = 0; ientry < digList.size(); ientry++) {

    auto& buffEntry = mTimeBins[ientry];
    auto& dig = digList.at(ientry);
    // auto geom = o2::emcal::Geometry::GetInstance("EMCAL_COMPLETE12SMV1_DCAL_8SM", "Geant4", "EMV-EMCAL");
    // TriggerMappingV2  mTriggerMap(geom);
    // auto whichTRU = std::get<0>(mTriggerMap.getTRUFromAbsFastORIndex(towerID));
    // auto whichFastOrTRU = std::get<1>(mTriggerMap.getTRUFromAbsFastORIndex(towerID));
    // LOG(info) << "DIG SIMONE addDigits in WriteoutTRU: after whichFastOr = " << whichFastOrTRU << ", whichTRU = " << whichTRU << ", AbsFastOr = " << towerID;

    auto towerEntry = buffEntry.mDigitMap->find(towerID);
    if (towerEntry == buffEntry.mDigitMap->end()) {
      towerEntry = buffEntry.mDigitMap->insert(std::pair<int, std::list<o2::emcal::Digit>>(towerID, std::list<o2::emcal::Digit>())).first;
    }
    towerEntry->second.push_back(dig);
  }
}
