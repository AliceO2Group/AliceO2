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
#include "EMCALSimulation/LZEROElectronics.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "EMCALBase/TriggerMappingV2.h"
#include "EMCALSimulation/DigitTimebin.h"
#include "TMath.h"

using namespace o2::emcal;

//_____________________________________________________________________
// Peak finding algorithm
//
// It checks if there is a rising trend on four consecutive Timebins
// If yes, then it compares the integral to a programmable threshold
bool LZEROElectronics::peakFinderOnPatch(Patches& p, unsigned int patchID)
{
  auto& CurrentPatchTimeSum = p.mTimesum[patchID];
  auto& TimeSums = std::get<1>(CurrentPatchTimeSum);
  bool trendOfDigitsInTower = false;
  if (TimeSums.size() < 4) {
    return false;
  } else if (TimeSums[0] < TimeSums[1] && TimeSums[1] < TimeSums[2] && TimeSums[2] >= TimeSums[3]) {
    trendOfDigitsInTower = true;
  }
  double integralOfADCvalues = 0;
  for (auto it = TimeSums.begin(); it != TimeSums.end(); it++) {
    integralOfADCvalues += *it;
  }
  bool peakOverThreshold = false;
  if (integralOfADCvalues > mThreshold && trendOfDigitsInTower) {
    peakOverThreshold = true;
  }
  return peakOverThreshold;
}
//_____________________________________________________________________
// Peak finding algorithm on all patches
// It fills the mPeakFound vector with potential 1s
void LZEROElectronics::peakFinderOnAllPatches(Patches& p)
{
  p.mFiredPatches.clear();
  for (auto& patches : p.mIndexMapPatch) {
    auto PatchID = std::get<0>(patches);
    auto isFound = peakFinderOnPatch(p, PatchID);
    if (isFound)
      p.mFiredPatches.push_back(PatchID);
  }
}
//________________________________________________________
void LZEROElectronics::init()
{
  mSimParam = &(o2::emcal::SimParam::Instance());
  mRandomGenerator = new TRandom3(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  mSimulateNoiseDigits = mSimParam->doSimulateNoiseDigits();
}
//________________________________________________________
void LZEROElectronics::clear()
{
}
//________________________________________________________
void LZEROElectronics::updatePatchesADC(Patches& p)
{
  p.updateADC();
}
//_______________________________________________________________________
void LZEROElectronics::addNoiseDigits(Digit& d1)
{
  double amplitude = d1.getAmplitude();
  double sigmaHG = mSimParam->getPinNoise();
  double sigmaLG = mSimParam->getPinNoiseLG();

  uint16_t noiseHG = std::floor(std::abs(mRandomGenerator->Gaus(0, sigmaHG) / constants::EMCAL_ADCENERGY));                                 // ADC
  uint16_t noiseLG = std::floor(std::abs(mRandomGenerator->Gaus(0, sigmaLG) / (constants::EMCAL_ADCENERGY * constants::EMCAL_HGLGFACTOR))); // ADC

  // MCLabel label(true, 1.0);
  Digit d(d1.getTower(), noiseLG, noiseHG, d1.getTimeStamp());

  d1 += d;
}
//_______________________________________________________________________
void LZEROElectronics::fill(std::deque<o2::emcal::DigitTimebinTRU>& digitlist, o2::InteractionRecord record, std::vector<Patches>& allPatches)
{
  std::map<unsigned int, std::list<Digit>> outputList;

  for (auto& digitsTimeBin : digitlist) {

    for (auto& [fastor, digitsList] : *digitsTimeBin.mDigitMap) {

      if (digitsList.size() == 0) {
        continue;
      }
      digitsList.sort();

      int digIndex = 0;
      for (auto& ld : digitsList) {

        // Loop over all digits in the time sample and sum the digits that belongs to the same fastor and falls in one time bin
        int digIndex1 = 0;
        for (auto ld1 = digitsList.begin(); ld1 != digitsList.end(); ++ld1) {

          if (digIndex == digIndex1) {
            digIndex1++;
            continue;
          }

          if (ld.canAdd(*ld1)) {
            ld += *ld1;
            digitsList.erase(ld1--);
          }
          digIndex1++;
        }

        if (mSimulateNoiseDigits) {
          addNoiseDigits(ld);
        }

        // if (mRemoveDigitsBelowThreshold && (ld.getAmplitude() < (mSimParam->getDigitThreshold() * constants::EMCAL_ADCENERGY))) {
        //   continue;
        // }
        if (ld.getAmplitude() < 0) {
          continue;
        }
        // if ((ld.getTimeStamp() >= mSimParam->getLiveTime()) || (ld.getTimeStamp() < 0)) {
        //   continue;
        // }

        outputList[fastor].push_back(ld);
        digIndex++;
      }
    }
  }

  TriggerMappingV2 triggerMap;
  for (const auto& [fastor, outdiglist] : outputList) {
    auto whichTRU = std::get<0>(triggerMap.getTRUFromAbsFastORIndex(fastor));
    auto whichFastOr = std::get<1>(triggerMap.getTRUFromAbsFastORIndex(fastor));
    Digit updateFastOrDigit;
    for (const auto& d : outdiglist) {
      updateFastOrDigit += d;
    }
    auto& patchTRU = allPatches[whichTRU];
    auto& fastOrPatchTRU = patchTRU.mFastOrs[whichFastOr];
    fastOrPatchTRU.updateADC(updateFastOrDigit.getAmplitudeADC());
  }

  // mTriggerRecords.emplace_back(record, o2::trigger::PhT, mStartIndex, numberOfNewDigits);
  // mStartIndex = mDigits.size();

  // LOG(info) << "Have " << mStartIndex << " digits ";
}
