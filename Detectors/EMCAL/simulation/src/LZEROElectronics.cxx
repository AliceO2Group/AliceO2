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
#include "EMCALSimulation/DigitTimebin.h"
#include "EMCALBase/TriggerMappingV2.h"
#include "TMath.h"
#include <fairlogger/Logger.h> // for LOG

using namespace o2::emcal;

//_____________________________________________________________________
// Peak finding algorithm
//
// It checks if there is a rising trend on four consecutive Timebins
// If yes, then it compares the integral to a programmable threshold
bool LZEROElectronics::peakFinderOnPatch(TRUElectronics& p, unsigned int patchID)
{
  auto& CurrentPatchTimeSum = p.mTimesum[patchID];
  auto& TimeSums = std::get<1>(CurrentPatchTimeSum);
  bool trendOfDigitsInTower = false;
  if (TimeSums.size() < 4) {
    return false;
    // } else if (TimeSums[0] < TimeSums[1] && TimeSums[1] < TimeSums[2] && TimeSums[2] >= TimeSums[3]) {
  } else if (TimeSums[3] < TimeSums[2] && TimeSums[2] < TimeSums[1] && TimeSums[1] >= TimeSums[0]) {
    trendOfDigitsInTower = true;
  } else if (TimeSums[0] > 0. || TimeSums[1] > 0. || TimeSums[2] > 0. || TimeSums[3] > 0.) {
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
bool LZEROElectronics::peakFinderOnAllPatches(TRUElectronics& p)
{
  bool isFoundGlobal = false;
  p.mFiredPatches.clear();
  for (auto& patches : p.mIndexMapPatch) {
    auto PatchID = std::get<0>(patches);
    auto isFound = peakFinderOnPatch(p, PatchID);
    if (isFound) {
      p.mFiredPatches.push_back(PatchID);
      isFoundGlobal = true;
    }
  }
  return isFoundGlobal;
}
//________________________________________________________
void LZEROElectronics::init()
{
  auto mSimParam = &(o2::emcal::SimParam::Instance());
  mSimulateNoiseDigits = mSimParam->doSimulateNoiseDigits();
  setThreshold(mSimParam->getThresholdLZERO());
  // setThreshold(132.);
}
//________________________________________________________
void LZEROElectronics::clear()
{
}
//________________________________________________________
void LZEROElectronics::updatePatchesADC(TRUElectronics& p)
{
  p.updateADC();
}
//_______________________________________________________________________
void LZEROElectronics::addNoiseDigits(Digit& d1)
{
  auto mSimParam = &(o2::emcal::SimParam::Instance());
  double amplitude = d1.getAmplitude();
  double sigma = mSimParam->getPinNoiseTRU();

  TRandom3 mRandomGenerator(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  uint16_t noise = std::floor(std::abs(mRandomGenerator.Gaus(0, sigma) / constants::EMCAL_TRU_ADCENERGY)); // ADC

  Digit d(d1.getTower(), 0., noise, d1.getTimeStamp());

  d1 += d;
}
//_______________________________________________________________________
void LZEROElectronics::fill(const std::deque<o2::emcal::DigitTimebinTRU>& digitlist, const o2::InteractionRecord record, std::vector<TRUElectronics>& patchesFromAllTRUs)
{
  int counterDigitTimeBin = 0;
  int sizemDigitMap = -999;
  TriggerMappingV2 mTriggerMap(mGeometry);

  for (auto& digitsTimeBin : digitlist) {
    // Inside the DigitTimebinTRU
    // Fill the LZEROElectronics with the new ADC value
    // At the end of the loop run the peak finder
    // Ship to LONEElectronics in case a peak is found
    // Entire logic limited to timebin by timebin -> effectively implementing time scan

    counterDigitTimeBin++;

    for (auto& [fastor, digitsList] : *digitsTimeBin.mDigitMap) {
      // Digit loop
      // The peak finding algorithm is run after getting out of the loop!

      if (digitsList.size() == 0) {
        continue;
      }
      digitsList.sort();

      int digIndex = 0;
      Digit summedDigit;
      bool first = true;
      for (auto& ld : digitsList) {
        if (first) {
          summedDigit = ld;
          first = false;
        } else {
          // summedDigit += ld;

          // safety device in case same fastOr
          // but different towers, i.e. remember
          // that the += operator fails WITHOUT
          // feedback if that were to happen
          Digit digitToSum(summedDigit.getTower(), ld.getAmplitude(), summedDigit.getTimeStamp());
          summedDigit += digitToSum;
        }
      }

      sizemDigitMap = (*digitsTimeBin.mDigitMap).size();
      if (mSimulateNoiseDigits) {
        addNoiseDigits(summedDigit);
      }

      auto [whichTRU, whichFastOrTRU] = mTriggerMap.getTRUFromAbsFastORIndex(fastor);

      auto whichFastOr = std::get<1>(mTriggerMap.convertFastORIndexTRUtoSTU(whichTRU, whichFastOrTRU));
      auto& patchTRU = patchesFromAllTRUs[whichTRU];
      auto& fastOrPatchTRU = patchTRU.mFastOrs[whichFastOr];
      fastOrPatchTRU.updateADC(summedDigit.getAmplitudeADC());

      digIndex++;
    }

    // Evaluate -> peak finder (ALL TRUElectronics in ALL TRUs)
    // in case peak found:
    // - Create trigger input (IR of that timebin - delay [typically 8 or 9 samples - rollback (0)])
    // - Create L1 timesums (trivial - last time integral) -> Collect from all fastOrs in ALL TRUs

    // Trigger Inputs needs to have the correction of the delay
    // Propagating for now all the interaction record
    // The typical delay is 8 to 9 BCs and the rollback as well
    // The rollback is taken from EMCALReconstruction/RecoParam.h
    // The delay is due to the interactions between EMCAL and CTP
    // It accounts for the difference in times between L0a, L0b, and then there will be a L1 and L1b delay
    // There is 1BC uncertainty on the trigger readout due to steps in the interaction between CTP and detector simulations
    bool foundPeak = false;
    for (auto& patches : patchesFromAllTRUs) {
      updatePatchesADC(patches);
      bool foundPeakCurrentTRU = peakFinderOnAllPatches(patches);
      auto firedPatches = getFiredPatches(patches);
      if (foundPeakCurrentTRU) {
        foundPeak = true;
      }
    }

    if (foundPeak == true) {
      LOG(debug) << "DIG TRU fill in LZEROElectronics: foundPeak = " << foundPeak;
    }
    EMCALTriggerInputs TriggerInputsForL1;
    if (foundPeak) {
      TriggerInputsForL1.mInterRecord = record;
      int whichTRU = 0;
      for (auto& patches : patchesFromAllTRUs) {
        int whichFastOr = 0;
        for (auto& fastor : patches.mFastOrs) {
          TriggerInputsForL1.mLastTimesumAllFastOrs.push_back(std::make_tuple(whichTRU, std::get<1>(mTriggerMap.convertFastORIndexSTUtoTRU(mTriggerMap.convertTRUIndexTRUtoSTU(whichTRU), whichFastOr, o2::emcal::TriggerMappingV2::DetType_t::DET_EMCAL)), fastor.timesum()));
          whichFastOr++;
        }
        whichTRU++;
      }
    }

    // EMCALTriggerInputsPatch TriggerInputsPatch;
    // if (foundPeak) {
    //   TriggerInputsPatch.mInterRecord = record;
    //   int whichTRU = 0;
    //   for (auto& patches : patchesFromAllTRUs) {
    //     if(whichTRU < 46){
    //     int whichPatch = 0;
    //     bool firedpatch = false;
    //     if(std::find(patches.mFiredPatches.begin(), patches.mFiredPatches.end(), whichPatch) != patches.mFiredPatches.end()){
    //       firedpatch = true;
    //     }
    //     for (auto& patchTimeSums : patches.mTimesum) {
    //       auto& CurrentPatchTimesum = std::get<1>(patchTimeSums);
    //       // if( whichTRU == 30 || whichTRU == 31 || whichTRU == 44 || whichTRU == 45 ){
    //       //   if( whichPatch > 68) continue;
    //       // }
    //       TriggerInputsPatch.mLastTimesumAllPatches.push_back(std::make_tuple(whichTRU, whichPatch, CurrentPatchTimesum[3], firedpatch ));
    //       whichPatch++;
    //     }
    //     }
    //     whichTRU++;
    //   }
    // }

    if (foundPeak) {
      mTriggers.push_back(TriggerInputsForL1);
      // mTriggersPatch.push_back(TriggerInputsPatch);
    }
  }
}
