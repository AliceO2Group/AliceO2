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
bool LZEROElectronics::peakFinderOnPatch(Patches& p, unsigned int patchID)
{
  auto& CurrentPatchTimeSum = p.mTimesum[patchID];
  auto& TimeSums = std::get<1>(CurrentPatchTimeSum);
  bool trendOfDigitsInTower = false;
  if (TimeSums.size() < 4) {
    return false;
  } else if (TimeSums[0] < TimeSums[1] && TimeSums[1] < TimeSums[2] && TimeSums[2] >= TimeSums[3]) {
    trendOfDigitsInTower = true;
    LOG(info) << "DIG SIMONE peakFinderOnPatch in LZEROElectronics: trendOfDigitsInTower = true";
  }
  double integralOfADCvalues = 0;
  for (auto it = TimeSums.begin(); it != TimeSums.end(); it++) {
    integralOfADCvalues += *it;
  }
  if( integralOfADCvalues != 0 )LOG(info) << "DIG SIMONE peakFinderOnPatch in LZEROElectronics: integralOfADCvalues = " << integralOfADCvalues;
  bool peakOverThreshold = false;
  if (integralOfADCvalues > mThreshold && trendOfDigitsInTower) {
    peakOverThreshold = true;
  }
  return peakOverThreshold;
}
//_____________________________________________________________________
// Peak finding algorithm on all patches
// It fills the mPeakFound vector with potential 1s
bool LZEROElectronics::peakFinderOnAllPatches(Patches& p)
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
  mSimParam = &(o2::emcal::SimParam::Instance());
  mRandomGenerator = new TRandom3(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  mSimulateNoiseDigits = mSimParam->doSimulateNoiseDigits();
  mTriggerMap = new TriggerMappingV2(mGeometry);
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

  uint16_t noiseHG = std::floor(std::abs(mRandomGenerator->Gaus(0, sigmaHG) / constants::EMCAL_ADCENERGY)); // ADC
  // uint16_t noiseLG = std::floor(std::abs(mRandomGenerator->Gaus(0, sigmaLG) / (constants::EMCAL_ADCENERGY * constants::EMCAL_HGLGFACTOR))); // ADC
  uint16_t noiseLG = 0; // ADC

  // MCLabel label(true, 1.0);
  Digit d(d1.getTower(), noiseLG, noiseHG, d1.getTimeStamp());

  d1 += d;
}
//_______________________________________________________________________
void LZEROElectronics::fill(std::deque<o2::emcal::DigitTimebinTRU>& digitlist, o2::InteractionRecord record, std::vector<Patches>& patchesFromAllTRUs)
{
  // std::map<unsigned int, std::list<Digit>> outputList;
  // LOG(info) << "DIG SIMONE fill in LZEROElectronics: beginning";
  // int counterDigitTimeBin = 0;
  int sizemDigitMap = -999;

  for (auto& digitsTimeBin : digitlist) {
    // Inside the DigitTimebinTRU
    // Fill the LZEROElectronics with the new ADC value
    // At the end of the loop run the peak finder
    // Ship to LONEElectronics in case a peak is found

    // LOG(info) << "DIG SIMONE fill in LZEROElectronics: beginning, counterDigitTimeBin = " << counterDigitTimeBin;
    // counterDigitTimeBin++;

    for (auto& [fastor, digitsList] : *digitsTimeBin.mDigitMap) {
      int counterhelp = 0;

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
          summedDigit += ld;
        }
      }

      sizemDigitMap = (*digitsTimeBin.mDigitMap).size();
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: before mSimulateNoiseDigits, sizemDigitMap = " << sizemDigitMap;
      if (mSimulateNoiseDigits) {
        addNoiseDigits(summedDigit);
      }

      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: before summedDigit.getAmplitude, fastor  =  " << fastor;

      if (summedDigit.getAmplitude() < 0) {
        continue;
      }

      auto whichTRU = std::get<0>(mTriggerMap->getTRUFromAbsFastORIndex(fastor));
      auto whichFastOr = std::get<1>(mTriggerMap->getTRUFromAbsFastORIndex(fastor));
      auto& patchTRU = patchesFromAllTRUs[whichTRU];
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: whichTRU = " << whichTRU;
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: whichTRwhichFastOrU = " << whichFastOr;
      if (patchTRU.mFastOrs.size() < 96) {
        // LOG(info) << "DIG SIMONE fill in LZEROElectronics: patchTRU.mFastOrs.resize, size = " << patchTRU.mFastOrs.size();
        patchTRU.mFastOrs.resize(96);
      }
      auto& fastOrPatchTRU = patchTRU.mFastOrs[whichFastOr];

      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: whichTRU = " << whichTRU;
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: whichTRwhichFastOrU = " << whichFastOr;

      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: before fastOrPatchTRU.updateADC, counterhelp = " << counterhelp;
      fastOrPatchTRU.updateADC(summedDigit.getAmplitudeADC());
      digIndex++;
    }

    // Evaluate -> peak finder (ALL Patches in ALL TRUs)
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
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: before updatePatchesADC";
      updatePatchesADC(patches);
      bool foundPeakCurrentTRU = peakFinderOnAllPatches(patches);
      auto firedPatches = getFiredPatches(patches);
      if(firedPatches.size() != 0) LOG(info) << "DIG SIMONE fill in LZEROElectronics: size of mFiredPatched = " << firedPatches.size();
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: foundPeakCurrentTRU = " << foundPeakCurrentTRU;
      if (foundPeakCurrentTRU)
        foundPeak = true;
    }

    LOG(info) << "DIG SIMONE fill in LZEROElectronics: foundPeak = " << foundPeak;
    LOG(info) << "DIG SIMONE fill in LZEROElectronics: before EMCALTriggerInputs";
    EMCALTriggerInputs TriggerInputsForL1;
    if (foundPeak) {
      TriggerInputsForL1.mInterRecord = record;
      int whichTRU = 0;
      LOG(info) << "DIG SIMONE fill in LZEROElectronics: before TriggerInputsForL1.mLastTimesumAllFastOrs";
      for (auto& patches : patchesFromAllTRUs) {
        int whichFastOr = 0;
        for (auto& fastor : patches.mFastOrs) {
          LOG(info) << "DIG SIMONE fill in LZEROElectronics: before TriggerInputsForL1.mLastTimesumAllFastOrs";
          LOG(info) << "DIG SIMONE fill in LZEROElectronics: (whichTRU, whichFastOr, fastor.timesum()) = " << whichTRU << ", " << whichFastOr << ", " << fastor.timesum();
          TriggerInputsForL1.mLastTimesumAllFastOrs.push_back(std::make_tuple(whichTRU, whichFastOr, fastor.timesum()));
          whichFastOr++;
        }
        whichTRU++;
      }
    }


    EMCALTriggerInputsPatch TriggerInputsPatch;
    if (foundPeak) {
      TriggerInputsPatch.mInterRecord = record;
      int whichTRU = 0;
      for (auto& patches : patchesFromAllTRUs) {
        int whichPatch = 0;
        bool firedpatch = false;
        if(std::find(patches.mFiredPatches.begin(), patches.mFiredPatches.end(), whichPatch) != patches.mFiredPatches.end()){
          firedpatch = true;
        }  
        for (auto& patchTimeSums : patches.mTimesum) {
          auto& CurrentPatchTimesum = std::get<1>(patchTimeSums);
          LOG(info) << "DIG SIMONE fill in LZEROElectronics: before TriggerInputsPatch.mLastTimesumAllFastOrs";
          TriggerInputsPatch.mLastTimesumAllPatches.push_back(std::make_tuple(whichTRU, whichPatch, CurrentPatchTimesum[3], firedpatch ));
          whichPatch++;
        }
        whichTRU++;
      }
    }


    mTriggers.clear();
    mTriggersPatch.clear();
    mTriggers.push_back(TriggerInputsForL1);
    mTriggersPatch.push_back(TriggerInputsPatch);
  }
}
