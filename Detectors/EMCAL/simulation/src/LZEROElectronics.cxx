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
    // } else if (TimeSums[0] < TimeSums[1] && TimeSums[1] < TimeSums[2] && TimeSums[2] >= TimeSums[3]) {
  } else if (TimeSums[3] < TimeSums[2] && TimeSums[2] < TimeSums[1] && TimeSums[1] >= TimeSums[0]) {
    trendOfDigitsInTower = true;
    // LOG(info) << "DIG SIMONE peakFinderOnPatch in LZEROElectronics: trendOfDigitsInTower = true";
  } else if (TimeSums[0] > 0. || TimeSums[1] > 0. || TimeSums[2] > 0. || TimeSums[3] > 0.) {

    // LOG(info) << "DIG SIMONE peakFinderOnPatch in LZEROElectronics: TimeSums[0] = " << TimeSums[0];
    // LOG(info) << "DIG SIMONE peakFinderOnPatch in LZEROElectronics: TimeSums[1] = " << TimeSums[1];
    // LOG(info) << "DIG SIMONE peakFinderOnPatch in LZEROElectronics: TimeSums[2] = " << TimeSums[2];
    // LOG(info) << "DIG SIMONE peakFinderOnPatch in LZEROElectronics: TimeSums[3] = " << TimeSums[3];
  }
  double integralOfADCvalues = 0;
  for (auto it = TimeSums.begin(); it != TimeSums.end(); it++) {
    integralOfADCvalues += *it;
  }
  // integralOfADCvalues += TimeSums[3];
  // if( integralOfADCvalues != 0 )LOG(info) << "DIG SIMONE peakFinderOnPatch in LZEROElectronics: integralOfADCvalues = " << integralOfADCvalues;
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
  setThreshold(132.);
  // setThreshold(66.);
  // setThreshold(0.);
}
//________________________________________________________
void LZEROElectronics::clear()
{
}
//________________________________________________________
void LZEROElectronics::updatePatchesADC(Patches& p)
{
  // LOG(info) << "DIG SIMONE updatePatchesADC in LZEROElectronics";
  p.updateADC();
}
//_______________________________________________________________________
void LZEROElectronics::addNoiseDigits(Digit& d1)
{
  double amplitude = d1.getAmplitude();
  double sigmaHG = mSimParam->getPinNoise();
  double sigmaLG = mSimParam->getPinNoiseLG();

  uint16_t noiseHG = std::floor(std::abs(mRandomGenerator->Gaus(0, sigmaHG) / constants::EMCAL_TRU_ADCENERGY)); // ADC
  // uint16_t noiseHG = std::floor(std::abs(mRandomGenerator->Gaus(0, sigmaHG) / constants::EMCAL_ADCENERGY)); // ADC
  // uint16_t noiseLG = std::floor(std::abs(mRandomGenerator->Gaus(0, sigmaLG) / (constants::EMCAL_ADCENERGY * constants::EMCAL_HGLGFACTOR))); // ADC
  uint16_t noiseLG = 0; // ADC

  // MCLabel label(true, 1.0);
  Digit d(d1.getTower(), noiseLG, noiseHG, d1.getTimeStamp());

  d1 += d;
}
//_______________________________________________________________________
void LZEROElectronics::fill(const std::deque<o2::emcal::DigitTimebinTRU>& digitlist, const o2::InteractionRecord record, std::vector<Patches>& patchesFromAllTRUs)
{
  // std::map<unsigned int, std::list<Digit>> outputList;
  // LOG(info) << "DIG SIMONE fill in LZEROElectronics: beginning";
  int counterDigitTimeBin = 0;
  int sizemDigitMap = -999;

  for (auto& digitsTimeBin : digitlist) {
    // Inside the DigitTimebinTRU
    // Fill the LZEROElectronics with the new ADC value
    // At the end of the loop run the peak finder
    // Ship to LONEElectronics in case a peak is found
    // Entire logic limited to timebin by timebin -> effectively implementing time scan

    // LOG(info) << "DIG SIMONE fill in LZEROElectronics: beginning, counterDigitTimeBin = " << counterDigitTimeBin;
    counterDigitTimeBin++;

    // LOG(info) << "DIG SIMONE fill in LZEROElectronics: beginning, counterhelp = 0 ";
    int counterhelp = 0;
    for (auto& [fastor, digitsList] : *digitsTimeBin.mDigitMap) {
      // Digit loop
      // The peak finding algorithm is run after getting out of the loop!

      auto whichTRU2 = std::get<0>(mTriggerMap->getTRUFromAbsFastORIndex(fastor));
      auto whichFastOrTRU2 = std::get<1>(mTriggerMap->getTRUFromAbsFastORIndex(fastor));
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: in loop whichFastOr2 = " << whichFastOrTRU2 << ", whichTRU2 = " << whichTRU2 << ", AbsFastOr2 = " << fastor;

      if (digitsList.size() == 0) {
        continue;
      }
      digitsList.sort();

      int digIndex = 0;
      Digit summedDigit;
      bool first = true;
      for (auto& ld : digitsList) {
        // LOG(info) << "DIG SIMONE fill in LZEROElectronics: after whichFastOr LD = " << whichFastOrTRU2 << ", whichTRU = " << whichTRU2 << ", AbsFastOr = " << fastor << ", getType() = " << ld.getType() << ", getAmplitude() = " << ld.getAmplitude() << ", getAmplitudeADC() = " << ld.getAmplitudeADC(ld.getType()) << ", isTRU = " << ld.getTRU() << ", TimeBin = " << counterDigitTimeBin;
        // LOG(info) << "DIG SIMONE fill in LZEROElectronics: after whichFastOr LD = " << whichFastOrTRU2 << ", whichTRU = " << whichTRU2 << ", AbsFastOr = " << fastor << ", getAmplitude() = " << ld.getAmplitude() << ", getAmplitudeADC() = " << ld.getAmplitudeADC() << ", isTRU = " << ld.getTRU() << ", TimeBin = " << counterDigitTimeBin;
        // if(ld.getAmplitude() > 0.05) LOG(info) << "DIG SIMONE fill in LZEROElectronics: after whichFastOr LD = " << whichFastOrTRU2 << ", whichTRU = " << whichTRU2 << ", AbsFastOr = " << fastor << ", getAmplitude() = " << ld.getAmplitude() << ", getAmplitudeADC() = " << ld.getAmplitudeADC() << ", isTRU = " << ld.getTRU();
        if (first) {
          summedDigit = ld;
          first = false;
        } else {
          summedDigit += ld;
        }
      }

      sizemDigitMap = (*digitsTimeBin.mDigitMap).size();
      if (mSimulateNoiseDigits) {
        addNoiseDigits(summedDigit);
      }

      if (summedDigit.getAmplitude() < 0) {
        continue;
      }

      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: after whichTRU = " << whichTRU;
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: after whichFastOr = " << whichFastOr;
      auto whichTRU = std::get<0>(mTriggerMap->getTRUFromAbsFastORIndex(fastor));
      auto whichFastOrTRU = std::get<1>(mTriggerMap->getTRUFromAbsFastORIndex(fastor));
      // auto whichFastOr = std::get<1>(mTriggerMap->getTRUFromAbsFastORIndex(fastor));
      // if(summedDigit.getAmplitude() > 0.1) LOG(info) << "DIG SIMONE fill in LZEROElectronics: after whichFastOr = " << whichFastOrTRU << ", whichTRU = " << whichTRU << ", AbsFastOr = " << fastor << ", getAmplitude() = " << summedDigit.getAmplitude() << ", getAmplitudeADC() = " << summedDigit.getAmplitudeADC() << ", isTRU = " << summedDigit.getTRU();
      // if(summedDigit.getAmplitude() > 0.008) LOG(info) << "DIG SIMONE fill in LZEROElectronics: after whichFastOr = " << whichFastOrTRU << ", whichTRU = " << whichTRU << ", AbsFastOr = " << fastor << ", getAmplitude() = " << summedDigit.getAmplitude() << ", getAmplitudeADC() = " << summedDigit.getAmplitudeADC() << ", isTRU = " << summedDigit.getTRU();

      auto whichFastOr = std::get<1>(mTriggerMap->convertFastORIndexTRUtoSTU(whichTRU, whichFastOrTRU));
      auto& patchTRU = patchesFromAllTRUs[whichTRU];
      auto& fastOrPatchTRU = patchTRU.mFastOrs[whichFastOr];
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: after patchTRU = " << ;
      // if (patchTRU.mFastOrs.size() < 96) {
      //   LOG(info) << "DIG SIMONE fill in LZEROElectronics: patchTRU.mFastOrs.resize, size = " << patchTRU.mFastOrs.size();
      //   patchTRU.mFastOrs.resize(96);
      //   // patchTRU.init();
      // }
      // if (std::get<1>(patchTRU.mPatchIDSeedFastOrIDs[0]) != 0) {
      //   // LOG(info) << "DIG SIMONE fill in LZEROElectronics: mPatchIDSeedFastOrIDs";
      //   // patchTRU.mFastOrs.resize(96);
      //   patchTRU.init();
      // }
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: mPatchIDSeedFastOrIDs[0] = " << std::get<1>(patchTRU.mPatchIDSeedFastOrIDs[0]);
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: mPatchIDSeedFastOrIDs[1] = " << std::get<1>(patchTRU.mPatchIDSeedFastOrIDs[1]);
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: mPatchIDSeedFastOrIDs[2] = " << std::get<1>(patchTRU.mPatchIDSeedFastOrIDs[2]);
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: whichTRU = " << whichTRU;
      // LOG(info) << "DIG SIMONE fill in LZEROElectronics: whichFastOr = " << whichFastOr;

      // if (patchTRU.mFastOrs.size() < 96) {
      //   // LOG(info) << "DIG SIMONE fill in LZEROElectronics: patchTRU.mFastOrs.resize, size = " << patchTRU.mFastOrs.size();
      //   patchTRU.mFastOrs.resize(96);
      // }
      // auto& fastOrPatchTRU = patchTRU.mFastOrs[whichFastOr];

      // if (summedDigit.getAmplitudeADC() != 0) LOG(info) << "DIG SIMONE fill in LZEROElectronics: digit number = " << counterhelp << ", digIndex = " << digIndex << ", energy in ADC = " << summedDigit.getAmplitudeADC() << ", fastOr = " << whichFastOr;
      // if (summedDigit.getAmplitudeADC() != 0) LOG(info) << "DIG SIMONE fill in LZEROElectronics: fastOr[0,1,2,3] = " << fastOrPatchTRU.mADCvalues[0] << ", " << fastOrPatchTRU.mADCvalues[1] << ", "<< fastOrPatchTRU.mADCvalues[2] << ", " << fastOrPatchTRU.mADCvalues[3] << " before update";
      // if (summedDigit.getTRU() == kFALSE) LOG(info) << "DIG SIMONE fill in LZEROElectronics: NO SET TRU!!!!! " ;
      fastOrPatchTRU.updateADC(summedDigit.getAmplitudeADC());
      // if (summedDigit.getAmplitudeADC() != 0) LOG(info) << "DIG SIMONE fill in LZEROElectronics: fastOr[0,1,2,3] = " << fastOrPatchTRU.mADCvalues[0] << ", " << fastOrPatchTRU.mADCvalues[1] << ", "<< fastOrPatchTRU.mADCvalues[2] << ", " << fastOrPatchTRU.mADCvalues[3];

      digIndex++;
      counterhelp++;
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
      updatePatchesADC(patches);
      bool foundPeakCurrentTRU = peakFinderOnAllPatches(patches);
      auto firedPatches = getFiredPatches(patches);
      if (foundPeakCurrentTRU)
        foundPeak = true;
    }

    if (foundPeak == true)
      LOG(info) << "DIG SIMONE fill in LZEROElectronics: foundPeak = " << foundPeak;
    EMCALTriggerInputs TriggerInputsForL1;
    if (foundPeak) {
      TriggerInputsForL1.mInterRecord = record;
      int whichTRU = 0;
      for (auto& patches : patchesFromAllTRUs) {
        if (whichTRU < 52) {
          int whichFastOr = 0;
          for (auto& fastor : patches.mFastOrs) {
            TriggerInputsForL1.mLastTimesumAllFastOrs.push_back(std::make_tuple(whichTRU, std::get<1>(mTriggerMap->convertFastORIndexSTUtoTRU(mTriggerMap->convertTRUIndexTRUtoSTU(whichTRU), whichFastOr, o2::emcal::TriggerMappingV2::DetType_t::DET_EMCAL)), fastor.timesum()));
            // TriggerInputsForL1.mLastTimesumAllFastOrs.push_back(std::make_tuple(whichTRU, whichFastOr, fastor.timesum()));
            whichFastOr++;
          }
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
