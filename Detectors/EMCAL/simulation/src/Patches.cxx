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

/// \file Patches.cxx
/// \brief Implementation of the EMCAL Patches for the LZEROElectronics
#include <cstring>
#include <gsl/span>
#include <fairlogger/Logger.h> // for LOG
#include "EMCALSimulation/Patches.h"
#include "EMCALBase/TriggerMappingV2.h"

using namespace o2::emcal;

//____________________________________________________________________________
Patches::Patches() : mPatchSize(2), mWhichSide(0), mWhichSuperModuleSize(0)
{
  // DEFAULT CONSTRUCTOR
  mFastOrs.resize(96);
}
//____________________________________________________________________________
Patches::Patches(int patchSize, int whichSide, int whichSuperModuleSize) : mPatchSize(patchSize), mWhichSide(whichSide), mWhichSuperModuleSize(whichSuperModuleSize)
{
  // CONSTRUCTOR
  mFastOrs.resize(96);
}
//____________________________________________________________________________
void Patches::init()
{
  mIndexMapPatch.resize(77);
  for (auto IndexMapPatch : mIndexMapPatch) {
    (std::get<1>(IndexMapPatch)).resize(mPatchSize * mPatchSize);
  }
  mFiredFastOrIndexMapPatch.resize(77);
  for (auto FiredFastOrIndexMapPatch : mFiredFastOrIndexMapPatch) {
    (std::get<1>(FiredFastOrIndexMapPatch)).resize(mPatchSize * mPatchSize);
  }
  mADCvalues.resize(77);
  mTimesum.resize(77);
  mPatchIDSeedFastOrIDs.resize(77);
  mPreviousTimebinADCvalue.resize(77);
  mFastOrs.resize(96);
  for (auto FastOr : mFastOrs) {
    FastOr.init();
  }
  assignSeedModuleToAllPatches();
  assignModulesToAllPatches();
}
//____________________________________________________________________________
void Patches::clear()
{
  mIndexMapPatch.clear();
  mFiredFastOrIndexMapPatch.clear();
  mADCvalues.clear();
  mTimesum.clear();
  mPatchIDSeedFastOrIDs.clear();
  mPreviousTimebinADCvalue.clear();
}
//____________________________________________________________________________
void Patches::assignSeedModuleToPatchWithSTUIndexingFullModule(int& patchID)
{
  if (mWhichSide == 0) {
    // A side
    int rowSeed = patchID / 7;
    int columnSeed = patchID % 7;
    int SeedID = rowSeed + patchID;
    std::get<1>(mPatchIDSeedFastOrIDs[patchID]) = SeedID;
  } else if (mWhichSide == 1) {
    // C side
    int rowSeed = patchID / 7;
    int columnSeed = patchID % 7;
    int SeedID = 95 - (rowSeed + patchID);
    std::get<1>(mPatchIDSeedFastOrIDs[patchID]) = SeedID;
  }
}
//____________________________________________________________________________
void Patches::assignSeedModuleToPatchWithSTUIndexingOneThirdModule(int& patchID)
{
  if (mWhichSide == 0) {
    // A side
    int rowSeed = patchID / 23;
    int columnSeed = patchID % 23;
    int SeedID = rowSeed + patchID;
    std::get<1>(mPatchIDSeedFastOrIDs[patchID]) = SeedID;
  } else if (mWhichSide == 1) {
    // C side
    int rowSeed = patchID / 23;
    int columnSeed = patchID % 23;
    int SeedID = 95 - (rowSeed + patchID);
    std::get<1>(mPatchIDSeedFastOrIDs[patchID]) = SeedID;
  }
}
//____________________________________________________________________________
void Patches::assignSeedModuleToAllPatches()
{
  if (mWhichSuperModuleSize == 0) {
    // Full Size
    for (int i = 0; i < 77; i++) {
      std::get<0>(mPatchIDSeedFastOrIDs[i]) = i;
      assignSeedModuleToPatchWithSTUIndexingFullModule(i);
    }
  } else if (mWhichSuperModuleSize == 1) {
    // One third Size
    for (int i = 0; i < 69; i++) {
      std::get<0>(mPatchIDSeedFastOrIDs[i]) = i;
      assignSeedModuleToPatchWithSTUIndexingOneThirdModule(i);
    }
  }
}
//____________________________________________________________________________
void Patches::assignModulesToAllPatches()
{
  if (mWhichSuperModuleSize == 0) {
    // Full Size
    for (int i = 0; i < 77; i++) {
      // std::get<0>(mIndexMapPatch[i]) = i;
      // std::get<0>(mFiredFastOrIndexMapPatch[i]) = i;
      // std::get<0>(mADCvalues[i]) = i;
      // std::get<0>(mPreviousTimebinADCvalue[i]) = i;
      auto& mIndexMapPatchID            = std::get<0>(mIndexMapPatch[i]);
      auto& mFiredFastOrIndexMapPatchID = std::get<0>(mFiredFastOrIndexMapPatch[i]);
      auto& mADCvaluesID                = std::get<0>(mADCvalues[i]);
      auto& mPreviousTimebinADCvalueID  = std::get<0>(mPreviousTimebinADCvalue[i]);
      mIndexMapPatchID            = i;
      mFiredFastOrIndexMapPatchID = i;
      mADCvaluesID                = i;
      mPreviousTimebinADCvalueID  = i;
      int SeedID = std::get<1>(mPatchIDSeedFastOrIDs[i]);
      // assigning a square
      if (mWhichSide == 0) {
        // A side
        for (int k = 0; k < mPatchSize; k++) {   // row advancement
          for (int l = 0; l < mPatchSize; l++) { // column advancement
            auto& IndexMapPatch = std::get<1>(mIndexMapPatch[i]);
            IndexMapPatch.push_back(SeedID + k + l * 8);
            // (std::get<1>(mIndexMapPatch[i])).push_back(SeedID + k + l * 8);
          }
        }
      } else if (mWhichSide == 1) {
        // C side
        for (int k = 0; k < mPatchSize; k++) {   // row advancement
          for (int l = 0; l < mPatchSize; l++) { // column advancement
            auto& IndexMapPatch = std::get<1>(mIndexMapPatch[i]);
            IndexMapPatch.push_back(SeedID - k - l * 8);
            // (std::get<1>(mIndexMapPatch[i])).push_back(SeedID - k - l * 8);
          }
        }
      }
    }
  } else if (mWhichSuperModuleSize == 1) {
    // One third Size
    for (int i = 0; i < 69; i++) {
      // std::get<0>(mIndexMapPatch[i]) = i;
      // std::get<0>(mFiredFastOrIndexMapPatch[i]) = i;
      // std::get<0>(mADCvalues[i]) = i;
      // std::get<0>(mPreviousTimebinADCvalue[i]) = i;
      auto& mIndexMapPatchID            = std::get<0>(mIndexMapPatch[i]);
      auto& mFiredFastOrIndexMapPatchID = std::get<0>(mFiredFastOrIndexMapPatch[i]);
      auto& mADCvaluesID                = std::get<0>(mADCvalues[i]);
      auto& mPreviousTimebinADCvalueID  = std::get<0>(mPreviousTimebinADCvalue[i]);
      mIndexMapPatchID            = i;
      mFiredFastOrIndexMapPatchID = i;
      mADCvaluesID                = i;
      mPreviousTimebinADCvalueID  = i;
      int SeedID = std::get<1>(mPatchIDSeedFastOrIDs[i]);
      // assigning a square
      if (mWhichSide == 0) {
        // A side
        for (int k = 0; k < mPatchSize; k++) {   // row advancement
          for (int l = 0; l < mPatchSize; l++) { // column advancement
            auto& IndexMapPatch = std::get<1>(mIndexMapPatch[i]);
            IndexMapPatch.push_back(SeedID + k + l * 24);
            // (std::get<1>(mIndexMapPatch[i])).push_back(SeedID + k + l * 24);
          }
        }
      } else if (mWhichSide == 1) {
        // C side
        for (int k = 0; k < mPatchSize; k++) {   // row advancement
          for (int l = 0; l < mPatchSize; l++) { // column advancement
            auto& IndexMapPatch = std::get<1>(mIndexMapPatch[i]);
            IndexMapPatch.push_back(SeedID - k - l * 8);
            // (std::get<1>(mIndexMapPatch[i])).push_back(SeedID - k - l * 8);
          }
        }
      }
    }
  }
}
//________________________________________________________
void Patches::updateADC()
{
  for (auto& patch : mADCvalues) {
    double integralPreviousADC = 0;
    auto& ADCvalues = std::get<1>(patch);
    auto& PatchID = std::get<0>(patch);
    if (ADCvalues.size() == 4) {
      std::get<1>(mPreviousTimebinADCvalue[PatchID]) = ADCvalues.front();
      ADCvalues.erase(ADCvalues.begin());
      // LOG(info) << "DIG SIMONE updateADC in Patches: size = 4";
    } else {
      // LOG(info) << "DIG SIMONE updateADC in Patches: size != 4";
    }
    double integralADCnew = 0;
    auto fastorattempt = std::get<1>(mIndexMapPatch[PatchID]);
    // if( fastorattempt.size() != 0 )LOG(info) << "DIG SIMONE updateADC in Patches: fastorattempt.size() = " << fastorattempt.size();

    for (auto FastOrs : std::get<1>(mIndexMapPatch[PatchID])) {
      auto elem = mFastOrs[FastOrs].mADCvalues;
      if (elem.size() == 0){
        // LOG(info) << "DIG SIMONE updateADC in Patches: elem.size() == 0 ";
        continue;
      }  
      auto it = elem.end() - 1;
      auto pointedvalue = *it;
      // LOG(info) << "DIG SIMONE updateADC in Patches: pointedvalue = " << pointedvalue;
      integralADCnew += pointedvalue;
    }
    // LOG(info) << "DIG SIMONE updateADC in Patches: integralADCnew = " << integralADCnew;
    ADCvalues.push_back(integralADCnew);

    // Saving the timesum
    auto& CurrentPatchTimesum = std::get<1>(mTimesum[PatchID]);
    if (CurrentPatchTimesum.size() == 4) {
      CurrentPatchTimesum.erase(CurrentPatchTimesum.begin());
    }
    double IntegralADCvalues = 0;
    for (auto ADC : ADCvalues) {
      IntegralADCvalues += ADC;
    }
    CurrentPatchTimesum.push_back(IntegralADCvalues);
  }
}