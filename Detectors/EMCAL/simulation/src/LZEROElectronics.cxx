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
