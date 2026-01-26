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

/// \file DigiParams.cxx
/// \brief Implementation of the ITS digitization steering params

#include <fairlogger/Logger.h> // for LOG
#include "ITSMFTSimulation/DigiParams.h"
#include <cassert>

ClassImp(o2::itsmft::DigiParams);

using namespace o2::itsmft;

DigiParams::DigiParams()
{
  // make sure the defaults are consistent
  setNSimSteps(mNSimSteps);
}

void DigiParams::setROFrameLength(float lNS, int layer)
{
  // set ROFrame length in nanosecongs
  assert(lNS > 1.f);
  if (layer < 0) {
    mROFrameLength = lNS;
    mROFrameLengthInv = 1.f / mROFrameLength;
  } else {
    mROFrameLayerLength.push_back(lNS);
    mROFrameLayerLengthInv.push_back(1.f / lNS);
  }
}

void DigiParams::setNSimSteps(int v)
{
  // set number of sampling steps in silicon
  mNSimSteps = v > 0 ? v : 1;
  mNSimStepsInv = 1.f / mNSimSteps;
}

void DigiParams::setChargeThreshold(int v, float frac2Account)
{
  // set charge threshold for digits creation and its fraction to account
  // contribution from single hit
  mChargeThreshold = v;
  mMinChargeToAccount = v * frac2Account;
  if (mMinChargeToAccount < 0 || mMinChargeToAccount > mChargeThreshold) {
    mMinChargeToAccount = mChargeThreshold;
  }
  LOG(info) << "Set Alpide charge threshold to " << mChargeThreshold
            << ", single hit will be accounted from " << mMinChargeToAccount
            << " electrons";
}

//______________________________________________
void DigiParams::print() const
{
  LOGF(info, "Alpide digitization params:");
  LOGF(info, "Continuous readout               : %s", mIsContinuous ? "ON" : "OFF");
  if (withStaggering()) {
    for (int i{0}; i < (int)mROFrameLayerLengthInBC.size(); ++i) {
      LOGF(info, " Readout Frame Layer:%d Length(ns)[BC]      : %f [%d]", i, mROFrameLayerLength[i], mROFrameLayerLengthInBC[i]);
      LOGF(info, "Strobe delay Layer %d (ns)                : %f", i, mStrobeDelay);
      LOGF(info, "Strobe length Layer %d (ns)               : %f", i, mStrobeLength);
    }
  } else {
    LOGF(info, "Readout Frame Length(ns)         : %f", mROFrameLength);
    LOGF(info, "Strobe delay (ns)                : %f", mStrobeDelay);
    LOGF(info, "Strobe length (ns)               : %f", mStrobeLength);
  }
  LOGF(info, "Threshold (N electrons)          : %d", mChargeThreshold);
  LOGF(info, "Min N electrons to account       : %d", mMinChargeToAccount);
  LOGF(info, "Number of charge sharing steps   : %d", mNSimSteps);
  LOGF(info, "ELoss to N electrons factor      : %e", mEnergyToNElectrons);
  LOGF(info, "Noise level per pixel            : %e", mNoisePerPixel);
  LOGF(info, "Charge time-response:");
  mSignalShape.print();
}
