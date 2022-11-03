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

void DigiParams::setROFrameLength(float lNS)
{
  // set ROFrame length in nanosecongs
  mROFrameLength = lNS;
  assert(mROFrameLength > 1.);
  mROFrameLengthInv = 1. / mROFrameLength;
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
  // print settings
  printf("Alpide digitization params:\n");
  printf("Continuous readout             : %s\n", mIsContinuous ? "ON" : "OFF");
  printf("Readout Frame Length(ns)       : %f\n", mROFrameLength);
  printf("Strobe delay (ns)              : %f\n", mStrobeDelay);
  printf("Strobe length (ns)             : %f\n", mStrobeLength);
  printf("Threshold (N electrons)        : %d\n", mChargeThreshold);
  printf("Min N electrons to account     : %d\n", mMinChargeToAccount);
  printf("Number of charge sharing steps : %d\n", mNSimSteps);
  printf("ELoss to N electrons factor    : %e\n", mEnergyToNElectrons);
  printf("Noise level per pixel          : %e\n", mNoisePerPixel);
  printf("Charge time-response:\n");
  mSignalShape.print();
}
