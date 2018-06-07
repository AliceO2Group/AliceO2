// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigiParams.cxx
/// \brief Implementation of the ITS digitization steering params

#include "FairLogger.h" // for LOG
#include "ITSMFTSimulation/DigiParams.h"
#include <cassert>

ClassImp(o2::ITSMFT::DigiParams);

using namespace o2::ITSMFT;

void DigiParams::setROFrameLenght(float lNS)
{
  // set ROFrame length in nanosecongs
  mROFrameLenght = lNS;
  assert(mROFrameLenght > 1.);
  mROFrameLenghtInv = 1. / mROFrameLenght;
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
  LOG(INFO) << "Set Alpide charge threshold to " << mChargeThreshold
            << ", single hit will be accounted from " << mMinChargeToAccount
            << " electrons" << FairLogger::endl;
}
