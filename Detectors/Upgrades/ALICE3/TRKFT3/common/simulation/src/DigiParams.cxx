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
/// \brief Implementation of the TRK digitization steering params. Based on the ITS2 code.

#include <cassert>
#include "Framework/Logger.h"
#include "TRKFT3Simulation/DigiParams.h"
#include "TRKFT3Simulation/ChipSimResponse.h"

using namespace o2::trkft3;

template <int DetIDV>
DigiParams<DetIDV>::DigiParams()
{
  // make sure the defaults are consistent
  setNSimSteps(mNSimSteps);
}

template <int DetIDV>
void DigiParams<DetIDV>::setROFrameLength(float lNS, int layer)
{
  // set ROFrame length in nanosecongs
  mROFrameLayerLength[layer] = lNS;
  assert(mROFrameLayerLength[layer] > 1.);
  mROFrameLayerLengthInv[layer] = 1. / mROFrameLayerLength[layer];
}

template <int DetIDV>
void DigiParams<DetIDV>::setNSimSteps(int v)
{
  // set number of sampling steps in silicon
  mNSimSteps = v > 0 ? v : 1;
  mNSimStepsInv = 1.f / mNSimSteps;
}

template <int DetIDV>
void DigiParams<DetIDV>::setChargeThreshold(int v, float frac2Account)
{
  // set charge threshold for digits creation and its fraction to account
  // contribution from single hit
  mChargeThreshold = v;
  mMinChargeToAccount = v * frac2Account;
  if (mMinChargeToAccount < 0 || mMinChargeToAccount > mChargeThreshold) {
    mMinChargeToAccount = mChargeThreshold;
  }
  LOG(info) << "Set charge threshold to " << mChargeThreshold
            << ", single hit will be accounted from " << mMinChargeToAccount
            << " electrons";
}

//______________________________________________
template <int DetIDV>
void DigiParams<DetIDV>::print() const
{
  // print settings
  printf("%s digitization params:\n", o2::detectors::DetID::getName(DetIDV));
  printf("Threshold (N electrons)        : %d\n", mChargeThreshold);
  printf("Min N electrons to account     : %d\n", mMinChargeToAccount);
  printf("Number of charge sharing steps : %d\n", mNSimSteps);
  printf("ELoss to N electrons factor    : %e\n", mEnergyToNElectrons);
  printf("Noise level per pixel          : %e\n", mNoisePerPixel);
  printf("Charge time-response:\n");
  mSignalShape.print();
}

template <int DetIDV>
void DigiParams<DetIDV>::setResponse(const o2::itsmft::AlpideSimResponse* resp)
{
  LOG(debug) << "Response function data path: " << resp->getDataPath();
  LOG(debug) << "Response function info: ";
  // resp->print();
  if (!resp) {
    LOGP(fatal, "cannot set response function from null");
  }
  mResponse = std::make_unique<o2::trkft3::ChipSimResponse>(resp);
}

template class o2::trkft3::DigiParams<o2::detectors::DetID::TRK>;
template class o2::trkft3::DigiParams<o2::detectors::DetID::FT3>;
