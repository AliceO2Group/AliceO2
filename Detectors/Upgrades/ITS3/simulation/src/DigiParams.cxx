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
/// \brief Implementation of the ITS3 digitization steering params

#include "Framework/Logger.h"
#include "ITS3Simulation/DigiParams.h"
#include <cstdio>

ClassImp(o2::its3::DigiParams);

namespace o2::its3
{

DigiParams::DigiParams()
{
  // make sure the defaults are consistent
  setIBNSimSteps(mIBNSimSteps);
}

void DigiParams::setIBNSimSteps(int v)
{
  // set number of sampling steps in silicon
  mIBNSimSteps = v > 0 ? v : 1;
  mIBNSimStepsInv = 1.f / mIBNSimSteps;
}

void DigiParams::setIBChargeThreshold(int v, float frac2Account)
{
  // set charge threshold for digits creation and its fraction to account
  // contribution from single hit
  mIBChargeThreshold = v;
  mIBMinChargeToAccount = v * frac2Account;
  if (mIBMinChargeToAccount < 0 || mIBMinChargeToAccount > mIBChargeThreshold) {
    mIBMinChargeToAccount = mIBChargeThreshold;
  }
  LOG(info) << "Set Mosaix charge threshold to " << mIBChargeThreshold
            << ", single hit will be accounted from " << mIBMinChargeToAccount
            << " electrons";
}

void DigiParams::print() const
{
  // print settings
  printf("ITS3 DigiParams settings:\n");
  printf("Continuous readout                   : %s\n", isContinuous() ? "ON" : "OFF");
  printf("Readout Frame Length(ns)             : %f\n", getROFrameLength());
  printf("Strobe delay (ns)                    : %f\n", getStrobeDelay());
  printf("Strobe length (ns)                   : %f\n", getStrobeLength());
  printf("IB Threshold (N electrons)           : %d\n", getIBChargeThreshold());
  printf("OB Threshold (N electrons)           : %d\n", getChargeThreshold());
  printf("Min N electrons to account for IB    : %d\n", getIBMinChargeToAccount());
  printf("Min N electrons to account for OB    : %d\n", getMinChargeToAccount());
  printf("Number of charge sharing steps of IB : %d\n", getIBNSimSteps());
  printf("Number of charge sharing steps of OB : %d\n", getNSimSteps());
  printf("ELoss to N electrons factor          : %e\n", getEnergyToNElectrons());
  printf("Noise level per pixel of IB          : %e\n", getIBNoisePerPixel());
  printf("Noise level per pixel of OB          : %e\n", getNoisePerPixel());
  printf("Charge time-response:\n");
  getSignalShape().print();
}

void DigiParams::setIBSimResponse(const o2::itsmft::AlpideSimResponse* resp)
{
  if (!resp) {
    LOGP(fatal, "cannot set response from nullptr");
  }
  mIBSimResponseExt = resp;
  mIBSimResponse = std::make_unique<o2::its3::ChipSimResponse>(mIBSimResponseExt);
  mIBSimResponse->computeCentreFromData();
}

} // namespace o2::its3
