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

#include "EMCALSimulation/SimParam.h"
#include <iostream>
#include <TMath.h>

O2ParamImpl(o2::emcal::SimParam);

using namespace o2::emcal;

std::ostream& operator<<(std::ostream& stream, const o2::emcal::SimParam& s)
{
  s.PrintStream(stream);
  return stream;
}

void SimParam::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL::SimParam.mDigitThreshold = " << mDigitThreshold;
  stream << "\nEMCAL::SimParam.mMeanPhotonElectron = " << mMeanPhotonElectron;
  stream << "\nEMCal::SimParam.mGainFluctuations = " << mGainFluctuations;
  stream << "\nEMCal::SimParam.mPinNoise = " << mPinNoise;
  stream << "\nEMCal::SimParam.mPinNoiseLG = " << mPinNoiseLG;
  stream << "\nEMCal::SimParam.mTimeNoise = " << mTimeNoise;
  stream << "\nEMCal::SimParam.mTimeDelay = " << mTimeDelay;
  stream << "\nEMCal::SimParam.mTimeDelayFromOCDB = " << ((mTimeDelayFromOCDB) ? "true" : "false");
  stream << "\nEMCal::SimParam.mTimeResolutionPar0 = " << mTimeResolutionPar0;
  stream << "\nEMCal::SimParam.mTimeResolutionPar1 = " << mTimeResolutionPar1;
  stream << "\nEMCal::SimParam.mTimeResponseTau = " << mTimeResponseTau;
  stream << "\nEMCal::SimParam.mTimeResponsePower = " << mTimeResponsePower;
  stream << "\nEMCal::SimParam.mTimeResponseThreshold = " << mTimeResponseThreshold;
  stream << "\nEMCal::SimParam.mNADCEC = " << mNADCEC;
  stream << "\nEMCal::SimParam.mA = " << mA;
  stream << "\nEMCal::SimParam.mB = " << mB;
  stream << "\nEMCal::SimParam.mECPrimThreshold = " << mECPrimThreshold;
  stream << "\nEMCal::SimParam.mSignalDelay = " << mSignalDelay;
  stream << "\nEMCal::SimParam.mTimeWindowStart = " << mTimeWindowStart;
  stream << "\nEMCal::SimParam.mLiveTime = " << mLiveTime;
  stream << "\nEMCal::SimParam.mBusyTime = " << mBusyTime;
  stream << "\nEMCal::SimParam.mPreTriggerTime = " << mPreTriggerTime;
  stream << "\nEMCal::SimParam.mSmearEnergy = " << ((mSmearEnergy) ? "true" : "false");
  stream << "\nEMCal::SimParam.mSimulateTimeResponse = " << ((mSimulateTimeResponse) ? "true" : "false");
  stream << "\nEMCal::SimParam.mRemoveDigitsBelowThreshold = " << ((mRemoveDigitsBelowThreshold) ? "true" : "false");
  stream << "\nEMCal::SimParam.mSimulateNoiseDigits = " << ((mSimulateNoiseDigits) ? "true" : "false");
  stream << "\nEMCal::SimParam.mDisablePileup = " << ((mDisablePileup) ? "true" : "false");
  stream << "\nEMCal::SimParam.mSimulateL1Phase = " << ((mSimulateL1Phase) ? "true" : "false");
}

Double_t SimParam::getTimeResolution(Double_t energy) const
{
  Double_t res = -1;
  if (energy > 0) {
    res = TMath::Sqrt(mTimeResolutionPar0 + mTimeResolutionPar1 / (energy * energy));
  }

  // parametrization above is for ns. Convert to seconds:
  // return res*1e-9;
  return res;
}
