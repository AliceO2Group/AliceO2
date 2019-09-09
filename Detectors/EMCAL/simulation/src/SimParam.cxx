// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALSimulation/SimParam.h"
#include <iostream>
#include <TMath.h>

using namespace o2::emcal;

SimParam* SimParam::mSimParam = nullptr;

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
  stream << "\nEMCal::SimParam.mTimeNoise = " << mTimeNoise;
  stream << "\nEMCal::SimParam.mTimeDelay = " << mTimeDelay;
  stream << "\nEMCal::SimParam.mTimeDelayFromCDB = " << ((mTimeDelayFromCDB) ? "true" : "false");
  stream << "\nEMCal::SimParam.mTimeResolutionPar0 = " << mTimeResolutionPar0;
  stream << "\nEMCal::SimParam.mTimeResolutionPar1 = " << mTimeResolutionPar1;
  stream << "\nEMCal::SimParam.mTimeResponseTau = " << mTimeResponseTau;
  stream << "\nEMCal::SimParam.mTimeResponsePower = " << mTimeResponsePower;
  stream << "\nEMCal::SimParam.mTimeResponseThreshold = " << mTimeResponseThreshold;
  stream << "\nEMCal::SimParam.mNADCEC = " << mNADCEC;
  stream << "\nEMCal::SimParam.mA = " << mA;
  stream << "\nEMCal::SimParam.mB = " << mB;
  stream << "\nEMCal::SimParam.mECPrimThreshold = " << mECPrimThreshold;
}

Double_t SimParam::getTimeResolution(Double_t energy) const
{
  Double_t res = -1;
  if (energy > 0) {
    res = TMath::Sqrt(mTimeResolutionPar0 + mTimeResolutionPar1 / (energy * energy));
  }

  // parametrization above is for ns. Convert to seconds:
  //return res*1e-9;
  return res;
}
