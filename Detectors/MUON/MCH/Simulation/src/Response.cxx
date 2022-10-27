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

/** @file Response.cxx
 * C++ MCH charge induction and signal generation taken from Aliroot.
 * @author Michael Winn, Laurent Aphecetche
 */

#include "MCHSimulation/Response.h"

#include <cmath>

#include "MCHBase/ResponseParam.h"

#include "TMath.h"
#include "TRandom.h"

using namespace o2::mch;

//_____________________________________________________________________
Response::Response(Station station)
{
  if (station == Station::Type1) {
    mMathieson.setPitch(ResponseParam::Instance().pitchSt1);
    mMathieson.setSqrtKx3AndDeriveKx2Kx4(ResponseParam::Instance().mathiesonSqrtKx3St1);
    mMathieson.setSqrtKy3AndDeriveKy2Ky4(ResponseParam::Instance().mathiesonSqrtKy3St1);
    mPitch = ResponseParam::Instance().pitchSt1;
    mChargeSlope = ResponseParam::Instance().chargeSlopeSt1;
    mChargeSpread = ResponseParam::Instance().chargeSpreadSt1;
  } else {
    mMathieson.setPitch(ResponseParam::Instance().pitchSt2345);
    mMathieson.setSqrtKx3AndDeriveKx2Kx4(ResponseParam::Instance().mathiesonSqrtKx3St2345);
    mMathieson.setSqrtKy3AndDeriveKy2Ky4(ResponseParam::Instance().mathiesonSqrtKy3St2345);
    mPitch = ResponseParam::Instance().pitchSt2345;
    mChargeSlope = ResponseParam::Instance().chargeSlopeSt2345;
    mChargeSpread = ResponseParam::Instance().chargeSpreadSt2345;
  }
  mSigmaIntegration = ResponseParam::Instance().chargeSigmaIntegration;
  mChargeCorr = ResponseParam::Instance().chargeCorrelation;
  mChargeThreshold = ResponseParam::Instance().chargeThreshold;
}

//_____________________________________________________________________
float Response::etocharge(float edepos) const
{
  int nel = int(edepos * 1.e9 / 27.4);
  if (nel == 0) {
    nel = 1;
  }
  float charge = 0.f;
  for (int i = 1; i <= nel; i++) {
    float arg = 0.f;
    do {
      arg = gRandom->Rndm();
    } while (!arg);
    charge -= mChargeSlope * TMath::Log(arg);
  }
  return charge;
}

//_____________________________________________________________________
float Response::getAnod(float x) const
{
  int n = int(x / mPitch);
  float wire = (x > 0) ? n + 0.5 : n - 0.5;
  return wire * mPitch;
}

//_____________________________________________________________________
float Response::chargeCorr() const
{
  return TMath::Exp(gRandom->Gaus(0.0, mChargeCorr / 2.0));
}

//_____________________________________________________________________
uint32_t Response::nSamples(float charge) const
{
  // the main purpose is to the pass the background rejection and signal selection
  // applied in data reconstruction (see MCH/DigitFiltering/src/DigitFilter.cxx).
  // a realistic estimate of nSamples would require a complete simulation of the electronic signal
  double signalParam[3] = {14., 13., 1.5};
  return std::round(std::pow(charge / signalParam[1], 1. / signalParam[2]) + signalParam[0]);
}
