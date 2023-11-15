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
//_____________________________________________________________________
float Response::eLossRatio(float logbetagamma) const
{
  // Ratio of particle mean eloss with respect MIP's Khalil Boudjemline, sep 2003, PhD.Thesis and Particle Data Book
  /// copied from aliroot AliMUONv1.cxx
  float eLossRatioParam[5] = {1.02138, -9.54149e-02, +7.83433e-02, -9.98208e-03, +3.83279e-04};
  return eLossRatioParam[0] + eLossRatioParam[1] * logbetagamma + eLossRatioParam[2] * std::pow(logbetagamma, 2) + eLossRatioParam[3] * std::pow(logbetagamma, 3) + eLossRatioParam[4] * std::pow(logbetagamma, 4);
}
//_____________________________________________________________________
float Response::angleEffect10(float elossratio) const
{
  /// Angle effect in tracking chambers at theta =10 degres as a function of ElossRatio (Khalil BOUDJEMLINE sep 2003 Ph.D Thesis) (in micrometers)
  /// copied from aliroot AliMUONv1.cxx
  float angleEffectParam[3] = {1.90691e+02, -6.62258e+01, 1.28247e+01};
  return angleEffectParam[0] + angleEffectParam[1] * elossratio + angleEffectParam[2] * std::pow(elossratio, 2);
}
//_____________________________________________________________________
float Response::angleEffectNorma(float elossratio) const
{
  /// Angle effect: Normalisation form theta=10 degres to theta between 0 and 10 (Khalil BOUDJEMLINE sep 2003 Ph.D Thesis)
  /// Angle with respect to the wires assuming that chambers are perpendicular to the z axis.
  /// copied from aliroot AliMUONv1.cxx
  float angleEffectParam[4] = {4.148, -6.809e-01, 5.151e-02, -1.490e-03};
  return angleEffectParam[0] + angleEffectParam[1] * elossratio + angleEffectParam[2] * std::pow(elossratio, 2) + angleEffectParam[3] * std::pow(elossratio, 3);
}
//_____________________________________________________________________
float Response::magAngleEffectNorma(float angle, float bfield) const
{
  /// Magnetic field effect: Normalisation form theta=16 degres (eq. 10 degrees B=0) to theta between -20 and 20 (Lamia Benhabib jun 2006 )
  /// Angle with respect to the wires assuming that chambers are perpendicular to the z axis.
  /// copied from aliroot AliMUONv1.cxx
  float angleEffectParam[7] = {8.6995, 25.4022, 13.8822, 2.4717, 1.1551, -0.0624, 0.0012};
  float aux = std::abs(angle - angleEffectParam[0] * bfield);
  return 121.24 / ((angleEffectParam[1] + angleEffectParam[2] * std::abs(bfield)) + angleEffectParam[3] * aux + angleEffectParam[4] * std::pow(aux, 2) + angleEffectParam[5] * std::pow(aux, 3) + angleEffectParam[6] * std::pow(aux, 4));
}
