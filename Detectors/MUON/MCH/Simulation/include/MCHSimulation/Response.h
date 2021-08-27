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

#ifndef O2_MCH_SIMULATION_RESPONSE_H_
#define O2_MCH_SIMULATION_RESPONSE_H_

#include "DataFormatsMCH/Digit.h"
#include "MCHSimulation/Detector.h"
#include "MCHSimulation/Hit.h"

namespace o2
{
namespace mch
{

enum class Station {
  Type1,
  Type2345
};

class Response
{
 public:
  Response(Station station);
  ~Response() = default;

  float getQspreadX() const { return mQspreadX; };
  float getQspreadY() const { return mQspreadY; };
  float getChargeThreshold() const { return mChargeThreshold; };

  /** Converts energy deposition into a charge.
   *
   * @param edepos deposited energy from Geant (in GeV)
   * @returns an equivalent charge (roughyl in ADC units)
   *
   */
  float etocharge(float edepos) const;

  /** Compute the charge fraction in a rectangle area for a unit charge
   * occuring at position (0,0)
   *
   * @param xmin, xmax, ymin, ymax coordinates (in cm) defining the area
   */
  double chargePadfraction(float xmin, float xmax, float ymin, float ymax) const;

  float getAnod(float x) const;
  float chargeCorr() const;

  bool isAboveThreshold(float charge) const { return charge > mChargeThreshold; };
  float getSigmaIntegration() const { return mSigmaIntegration; };

 private:
  double chargefrac1d(float min, float max, double k2, double sqrtk3, double k4) const;

  //parameter for station number
  Station mStation;
  //proper parameter in aliroot in AliMUONResponseFactory.cxx
  float mQspreadX; //charge spread in cm
  float mQspreadY;

  //ChargeSlope for Station 2-5
  float mChargeSlope;
  const float mChargeCorr = 0.11; // number from line 122
  //of AliMUONResponseFactory.cxx
  //AliMUONResponseV0.h: amplitude of charge correlation on 2 cathods, is RMS of ln(q1/q2)

  float mChargeThreshold = 1e-4;
  float mSigmaIntegration;

  double mK2x;
  double mSqrtK3x;
  double mK4x;
  double mK2y;
  double mSqrtK3y;
  double mK4y;

  float mInversePitch; // anode-cathode Pitch in 1/cm
  float mPitch;
};
} // namespace mch
} // namespace o2
#endif
