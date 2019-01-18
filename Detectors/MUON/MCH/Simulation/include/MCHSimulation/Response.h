// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_SIMULATION_RESPONSE_H_
#define O2_MCH_SIMULATION_RESPONSE_H_

#include "MCHSimulation/Digit.h"
#include "MCHSimulation/Detector.h"
#include "MCHSimulation/Hit.h"

namespace o2
{
namespace mch
{

class Response
{
 public:
  Response() = default;

  ~Response() = default;
  float getQspreadX() { return mQspreadX; };
  float getQspreadY() { return mQspreadY; };
  float getChargeSat() { return mChargeSat; };
  float getChargeThreshold() { return mChargeThreshold; };
  float etocharge(float edepos);
  double chargePad(float xmin, float xmax, float ymin, float ymax, int detID, float charge);
  double response(float charge, int detID);
  float getAnod(float x, int detID);
  float chargeCorr();

 private:
  //proper parameter in aliroot in AliMUONResponseFactory.cxx
  //to be discussed n-sigma to be put, use detID to choose value?
  //anything in segmentation foreseen?
  //seem to be only two different values (st. 1 and st. 2-5)...overhead limited
  //any need for separate values as in old code? in principle not...I think
  const float mQspreadX = 0.144; //charge spread in cm
  const float mQspreadY = 0.144;

  //ChargeSlope for Station 2-5
  //to be done: ChargeSlope 10 for station 1
  const float mChargeSlope = 25;  //why float in Aliroot?
  const float mChargeCorr = 0.11; // number from line 122
  //of AliMUONResponseFactory.cxx

  const float mChargeThreshold = 1e-4;
  //AliMUONResponseV0.cxx constr.
  const float mChargeSat = 0.61 * 1.25 * 0.2;
  //from AliMUONResponseV0.cxx
  //equals AliMUONConstants::DefaultADC2MV()*AliMUONConstants::DefaultA0()*AliMUONConstants::DefaultCapa()
  //Mathieson parameter: NIM A270 (1988) 602-603
  //should be a common place for MCH

  //Station 1 first entry, Station 2-5 second entry
  // Mathieson parameters from L.Kharmandarian's thesis, page 190
  //  fKy2 = TMath::Pi() / 2. * (1. - 0.5 * fSqrtKy3);//AliMUONMathieson::SetSqrtKx3AndDeriveKx2Kx4(Float_t SqrtKx3)
  //  Float_t cy1 = fKy2 * fSqrtKy3 / 4. / TMath::ATan(Double_t(fSqrtKy3));
  //  fKy4 = cy1 / fKy2 / fSqrtKy3; //this line from AliMUONMathieson::SetSqrtKy3AndDeriveKy2Ky4
  //why this multiplicitation before again division? any number small compared to Float precision?
  const double mK2x[2] = { 1.021026, 1.010729 };
  const double mSqrtK3x[2] = { 0.7000, 0.7131 };
  const double mK4x[2] = { 0.40934890, 0.40357476 };
  const double mK2y[2] = { 0.9778207, 0.970595 };
  const double mSqrtK3y[2] = { 0.7550, 0.7642 };
  const double mK4y[2] = { 0.38658194, 0.38312571 };

  //anode-cathode Pitch in 1/cm
  //Station 1 first entry, Station 2-5 second entry
  const float mInversePitch[2] = { 1. / 0.21, 1. / 0.25 };
};
} // namespace mch
} // namespace o2
#endif
