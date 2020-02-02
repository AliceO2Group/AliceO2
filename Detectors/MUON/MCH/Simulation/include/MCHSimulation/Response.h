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

#include "MCHBase/Digit.h"
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

  float getQspreadX() { return mQspreadX; };
  float getQspreadY() { return mQspreadY; };
  float getChargeSat() { return mChargeSat; };
  float getChargeThreshold() { return mChargeThreshold; };
  float etocharge(float edepos);
  double chargePadfraction(float xmin, float xmax, float ymin, float ymax);
  double chargefrac1d(float min, float max, double k2, double sqrtk3, double k4);
  unsigned long response(float charge);
  float getAnod(float x);
  float chargeCorr();

 private:
  //parameter for station number
  Station mStation;
  //proper parameter in aliroot in AliMUONResponseFactory.cxx
  const float mQspreadX = 0.144; //charge spread in cm
  const float mQspreadY = 0.144;

  //ChargeSlope for Station 2-5
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
  // Mathieson parameters from L.Kharmandarian's thesis, page 190
  //  fKy2 = TMath::Pi() / 2. * (1. - 0.5 * fSqrtKy3);//AliMUONMathieson::SetSqrtKx3AndDeriveKx2Kx4(Float_t SqrtKx3)
  //  Float_t cy1 = fKy2 * fSqrtKy3 / 4. / TMath::ATan(Double_t(fSqrtKy3));
  //  fKy4 = cy1 / fKy2 / fSqrtKy3; //this line from AliMUONMathieson::SetSqrtKy3AndDeriveKy2Ky4
  //why this multiplicitation before again division? any number small compared to Float precision?
  double mK2x = 0.0;
  double mSqrtK3x = 0.0;
  double mK4x = 0.0;
  double mK2y = 0.0;
  double mSqrtK3y = 0.0;
  double mK4y = 0.0;

  //anode-cathode Pitch in 1/cm
  float mInversePitch = 0.0;
};
} // namespace mch
} // namespace o2
#endif
