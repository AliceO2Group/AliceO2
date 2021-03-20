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
  float getFCtoADC() const { return mFCtoADC; };
  float getChargeThreshold() const { return mChargeThreshold; };
  float getInverseChargeThreshold() const { return mInverseChargeThreshold; };
  float etocharge(float edepos);
  double chargePadfraction(float xmin, float xmax, float ymin, float ymax);
  double chargefrac1d(float min, float max, double k2, double sqrtk3, double k4);
  unsigned long response(unsigned long adc);
  float getAnod(float x);
  float chargeCorr();
  bool aboveThreshold(float charge) { return charge > mChargeThreshold; };
  float getSigmaIntegration() const { return mSigmaIntegration; };
  bool getIsSampa() { return mSampa; };
  void setIsSampa(bool isSampa = true) { mSampa = isSampa; };

 private:
  //setter to get Aliroot-readout-chain or Run 3 (Sampa) one
  bool mSampa = true;

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
  float mInverseChargeThreshold = 10000.;
  //AliMUONResponseV0.cxx constr.
  //"charges below this threshold are 0"
  float mFCtoADC = 1 / (0.61 * 1.25 * 0.2);
  float mADCtoFC = 0.61 * 1.25 * 0.2;
  //transitions between fc and ADD
  //from AliMUONResponseV0.cxx
  //equals (for Aliroo) AliMUONConstants::DefaultADC2MV()*AliMUONConstants::DefaultA0()*AliMUONConstants::DefaultCapa()
  //for the moment not used since directly transition into ADC

  //Mathieson parameter: NIM A270 (1988) 602-603
  //should be a common place for MCH
  // Mathieson parameters from L.Kharmandarian's thesis, page 190
  //  fKy2 = TMath::Pi() / 2. * (1. - 0.5 * fSqrtKy3);//AliMUONMathieson::SetSqrtKx3AndDeriveKx2Kx4(Float_t SqrtKx3)
  //  Float_t cy1 = fKy2 * fSqrtKy3 / 4. / TMath::ATan(Double_t(fSqrtKy3));
  //  fKy4 = cy1 / fKy2 / fSqrtKy3; //this line from AliMUONMathieson::SetSqrtKy3AndDeriveKy2Ky4
  //why this multiplicitation before again division? any number small compared to Float precision?

  float mSigmaIntegration;

  double mK2x;
  double mSqrtK3x;
  double mK4x;
  double mK2y;
  double mSqrtK3y;
  double mK4y;

  //anode-cathode Pitch in 1/cm
  float mInversePitch;
  float mPitch;
  //maximal bit number
  int mMaxADC = (1 << 12) - 1;
};
} // namespace mch
} // namespace o2
#endif
