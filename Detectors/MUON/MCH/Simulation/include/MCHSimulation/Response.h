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
  Response() =default;
  
  ~Response() = default;
  Float_t getQspreadX(){ return mQspreadX;};
  Float_t getQspreadY(){ return mQspreadY;};
  Float_t getChargeSat(){ return mChargeSat;};
  Float_t getChargeThreshold(){ return mChargeThreshold;};
  Float_t etocharge(Float_t edepos);  
  Double_t chargePad(Float_t x, Float_t y, Float_t xmin, Float_t xmax, Float_t ymin, Float_t ymax, Int_t detID, Float_t charge);
  Double_t response(Float_t charge, Int_t detID);
  Float_t getAnod(Float_t x, Int_t detID);
Float_t chargeCorr();

 private:  
  //proper parameter in aliroot in AliMUONResponseFactory.cxx
  //to be discussed n-sigma to be put, use detID to choose value?
  //anything in segmentation foreseen?
  //seem to be only two different values (st. 1 and st. 2-5)...overhead limited
  //any need for separate values as in old code? in principle not...I think
  const Float_t mQspreadX = 0.144; //charge spread in cm
  const Float_t mQspreadY = 0.144;

  //ChargeSlope for Station 2-5
  //to be done: ChargeSlope 10 for station 1
  const Float_t mChargeSlope = 25;//why float in Aliroot?
  const Float_t mChargeCorr = 0.11;// number from line 122
  //of AliMUONResponseFactory.cxx

  const Float_t mChargeThreshold= 1e-4;
  //AliMUONResponseV0.cxx constr.
  const Float_t mChargeSat=0.61*1.25*0.2;
  //from AliMUONResponseV0.cxx
  //equals AliMUONConstants::DefaultADC2MV()*AliMUONConstants::DefaultA0()*AliMUONConstants::DefaultCapa()
  
  //Mathieson parameter: NIM A270 (1988) 602-603 
  //should be a common place for MCH
  
  //difference made between param for x and y
  //why needed? should be symmetric...
  //just take different
  //Station 1 first entry, Station 2-5 second entry
  // Mathieson parameters from L.Kharmandarian's thesis, page 190
   //  fKy2 = TMath::Pi() / 2. * (1. - 0.5 * fSqrtKy3);
  //  Float_t cy1 = fKy2 * fSqrtKy3 / 4. / TMath::ATan(Double_t(fSqrtKy3));
  //  fKy4 = cy1 / fKy2 / fSqrtKy3; //this line from AliMUONMathieson::SetSqrtKy3AndDeriveKy2Ky4
  //why this multiplicitation before again division? any number small compared to Float precision?
  const Double_t mK2x[2] = {1.021026,1.010729};
  const Double_t mSqrtK3x[2] = {0.7000,0.7131};
  const Double_t mK4x[2]     = {0.0,0.0};
   const Double_t mK2y[2] = {0.9778207,0.970595};
  const Double_t mSqrtK3y[2] = {0.7550,0.7642};
  const Double_t mK4y[2]     = {0.0,0.0};
  //chargecorr 0.11
  
  //anode-cathode Pitch in 1/cm
  //Station 1 first entry, Station 2-5 second entry
  const Float_t mInversePitch[2] ={1./0.21,1./0.25};


  

};

} // namespace mch
} // namespace o2
#endif
