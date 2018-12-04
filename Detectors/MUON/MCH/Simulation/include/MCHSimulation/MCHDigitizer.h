// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MCH_MCHDIGITIZER_H_
#define ALICEO2_MCH_MCHDIGITIZER_H_

#include "MCHBase/Digit.h"
#include "MCHSimulation/Detector.h"
#include "MCHSimulation/Hit.h"
#include "MCHMappingInterface/Segmentation.h"
//#include "Mapping/Interface/Segmentation.h"//to be replaced, which one? 


namespace o2
{
namespace mch
{
class MCHDigitizer
{
 public:
  MCHDigitizer(Int_t mode = 0) : mReadoutWindowCurrent(0) { init(); };
  ~MCHDigitizer() = default;

  void init();

  void process(const std::vector<HitType>* hits, std::vector<Digit>* digits);

  Float_t getCharge(Float_t eDep);
  Double_t getXmin(Int_t detID, Double_t hitX);
  Double_t getXmax(Int_t detID, Double_t hitX);
  Double_t getYmin(Int_t detID, Double_t hitY);
  Double_t getYmax(Int_t detID, Double_t hitY);
  
  void setEventTime(double value) { mEventTime = value; }
  void setEventID(Int_t id) { mEventID = id; }
  void setSrcID(Int_t id) { mSrcID = id; }

  void initParameters();

  void fillOutputContainer(std::vector<Digit>& digits);
  void flushOutputContainer(std::vector<Digit>& digits); // flush all residual buffered data

  void setContinuous(bool val) { mContinuous = val; }
  bool isContinuous() const { return mContinuous; }

 private:  
  Double_t mEventTime;
  Int_t mReadoutWindowCurrent;  
  Int_t mEventID = 0;
  Int_t mSrcID = 0;
  
  bool mContinuous = false; 

  //number of detector elements 5(stations)*2(layer per station)* 2(?) +1 (?)
  const Int_t mNdE = 21;
  // digit per pad
  std::vector<Digit> mDigits;

  std::vector<int> mPadIDsbend;
  std::vector<int> mPadIDsnon;

  //detector segmentation handler to convert pad-id to coordinates and vice versa
  Segmentation mSegbend[nNdE];
  Segmentation  mSegnon[nNdE];

  //proper parameter in aliroot in AliMUONResponseFactory.cxx
  //to be discussed n-sigma to be put, use detID to choose value?
  //anything in segmentation foreseen?
  //seem to be only two different values (st. 1 and st. 2-5)...overhead limited
  //any need for separate values as in old code? in principle not...I think
  const Float_t mQspreadX = 0.144; //charge spread in cm
  const Float_t mQspreadY = 0.144;
  
  Int_t processHit(const HitType& hit, Double_t event_time);
  Double_t chargePad(Float_t x, Float_t y, Float_t xmin, Float_t xmax, Float_t ymin, Float_t ymax, Int_t detID, Float_t charge);
  Double_t response(Float_t charge, Int_t detID);

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

  
  //anode-cathode Pitch in 1/cm
  //Station 1 first entry, Station 2-5 second entry
  const Double_t mInversePitch[2] ={1./0.21,1./0.25};


  
  ClassDefNV(MCHDigitizer, 1);
};
} // namespace mch
} // namespace o2
#endif
