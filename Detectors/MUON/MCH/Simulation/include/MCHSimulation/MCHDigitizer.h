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

  //detector segmentation handler to convert pad-id to coordinates and vice versa
  Segmentation mSegbend[nNdE];
  Segmentation mSegnon[nNdE];
  
  Int_t processHit(const HitType& hit, Double_t event_time);
  


  
  ClassDefNV(MCHDigitizer, 1);
};
} // namespace mch
} // namespace o2
#endif
