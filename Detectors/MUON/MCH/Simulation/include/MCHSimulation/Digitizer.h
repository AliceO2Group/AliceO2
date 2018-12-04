// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MCH_DIGITIZER_H_
#define ALICEO2_MCH_DIGITIZER_H_

#include "MCHBase/DigitBlock.h"// not clear if this is sufficient as structure
#include "MCHSimulation/Detector.h"
#include "MCHSimulation/Hit.h"



namespace o2
{
namespace mch
{
class Digitizer
{
 public:
  Digitizer(Int_t mode = 0) : mReadoutWindowCurrent(0) { init(); };
  ~Digitizer() = default;

  void init();

  void process(const std::vector<HitType>* hits, std::vector<Digit>* digits);

  Float_t getCharge(Float_t eDep); 

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

  
  // digit per pad info
  std::vector<DigitStruct> mDigits;
  
  
  Int_t processHit(const HitType& hit, Double_t event_time);
  


  
  ClassDefNV(Digitizer, 1);
};
} // namespace mch
} // namespace o2
#endif
