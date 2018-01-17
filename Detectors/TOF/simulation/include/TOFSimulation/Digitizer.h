// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TOF_DIGITIZER_H_
#define ALICEO2_TOF_DIGITIZER_H_

#include "TOFBase/Geo.h"
#include "TOFSimulation/Detector.h"

namespace o2
{
namespace tof
{
class Digitizer : public TObject
{
 public:
 Digitizer(Int_t mode = 0) : mMode(mode), mTimeFrameCurrent(0) { initParameters(); };

  ~Digitizer() override = default;

  void digitize();

  void process(const std::vector<HitType>* hits,Double_t event_time=0); 

  void processHit(const HitType &hit,Double_t event_time=0);
  void addDigit(Int_t channel, Float_t time, Float_t x, Float_t z, Float_t charge, Int_t iX, Int_t iZ, Int_t padZfired);
  Float_t getShowerTimeSmeared(Float_t time, Float_t charge);
  Float_t getDigitTimeSmeared(Float_t time, Float_t x, Float_t z, Float_t charge);
  Float_t getCharge(Float_t eDep);
  Bool_t isFired(Float_t x, Float_t z, Float_t charge);
  Float_t getEffX(Float_t x);
  Float_t getEffZ(Float_t z);
  Float_t getFractionOfCharge(Float_t x, Float_t z);

  Int_t getCurrentTimeFrame() const {return mTimeFrameCurrent;}
  void  setCurrentTimeFrame(Double_t value) {mTimeFrameCurrent = value;}

  Int_t getNumDigitLastHit() const { return mNumDigit; }
  Float_t getTimeLastHit(Int_t idigit) const { return mTime[idigit]; }
  Float_t getTotLastHit(Int_t idigit) const { return mTot[idigit]; }
  Int_t getXshift(Int_t idigit) const { return mXshift[idigit]; }
  Int_t getZshift(Int_t idigit) const { return mZshift[idigit]; }

  void setEventTime(double) {};

  void initParameters();
  void printParameters();

  void test(const char* geo = "O2geometry.root");
  void testFromHits(const char* geo = "O2geometry.root", const char* hits = "AliceO2_TGeant3.tof.mc_10_event.root");

 private:
  // parameters
  Int_t mMode;
  Float_t mBound1;
  Float_t mBound2;
  Float_t mBound3;
  Float_t mBound4;
  Float_t mTOFresolution;
  Float_t mShowerResolution;
  Float_t mDigitResolution;
  Float_t mTimeSlope;
  Float_t mTimeDelay;
  Float_t mTimeDelayCorr;
  Float_t mTimeWalkeSlope;
  Float_t mEffCenter;
  Float_t mEffBoundary1;
  Float_t mEffBoundary2;
  Float_t mEffBoundary3;

  // info TOF timewindow
  Int_t mTimeFrameCurrent;

  // keep info of last digitization
  Int_t mNumDigit;  //! number of digits of last hit processed
  Float_t mTime[6]; //! time of digitis in the last hit processed
  Float_t mTot[6];  //! tot of digitis in the last hit processed
  Int_t mXshift[6]; //! shift wrt central pad
  Int_t mZshift[6]; //! shift wrt central pad

  ClassDefOverride(Digitizer, 2);
};
}
}
#endif
