// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FIT_DIGITIZER_H
#define ALICEO2_FIT_DIGITIZER_H

#include "FITBase/Digit.h"
#include "FITSimulation/Detector.h"

namespace o2
{
namespace fit
{
class Digitizer
{
 public:
  Digitizer(Int_t mode = 0) : mMode(mode), mTimeFrameCurrent(0) { initParameters(); };
  ~Digitizer() = default;

  //void process(const std::vector<HitType>* hits, std::vector<Digit>* digits);
  void process(const std::vector<HitType>* hits, Digit* digit);

  void initParameters();
  // void printParameters();
  void setEventTime(double value) { mEventTime = value; }
  void setEventID(Int_t id) { mEventID = id; }
  Int_t getCurrentTimeFrame() const { return mTimeFrameCurrent; }
  void setCurrentTimeFrame(Double_t value) { mTimeFrameCurrent = value; }

  void init();
  void finish();

 private:
  // digit info
  //std::vector<Digit>* mDigits;

  void addDigit(Double_t time, Int_t channel, Double_t cfd, Int_t amp, Int_t bc);
  // parameters
  Int_t mMode;
  Int_t mTimeFrameCurrent;
  Double_t mEventTime; // Initialized in initParameters
  Int_t mEventID = 0;
  Int_t mSrcID = 0;
  Int_t mAmpThreshold; // Initialized in initParameters
  Double_t mLowTime;   // Initialized in initParameters
  Double_t mHighTime;  // Initialized in initParameters
  Double_t mTimeDiffAC = (Geometry::ZdetA - Geometry::ZdetC) * TMath::C();
  ClassDefNV(Digitizer, 1);
};
} // namespace fit
} // namespace o2

#endif
