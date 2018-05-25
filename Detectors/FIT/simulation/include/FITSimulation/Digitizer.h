// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FIT_DIGITIZER_H_
#define ALICEO2_FIT_DIGITIZER_H_

#include "FITBase/Digit.h"
#include "FITSimulation/Detector.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "FITSimulation/MCLabel.h"

namespace o2
{
namespace fit
{
class Digitizer
{
 public:
  Digitizer(Int_t mode = 0) : mMode(mode), mTimeFrameCurrent(0) { initParameters(); };
  ~Digitizer() = default;

  void process(const std::vector<HitType>* hits, std::vector<Digit>* digits);

  void initParameters();
  // void printParameters();
  void setEventTime(double value) { mEventTime = value; }
  void setEventID(Int_t id) { mEventID = id; }
  void setMCTruthContainer(o2::dataformats::MCTruthContainer<o2::fit::MCLabel>* truthcontainer)
  {
    mMCTruthContainer = truthcontainer;
  }

  Int_t getCurrentTimeFrame() const { return mTimeFrameCurrent; }
  void setCurrentTimeFrame(Double_t value) { mTimeFrameCurrent = value; }

 void init();
  void finish();


 private:
  // digit info
  std::vector<Digit>* mDigits;
  o2::dataformats::MCTruthContainer<o2::fit::MCLabel>* mMCTruthContainer =
    nullptr; ///< Array for MCTruth information associated to digits in mDigitsArrray. Passed from the digitization

  void addDigit(Double_t time, Int_t channel, Double_t cfd, Int_t amp, Int_t bc, Int_t trackID);
  // parameters
  Int_t mMode;
  Int_t mTimeFrameCurrent;
  Double_t mEventTime;
  Int_t mEventID = 0;
  Int_t mSrcID = 0;

  ClassDefNV(Digitizer, 1);
};
} // namespace fit
} // namespace o2

#endif
