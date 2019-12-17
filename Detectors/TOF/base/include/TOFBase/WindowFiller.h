// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TOF_WINDOWFILLER_H_
#define ALICEO2_TOF_WINDOWFILLER_H_

#include "TOFBase/Geo.h"
#include "TOFBase/Digit.h"
#include "TOFBase/Strip.h"

namespace o2
{
namespace tof
{

class WindowFiller
{

 public:
  WindowFiller() { initObj(); };
  ~WindowFiller() = default;

  void initObj();

  Int_t getCurrentReadoutWindow() const { return mReadoutWindowCurrent; }
  void setCurrentReadoutWindow(Double_t value) { mReadoutWindowCurrent = value; }
  void setEventTime(double value) { mEventTime = value; }

  std::vector<std::vector<Digit>>* getDigitPerTimeFrame() { return &mDigitsPerTimeFrame; }

  void fillOutputContainer(std::vector<Digit>& digits);
  void flushOutputContainer(std::vector<Digit>& digits); // flush all residual buffered data
  void setContinuous(bool value=true) {mContinuous=value;}
  bool isContinuous() const {return mContinuous;}

 protected:
  // info TOF timewindow
  Int_t mReadoutWindowCurrent = 0;
  Double_t mEventTime;

  bool mContinuous = false;

  // digit info
  //std::vector<Digit>* mDigits;

  static const int MAXWINDOWS = 2; // how many readout windows we can buffer

  std::vector<std::vector<Digit>> mDigitsPerTimeFrame;

  int mIcurrentReadoutWindow = 0;

  // array of strips to store the digits per strip (one for the current readout window, one for the next one)
  std::vector<Strip> mStrips[MAXWINDOWS];
  std::vector<Strip>* mStripsCurrent = &(mStrips[0]);
  std::vector<Strip>* mStripsNext[MAXWINDOWS - 1];

  // arrays with digit and MCLabels out of the current readout windows (stored to fill future readout window)
  std::vector<Digit> mFutureDigits;

  void fillDigitsInStrip(std::vector<Strip>* strips, int channel, int tdc, int tot, int nbc, UInt_t istrip);
  //  void fillDigitsInStrip(std::vector<Strip>* strips, o2::dataformats::MCTruthContainer<o2::tof::MCLabel>* mcTruthContainer, int channel, int tdc, int tot, int nbc, UInt_t istrip, Int_t trackID, Int_t eventID, Int_t sourceID);

  void checkIfReuseFutureDigits();

  void insertDigitInFuture(int digitInfo0, int digitInfo1, int digitInfo2, int digitInfo3, int lbl=0){
    mFutureDigits.emplace_back(digitInfo0, digitInfo1, digitInfo2, digitInfo3, lbl);
    // sort digit in descending BC order: kept last as first
    std::sort(mFutureDigits.begin(), mFutureDigits.end(),
	      [](o2::tof::Digit a, o2::tof::Digit b) { return a.getBC() > b.getBC(); });
  }

  bool isMergable(Digit digit1, Digit digit2)
  {
    if (digit1.getChannel() != digit2.getChannel()) {
      return false;
    }

    if (digit1.getBC() != digit2.getBC()) {
      return false;
    }

    // Check if the difference is larger than the TDC dead time
    if (std::abs(digit1.getTDC() - digit2.getTDC()) > Geo::DEADTIMETDC) {
      return false;
    }
    return true;
  }

  ClassDefNV(WindowFiller, 1);
};
} // namespace tof
} // namespace o2
#endif
