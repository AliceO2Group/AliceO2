// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TOF_DIGITIZER_H_
#define ALICEO2_TOF_DIGITIZER_H_

#include "TOFBase/Geo.h"
#include "TOFBase/Digit.h"
#include "TOFBase/Strip.h"
#include "TOFBase/WindowFiller.h"
#include "TOFSimulation/Detector.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TOFSimulation/MCLabel.h"
#include "TOFBase/CalibTOFapi.h"

namespace o2
{
namespace tof
{

class Digitizer : public WindowFiller
{
  using CalibApi = o2::tof::CalibTOFapi;

 public:
  Digitizer(Int_t mode = 0) : WindowFiller(), mMode(mode) { init(); };
  ~Digitizer() = default;

  void init();

  int process(const std::vector<HitType>* hits, std::vector<Digit>* digits);

  void setCalibApi(CalibApi* calibApi) { mCalibApi = calibApi; }

  void setMCTruthContainer(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* truthcontainer)
  {
    mMCTruthOutputContainer = truthcontainer;
  }

  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>* getMCTruthPerTimeFrame() { return &mMCTruthOutputContainerPerTimeFrame; }

  void fillOutputContainer(std::vector<Digit>& digits);
  void flushOutputContainer(std::vector<Digit>& digits); // flush all residual buffered data

  // Method used for digitization
  void initParameters();
  void printParameters();
  Double_t getShowerTimeSmeared(Double_t time, Float_t charge);
  Double_t getDigitTimeSmeared(Double_t time, Float_t x, Float_t z, Float_t charge);
  Float_t getCharge(Float_t eDep);
  Bool_t isFired(Float_t x, Float_t z, Float_t charge);
  Float_t getEffX(Float_t x);
  Float_t getEffZ(Float_t z);
  Float_t getFractionOfCharge(Float_t x, Float_t z);

  Float_t getTimeLastHit(Int_t idigit) const { return 0; }
  Float_t getTotLastHit(Int_t idigit) const { return 0; }
  Int_t getXshift(Int_t idigit) const { return 0; }
  Int_t getZshift(Int_t idigit) const { return 0; }
  void setEventID(Int_t id) { mEventID = id; }
  void setSrcID(Int_t id) { mSrcID = id; }

  void test(const char* geo = "");
  void testFromHits(const char* geo = "", const char* hits = "AliceO2_TGeant3.tof.mc_10_event.root");

  void setShowerSmearing();
  void setResolution(float val)
  {
    mTOFresolution = val;
    setShowerSmearing();
  }
  void setEffCenter(float val) { mEffCenter = val; }
  void setEffBoundary1(float val) { mEffBoundary1 = val; }
  void setEffBoundary2(float val) { mEffBoundary2 = val; }
  void setEffBoundary3(float val) { mEffBoundary3 = val; }

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

  // info TOF timewindow for MC
  Int_t mEventID = 0;
  Int_t mSrcID = 0;

  // digit info
  //std::vector<Digit>* mDigits;

  // final vector of tof readout window MC
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mMCTruthOutputContainerPerTimeFrame;

  // temporary MC info in the current tof readout windows
  o2::dataformats::MCTruthContainer<o2::tof::MCLabel> mMCTruthContainer[MAXWINDOWS];
  o2::dataformats::MCTruthContainer<o2::tof::MCLabel>* mMCTruthContainerCurrent = &mMCTruthContainer[0]; ///< Array for MCTruth information associated to digits in mDigitsArrray.
  o2::dataformats::MCTruthContainer<o2::tof::MCLabel>* mMCTruthContainerNext[MAXWINDOWS - 1];            ///< Array for MCTruth information associated to digits in mDigitsArrray.
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mMCTruthOutputContainer;

  // arrays with digit and MCLabels out of the current readout windows (stored to fill future readout window)
  std::vector<int> mFutureIevent;
  std::vector<int> mFutureIsource;
  std::vector<int> mFutureItrackID;

  o2::dataformats::MCTruthContainer<o2::tof::MCLabel> mFutureMCTruthContainer;

  CalibApi* mCalibApi = nullptr; //! calib api to handle the TOF calibration

  void fillDigitsInStrip(std::vector<Strip>* strips, o2::dataformats::MCTruthContainer<o2::tof::MCLabel>* mcTruthContainer, int channel, int tdc, int tot, uint64_t nbc, UInt_t istrip, Int_t trackID, Int_t eventID, Int_t sourceID);

  Int_t processHit(const HitType& hit, Double_t event_time);
  void addDigit(Int_t channel, UInt_t istrip, Double_t time, Float_t x, Float_t z, Float_t charge, Int_t iX, Int_t iZ, Int_t padZfired,
                Int_t trackID);

  void checkIfReuseFutureDigits();

  ClassDefNV(Digitizer, 1);
};
} // namespace tof
} // namespace o2
#endif
