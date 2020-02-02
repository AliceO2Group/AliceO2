// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FT0_DIGITIZER_H
#define ALICEO2_FT0_DIGITIZER_H

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/MCLabel.h"
#include "FT0Simulation/Detector.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "FT0Simulation/DigitizationParameters.h"
#include <TH1F.h>
#include <bitset>
#include <vector>

namespace o2
{
namespace ft0
{
class Digitizer
{
 public:
  Digitizer(const DigitizationParameters& params, Int_t mode = 0) : mMode(mode), parameters(params) { initParameters(); };
  ~Digitizer() = default;

  void process(const std::vector<o2::ft0::HitType>* hits);
  //           std::vector<o2::ft0::Digit>& digitsBC,
  //    std::vector<o2::ft0::ChannelData>& digitsCh,
  // o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& labels);

  void setDigits(std::vector<o2::ft0::Digit>& digitsBC,
                 std::vector<o2::ft0::ChannelData>& digitsCh);
  void initParameters();
  void printParameters();
  void setTimeStamp(double value) { mEventTime = value; }
  void setEventID(Int_t id) { mEventID = id; }
  void setSrcID(Int_t id) { mSrcID = id; }
  const o2::InteractionRecord& getInteractionRecord() const { return mIntRecord; }
  o2::InteractionRecord& getInteractionRecord(o2::InteractionRecord& src) { return mIntRecord; }
  void setInteractionRecord(const o2::InteractionRecord& src) { mIntRecord = src; }
  // uint32_t getOrbit() const { return mIntRecord.orbit; }
  //  uint16_t getBC() const { return mIntRecord.bc; }

  void init();
  void finish();

  void setMCLabels(o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>* mclb) { mMCLabels = mclb; }
  double get_time(const std::vector<double>& times);
  std::vector<std::vector<double>> mChannel_times;

  void setContinuous(bool v = true) { mIsContinuous = v; }
  bool isContinuous() const { return mIsContinuous; }
  void cleanChannelData()
  {
    for (Int_t ipmt = 0; ipmt < parameters.mMCPs; ++ipmt)
      mNumParticles[ipmt] = 0;
  }
  void clearDigits()
  {
    for (int i = 0; i < parameters.mMCPs; ++i) {
      mNumParticles[i] = 0;
    }
  }

 private:
  // digit info
  // parameters
  Int_t mMode;                      //triggered or continuos
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)
  Int_t mEventID;
  Int_t mSrcID;        // signal, background or QED
  Double_t mEventTime; // timestamp
  int mNoisePeriod;    //low frequency noise period
  int mBinshift;       // number of bin to shift positive part of CFD signal
  bool mIsContinuous = true; // continuous (self-triggered) or externally-triggered readout
  int mNumParticles[208];

  DigitizationParameters parameters;

  o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>* mMCLabels = nullptr;

  o2::ft0::Triggers mTriggers;

  TH1F* mHist;      // ("time_histogram", "", 1000, -0.5 * signal_width, 0.5 * signal_width);
  TH1F* mHistsum;   //("time_sum", "", 1000, -0.5 * signal_width, 0.5 * signal_width);
  TH1F* mHistshift; //("time_shift", "", 1000, -0.5 * signal_width, 0.5 * signal_width);

  ClassDefNV(Digitizer, 1);
};
inline double sinc(const double x)
{
  return (std::abs(x) < 1e-12) ? 1 : std::sin(x) / x;
}
inline double signalForm_i(double x)
{
  return x > 0 ? -(exp(-0.83344945 * x) - exp(-0.45458 * x)) / 7.8446501 : 0.;
  //return -(exp(-0.83344945 * x) - exp(-0.45458 * x)) * (x >= 0) / 7.8446501; // Maximum should be 7.0/250 mV
};
} // namespace ft0
} // namespace o2

#endif
