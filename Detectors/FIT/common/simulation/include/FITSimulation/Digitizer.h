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

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFITT0/Digit.h"
#include "DataFormatsFITT0/MCLabel.h"
#include "T0Simulation/Detector.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "FITSimulation/DigitizationParameters.h"
#include <TH1F.h>

namespace o2
{
namespace fit
{
class Digitizer
{
 public:
  Digitizer(const DigitizationParameters& params, Int_t mode = 0) : mMode(mode), parameters(params) { initParameters(); };
  ~Digitizer() = default;

  //void process(const std::vector<HitType>* hits, std::vector<Digit>* digits);
  void process(const std::vector<o2::t0::HitType>* hits, o2::t0::Digit* digit, std::vector<std::vector<double>>& channel_times);
  void computeAverage(o2::t0::Digit& digit);

  void initParameters();
  // void printParameters();
  void setEventTime(double value) { mEventTime = value; }
  void setEventID(Int_t id) { mEventID = id; }
  void setSrcID(Int_t id) { mSrcID = id; }
  void setInteractionRecord(uint16_t bc, uint32_t orbit)
  {
    mIntRecord.bc = bc;
    mIntRecord.orbit = orbit;
  }
  const o2::InteractionRecord& getInteractionRecord() const { return mIntRecord; }
  o2::InteractionRecord& getInteractionRecord(o2::InteractionRecord& src) { return mIntRecord; }
  void setInteractionRecord(const o2::InteractionRecord& src) { mIntRecord = src; }
  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }

  void setTriggers(o2::t0::Digit* digit);
  void smearCFDtime(o2::t0::Digit* digit, std::vector<std::vector<double>> const& channel_times);

  void init();
  void finish();

  void setMCLabels(o2::dataformats::MCTruthContainer<o2::t0::MCLabel>* mclb) { mMCLabels = mclb; }
  double get_time(const std::vector<double>& times, double signal_width);

 private:
  // digit info
  // parameters
  Int_t mMode;  //triggered or continuos
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)
  Int_t mEventID;
  Int_t mSrcID;        // signal, background or QED
  Double_t mEventTime; // timestamp

  DigitizationParameters parameters;

  o2::dataformats::MCTruthContainer<o2::t0::MCLabel>* mMCLabels = nullptr;

  static constexpr Float_t C_side_cable_cmps = 2.877; //ns
  static constexpr Float_t A_side_cable_cmps = 11.08; //ns

  TH1F* mHist;      // ("time_histogram", "", 1000, -0.5 * signal_width, 0.5 * signal_width);
  TH1F* mHistsum;   //("time_sum", "", 1000, -0.5 * signal_width, 0.5 * signal_width);
  TH1F* mHistshift; //("time_shift", "", 1000, -0.5 * signal_width, 0.5 * signal_width);

  //static constexpr Float_t signal_width = 5.;         // time gate for signal, ns

  //static std::vector<double> aggregate_channels(const std::vector<o2::fit::HitType>& hits, DigitizationParameters const& parameters);

  ClassDefNV(Digitizer, 1);
};
} // namespace fit
} // namespace o2

#endif
