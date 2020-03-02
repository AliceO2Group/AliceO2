// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FDD_DIGITIZER_H
#define ALICEO2_FDD_DIGITIZER_H

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/MCLabel.h"
#include "FDDSimulation/Detector.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "FDDSimulation/DigitizationParameters.h"
#include "MathUtils/RandomRing.h"
#include "MathUtils/CachingTF1.h"

namespace o2
{
namespace fdd
{
class Digitizer
{

 private:
  typedef math_utils::RandomRing<float_v::size() * DigitizationParameters::mPheRRSize> HitRandomRingType;
  typedef math_utils::RandomRing<float_v::size() * DigitizationParameters::mHitRRSize> PheRandomRingType;

 public:
  Digitizer(const DigitizationParameters& params, Int_t mode = 0) : mEventTime(0), mIntRecord(), mEventID(-1), mSrcID(-1), mMCLabels(), parameters(params), mTime(), mRndScintDelay(HitRandomRingType::RandomType::CustomTF1), mRndGainVar(PheRandomRingType::RandomType::CustomTF1), mRndSignalShape(PheRandomRingType::RandomType::CustomTF1), mPMResponseTables() { init(); };
  ~Digitizer() = default;

  void process(const std::vector<o2::fdd::Hit>* hits, o2::fdd::Digit* digit);

  void initParameters();
  void SetEventTime(long value) { mEventTime = value; }
  void SetEventID(Int_t id) { mEventID = id; }
  void SetSrcID(Int_t id) { mSrcID = id; }
  void SetInteractionRecord(uint16_t bc, uint32_t orbit)
  {
    mIntRecord.bc = bc;
    mIntRecord.orbit = orbit;
  }
  const o2::InteractionRecord& GetInteractionRecord() const { return mIntRecord; }
  o2::InteractionRecord& GetInteractionRecord(o2::InteractionRecord& src) { return mIntRecord; }
  void SetInteractionRecord(const o2::InteractionRecord& src) { mIntRecord = src; }
  uint32_t GetOrbit() const { return mIntRecord.orbit; }
  uint16_t GetBC() const { return mIntRecord.bc; }

  void SetTriggers(o2::fdd::Digit* digit);
  Int_t SimulateLightYield(Int_t pmt, Int_t nPhot);
  Float_t SimulateTimeCFD(Int_t channel);

  void init();
  void finish();

  void setMCLabels(o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>* mclb) { mMCLabels = mclb; }

 private:
  long mEventTime;              // TF (run) timestamp
  InteractionRecord mIntRecord; // Interaction record (orbit, bc) -> InteractionTimeRecord
  Int_t mEventID;               // ID of the current event
  Int_t mSrcID;                 // signal, background or QED

  DigitizationParameters parameters;
  o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>* mMCLabels = nullptr;

  std::array<std::vector<Float_t>, DigitizationParameters::mNchannels> mTime; // Charge time series aka analogue signal pulse from PM
  std::vector<Float_t> mTimeCFD;                                              // Time series for CFD measurement
  UInt_t mNBins;                                                              // Number of bins in pulse series
  Float_t mBinSize;                                                           // Time width of the pulse bin - HPTDC resolution
  Float_t mPmtTimeIntegral;

  // Random rings
  HitRandomRingType mRndScintDelay;
  PheRandomRingType mRndGainVar;
  PheRandomRingType mRndSignalShape;

  // 8 tables starting at different sub-bin positions, i.e, [-4:4] / 8 * mBinSize
  // wit each table containg values for start + [-2:2:mBinSize] * DigitizationParameters::mPmtTransitTime
  std::array<std::vector<Float_t>, DigitizationParameters::mNResponseTables> mPMResponseTables;

  static Double_t PMResponse(Double_t x);
  static Double_t PMResponse(Double_t* x, Double_t*);
  static Double_t SinglePhESpectrum(Double_t* x, Double_t* par);

  ClassDefNV(Digitizer, 3);
};
} // namespace fdd
} // namespace o2

#endif
