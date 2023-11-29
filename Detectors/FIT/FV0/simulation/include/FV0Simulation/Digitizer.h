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

#ifndef ALICEO2_FV0_DIGITIZER_H
#define ALICEO2_FV0_DIGITIZER_H

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFV0/Digit.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/MCLabel.h"
#include "FV0Simulation/Detector.h"
#include "FV0Base/Constants.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "FV0Simulation/DigitizationConstant.h"
#include <array>
#include <vector>

namespace o2
{
namespace fv0
{
class Digitizer
{
 private:
  using DP = DigitizationConstant;

 public:
  Digitizer()
    : mTimeStamp(0), mIntRecord(), mEventId(-1), mSrcId(-1), mMCLabels(), mCache(), mPmtChargeVsTime(), mNBins(), mNTimeBinsPerBC(), mPmtResponseGlobalRing5(), mPmtResponseGlobalRingA1ToA4(), mPmtResponseTemp(), mLastBCCache(), mCfdStartIndex()
  {
  }

  /// Destructor
  ~Digitizer() = default;

  Digitizer(const Digitizer&) = delete;
  Digitizer& operator=(const Digitizer&) = delete;

  void clear();
  void init();

  void setTimeStamp(long t) { mTimeStamp = t; }
  void setEventId(Int_t id) { mEventId = id; }
  void setSrcId(Int_t id) { mSrcId = id; }
  void setInteractionRecord(const InteractionTimeRecord& ir) { mIntRecord = ir; }

  void process(const std::vector<o2::fv0::Hit>& hits, std::vector<o2::fv0::Digit>& digitsBC,
               std::vector<o2::fv0::ChannelData>& digitsCh, std::vector<o2::fv0::DetTrigInput>& digitsTrig,
               o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>& labels);

  void flush(std::vector<o2::fv0::Digit>& digitsBC,
             std::vector<o2::fv0::ChannelData>& digitsCh,
             std::vector<o2::fv0::DetTrigInput>& digitsTrig,
             o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>& labels);

  const InteractionRecord& getInteractionRecord() const { return mIntRecord; }
  InteractionRecord& getInteractionRecord(InteractionRecord& src) { return mIntRecord; }
  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }

  using ChannelDigitF = std::vector<float>;

  struct BCCache : public o2::InteractionRecord {
    std::vector<o2::fv0::MCLabel> labels;
    std::array<ChannelDigitF, Constants::nFv0Channels> mPmtChargeVsTime = {};

    void clear()
    {
      for (auto& channel : mPmtChargeVsTime) {
        std::fill(std::begin(channel), std::end(channel), 0.);
      }
      labels.clear();
    }

    BCCache& operator=(const o2::InteractionRecord& ir)
    {
      o2::InteractionRecord::operator=(ir);
      return *this;
    }
    void print() const;
  };

 private:
  static constexpr int BCCacheMin = 0, BCCacheMax = 7, NBC2Cache = 1 + BCCacheMax - BCCacheMin;
  /// Create signal pulse based on MC hit
  /// \param mipFraction Fraction of the MIP energy deposited in the cell
  /// \param parID       Particle ID
  /// \param hitTime     Time of the hit
  /// \param hitR        Length to IP from the position of the hit
  /// \param cachedIR    Cached interaction records
  /// \param nCachedIR   Number of cached interaction records
  /// \param detID       Detector cell ID
  void createPulse(float mipFraction, int parID, const double hitTime, const float hitR,
                   std::array<o2::InteractionRecord, NBC2Cache> const& cachedIR, int nCachedIR, const int detID);

  long mTimeStamp;                  // TF (run) timestamp
  InteractionTimeRecord mIntRecord; // Interaction record (orbit, bc) -> InteractionTimeRecord
  Int_t mEventId;                   // ID of the current event
  Int_t mSrcId;                     // signal, background or QED
  std::deque<fv0::MCLabel> mMCLabels;
  std::deque<BCCache> mCache;

  BCCache& setBCCache(const o2::InteractionRecord& ir);
  BCCache* getBCCache(const o2::InteractionRecord& ir);

  void storeBC(const BCCache& bc,
               std::vector<o2::fv0::Digit>& digitsBC,
               std::vector<o2::fv0::ChannelData>& digitsCh,
               std::vector<o2::fv0::DetTrigInput>& digitsTrig,
               o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>& labels);
  bool isRing5(int detID);

  std::array<std::vector<Float_t>, Constants::nFv0Channels> mPmtChargeVsTime; // Charge time series aka analogue signal pulse from PM
  UInt_t mNBins;                                                              //
  UInt_t mNTimeBinsPerBC;
  Float_t mBinSize; // Time width of the pulse bin - HPTDC resolution

  /// vectors to store the PMT signal from cosmic muons
  std::vector<Double_t> mPmtResponseGlobalRing5;
  std::vector<Double_t> mPmtResponseGlobalRingA1ToA4;
  std::vector<Double_t> mPmtResponseTemp;

  /// for CFD
  BCCache mLastBCCache;                                    // buffer for the last BC
  std::array<int, Constants::nFv0Channels> mCfdStartIndex; // start indices for the CFD detector

  /// Internal helper methods related to conversion of energy-deposition into el. signal
  Int_t SimulateLightYield(Int_t pmt, Int_t nPhot) const;
  Float_t SimulateTimeCfd(int& startIndex, const ChannelDigitF& pulseLast, const ChannelDigitF& pulse) const;
  Float_t IntegrateCharge(const ChannelDigitF& pulse) const;

  /// Functions related to splitting ring-5 cell signal to two readout channels
  static float getDistFromCellCenter(UInt_t cellId, double hitx, double hity);
  static float getSignalFraction(float distanceFromXc, bool isFirstChannel);

  ClassDefNV(Digitizer, 2);
};

// Function used to split the ring-5 cell signal into two readout channels depending on hit position
inline float sigmoidPmtRing5(float x)
{
  return -0.668453 / (1.0 + TMath::Exp(TMath::Abs(x) / 3.64327)) + 0.834284;
};

} // namespace fv0
} // namespace o2

#endif
