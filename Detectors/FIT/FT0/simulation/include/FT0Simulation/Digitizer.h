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

#ifndef ALICEO2_FT0_DIGITIZER_H
#define ALICEO2_FT0_DIGITIZER_H

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/MCLabel.h"
#include "FT0Simulation/Detector.h"
#include "FT0Base/Geometry.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "FT0Simulation/DigitizationConstants.h"
#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"
#include "FT0Base/FT0DigParam.h"
#include "MathUtils/RandomRing.h"
#include <array>
#include <bitset>
#include <vector>
#include <deque>
#include <optional>
#include <set>

namespace o2
{
namespace ft0
{
class Digitizer
{
 private:
  using DP = DigitizationConstants;
  typedef math_utils::RandomRing</*float_v::size()*/ 4 * DP::NOISE_RANDOM_RING_SIZE> NoiseRandomRingType;
  static constexpr int NCHANNELS = o2::ft0::Geometry::Nchannels;

 public:
  Digitizer(Int_t mode = 0) : mMode(mode), mRndGaus(NoiseRandomRingType::RandomType::Gaus), mNumNoiseSamples(), mNoiseSamples(), mSincTable(), mSignalTable(), mSignalCache() { initParameters(); }
  ~Digitizer() = default;

  void process(const std::vector<o2::ft0::HitType>* hits, std::vector<o2::ft0::Digit>& digitsBC,
               std::vector<o2::ft0::ChannelData>& digitsCh,
               std::vector<o2::ft0::DetTrigInput>& digitsTrig,
               o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& label);
  void flush(std::vector<o2::ft0::Digit>& digitsBC,
             std::vector<o2::ft0::ChannelData>& digitsCh,
             std::vector<o2::ft0::DetTrigInput>& digitsTrig,
             o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& label);
  void flush_all(std::vector<o2::ft0::Digit>& digitsBC,
                 std::vector<o2::ft0::ChannelData>& digitsCh,
                 std::vector<o2::ft0::DetTrigInput>& digitsTrig,
                 o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& label);
  void initParameters();
  void printParameters() const;
  void setEventID(Int_t id) { mEventID = id; }
  void setSrcID(Int_t id) { mSrcID = id; }
  const o2::InteractionRecord& getInteractionRecord() const { return mIntRecord; }
  o2::InteractionRecord& getInteractionRecord(o2::InteractionRecord& src) { return mIntRecord; }
  void setInteractionRecord(const o2::InteractionTimeRecord& src) { mIntRecord = src; }
  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }
  int getEvent() const { return mEventID; }
  double measure_amplitude(const std::vector<float>& times) const;
  void init();
  void finish();

  void SetChannelOffset(o2::ft0::FT0ChannelTimeCalibrationObject const*
                          caliboffsets) { mCalibOffset = caliboffsets; };

  struct CFDOutput {
    std::optional<double> particle;
    double deadTime;
  };
  CFDOutput get_time(const std::vector<float>& times, float deadTime);

  void setContinuous(bool v = true) { mIsContinuous = v; }
  bool isContinuous() const { return mIsContinuous; }
  struct BCCache {
    struct particle {
      int hit_ch;
      double hit_time;
      friend bool operator<(particle const& a, particle const& b)
      {
        return (a.hit_ch != b.hit_ch) ? (a.hit_ch < b.hit_ch) : (a.hit_time < b.hit_time);
      }
    };
    std::vector<particle> hits;
    std::set<ft0::MCLabel> labels;
  };
  struct GoodInteractionTimeRecord {
    o2::InteractionRecord intrec;
    double deadTime;
  };

 protected:
  inline float signalForm(float x) const
  { // table lookup for the signal shape
    if (x <= 0.0f) {
      return 0.0f;
    }
    float const y = x / FT0DigParam::Instance().mBunchWidth * DP::SIGNAL_TABLE_SIZE;
    int const index = std::floor(y);
    if (index + 1 >= DP::SIGNAL_TABLE_SIZE) {
      return mSignalTable.back();
    }
    float const rem = y - index;
    return mSignalTable[index] + rem * (mSignalTable[index + 1] - mSignalTable[index]);
  }

  template <typename VcType>
  inline VcType signalFormVc(VcType x) const
  { // table lookup for the signal shape (SIMD version)
    // implemented as template function, so that we don't need to include <Vc/Vc> here
    auto const y = x / FT0DigParam::Instance().mBunchWidth * DP::SIGNAL_TABLE_SIZE;
    typename VcType::IndexType const index = floor(y);
    auto const rem = y - index;
    VcType val(0);
    for (size_t i = 0; i < VcType::size(); ++i) {
      if (y[i] < 0.0f) {
        continue;
      }
      if (index[i] + 1 < DP::SIGNAL_TABLE_SIZE) {
        val[i] = mSignalTable[index[i]] + rem[i] * (mSignalTable[index[i] + 1] - mSignalTable[index[i]]);
      } else {
        val[i] = mSignalTable.back();
      }
    }
    return val;
  }

 private:
  // digit info
  // parameters
  Int_t mMode;                          // triggered or continuos
  o2::InteractionTimeRecord mIntRecord; // Interaction record (orbit, bc)
  Int_t mEventID;
  Int_t mSrcID;              // signal, background or QED
  bool mIsContinuous = true; // continuous (self-triggered) or externally-triggered readout

  o2::InteractionRecord firstBCinDeque = 0;
  std::deque<BCCache> mCache;
  std::array<GoodInteractionTimeRecord, NCHANNELS> mDeadTimes;

  o2::ft0::Geometry mGeometry;

  NoiseRandomRingType mRndGaus;
  int mNumNoiseSamples; // number of noise samples in one BC
  std::vector<float> mNoiseSamples;
  std::array<std::vector<float>, DP::SINC_TABLE_SIZE> mSincTable;
  std::array<float, DP::SIGNAL_TABLE_SIZE> mSignalTable;
  std::vector<float> mSignalCache; // cached summed signal used by the CFD

  void storeBC(BCCache& bc,
               std::vector<o2::ft0::Digit>& digitsBC,
               std::vector<o2::ft0::ChannelData>& digitsCh,
               std::vector<o2::ft0::DetTrigInput>& digitsTrig,
               o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& labels);

  o2::ft0::FT0ChannelTimeCalibrationObject const* mCalibOffset = nullptr;

  ClassDefNV(Digitizer, 3);
};

} // namespace ft0
} // namespace o2

#endif
