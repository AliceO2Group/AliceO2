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
#include "MathUtils/RandomRing.h"
#include "FT0Simulation/Detector.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "FT0Simulation/DigitizationConstants.h"
#include "FT0Simulation/DigitizationParameters.h"
#include <TH1F.h>
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
  typedef math_utils::RandomRing<float_v::size() * DP::NOISE_RANDOM_RING_SIZE> NoiseRandomRingType;

 public:
  Digitizer(const DigitizationParameters& params, Int_t mode = 0) : mMode(mode), mParameters(params), mRndGaus(NoiseRandomRingType::RandomType::Gaus), mNumNoiseSamples(), mNoiseSamples(), mSincTable(), mSignalTable(), mSignalCache() { initParameters(); }
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
    float const y = x / mParameters.bunchWidth * DP::SIGNAL_TABLE_SIZE;
    int const index = std::floor(y);
    if (index + 1 >= DP::SIGNAL_TABLE_SIZE) {
      return mSignalTable.back();
    }
    float const rem = y - index;
    return mSignalTable[index] + rem * (mSignalTable[index + 1] - mSignalTable[index]);
  }

  inline Vc::float_v signalFormVc(Vc::float_v x) const
  { // table lookup for the signal shape (SIMD version)
    auto const y = x / mParameters.bunchWidth * DP::SIGNAL_TABLE_SIZE;
    Vc::float_v::IndexType const index = Vc::floor(y);
    auto const rem = y - index;
    Vc::float_v val(0);
    for (size_t i = 0; i < float_v::size(); ++i) {
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
  std::array<GoodInteractionTimeRecord, 208> mDeadTimes;

  DigitizationParameters mParameters;

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

  ClassDefNV(Digitizer, 1);
};

// signal shape function
template <typename Float>
Float signalForm_i(Float x)
{
  using namespace std;
  Float const a = -0.45458;
  Float const b = -0.83344945;
  return x > Float(0) ? -(exp(b * x) - exp(a * x)) / Float(7.8446501) : Float(0);
  //return -(exp(-0.83344945 * x) - exp(-0.45458 * x)) * (x >= 0) / 7.8446501; // Maximum should be 7.0/250 mV
};

// integrated signal shape function
inline float signalForm_integral(float x)
{
  using namespace std;
  double const a = -0.45458;
  double const b = -0.83344945;
  if (x < 0) {
    x = 0;
  }
  return -(exp(b * x) / b - exp(a * x) / a) / 7.8446501;
};

// SIMD version of the integrated signal shape function
inline Vc::float_v signalForm_integralVc(Vc::float_v x)
{
  auto const mask = (x >= 0.0f);
  Vc::float_v arg(0);
  arg.assign(x, mask); // branchless if
  Vc::float_v const a(-0.45458f);
  Vc::float_v const b(-0.83344945f);
  Vc::float_v result = -(Vc::exp(b * arg) / b - Vc::exp(a * arg) / a) / 7.8446501f;
  return result;
};
} // namespace ft0
} // namespace o2

#endif
