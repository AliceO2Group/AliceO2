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

#include "FT0Simulation/Digitizer.h"
#include "FT0Simulation/DigitizationConstants.h"
#include "FT0Base/FT0DigParam.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonConstants/PhysicsConstants.h"
#include "CommonDataFormat/InteractionRecord.h"

#include "TMath.h"
#include "TRandom.h"
#include <TH1F.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <optional>
#include <Vc/Vc>

using namespace o2::ft0;

ClassImp(Digitizer);

namespace o2::ft0
{
// signal shape function
template <typename Float>
Float signalForm_i(Float x)
{
  using namespace std;
  Float const a = -0.45458;
  Float const b = -0.83344945;
  return x > Float(0) ? -(exp(b * x) - exp(a * x)) / Float(7.8446501) : Float(0);
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
} // namespace o2::ft0

Digitizer::CFDOutput Digitizer::get_time(const std::vector<float>& times, float deadTime)
{
  assert(std::is_sorted(std::begin(times), std::end(times)));

  // get a new batch of random values
  for (float& n : mNoiseSamples) {
    n = mRndGaus.getNextValue();
  }

  // this lambda function evaluates the signal shape at a give time
  auto value_at = [&](float time) {
    float val = 0;
    // (1) sum over individual hits
    Vc::float_v acc(0);
    Vc::float_v tableVal(0);
    const float* tp = times.data();
    size_t m = times.size() / Vc::float_v::size();
    for (size_t i = 0; i < m; ++i) {
      tableVal.load(tp);
      tp += Vc::float_v::size();
      Vc::prefetchForOneRead(tp);
      acc += signalFormVc(time - tableVal);
    }
    val += acc.sum();
    // non-SIMD tail
    for (size_t i = Vc::float_v::size() * m; i < times.size(); ++i, ++tp) {
      val += signalForm(time - (*tp));
    }
    // (2) add noise
    // find the right indices into the sinc table
    int timeIndex = std::lround(time / FT0DigParam::Instance().mNoisePeriod * mSincTable.size());
    int timeOffset = timeIndex / mSincTable.size();
    timeIndex %= mSincTable.size();
    if (timeOffset >= mNumNoiseSamples) { // this happens when time >= 25 ns
      timeOffset = mNumNoiseSamples - 1;
      LOG(debug) << "timeOffset >= mNumNoiseSamples";
    }
    if (timeOffset <= -mNumNoiseSamples) { // this happens when time <= -25 ns
      timeOffset = -mNumNoiseSamples + 1;
      LOG(debug) << "timeOffset <= -mNumNoiseSamples";
    }
    Vc::float_v noiseVal(0);
    const float* np = mNoiseSamples.data();
    tp = mSincTable[timeIndex].data() + mNumNoiseSamples - timeOffset;
    acc = 0.0f;
    m = mNumNoiseSamples / Vc::float_v::Size;
    for (size_t i = 0; i < m; ++i) {
      tableVal.load(tp);
      tp += Vc::float_v::Size;
      Vc::prefetchForOneRead(tp);
      noiseVal.load(np);
      np += Vc::float_v::Size;
      Vc::prefetchForOneRead(np);
      acc += noiseVal * tableVal;
    }
    val += acc.sum(); // horizontal sum
    // non-SIMD tail
    for (size_t i = Vc::float_v::Size * m; i < mNumNoiseSamples; ++i, ++tp, ++np) {
      val += (*np) * (*tp);
    }
    return val;
  };
  auto const min_time = std::max(deadTime, *std::min_element(std::begin(times),
                                                             std::end(times)));
  CFDOutput result{std::nullopt, -0.5f * FT0DigParam::Instance().mBunchWidth};
  bool is_positive = true;

  // reset the chache
  std::fill_n(std::begin(mSignalCache), std::size(mSignalCache), -1.0f);
  const auto& params = FT0DigParam::Instance();
  // we need double precision for time in order to match previous behaviour
  for (double time = min_time; time < 0.5 * params.mBunchWidth; time += DP::SIGNAL_CACHE_DT) {
    float const val = value_at(time);
    int const index = std::lround((time + 0.5 * params.mBunchWidth) / DP::SIGNAL_CACHE_DT);
    if (index >= 0 && index < mSignalCache.size()) { // save the value for later use
      mSignalCache[index] = val;
    }
    // look up the time-shifted signal value from the past
    float val_prev = 0.0f;
    int const index_prev = std::lround((time - params.mCFDShiftPos + 0.5f * params.mBunchWidth) / DP::SIGNAL_CACHE_DT);
    val_prev = ((index_prev < 0 || index_prev >= mSignalCache.size() || mSignalCache[index_prev] < 0.0f)
                  ? value_at(time - params.mCFDShiftPos) //  was not computed before
                  : mSignalCache[index_prev]);           //  is available in the cache
    float const cfd_val = 5.0f * val_prev - val;
    if (std::abs(val) > params.mCFD_trsh && !is_positive && cfd_val > 0.0f) {
      if (!result.particle) {
        result.particle = time;
      }
      result.deadTime = time + params.mCFDdeadTime;
      time += params.mCFDdeadTime - DP::SIGNAL_CACHE_DT;
      is_positive = true;
    } else {
      is_positive = cfd_val > 0.0f;
    }
  }
  return result;
}

double Digitizer::measure_amplitude(const std::vector<float>& times) const
{
  float const from = FT0DigParam::Instance().mAmpRecordLow;
  float const to = from + FT0DigParam::Instance().mAmpRecordUp;
  // SIMD version has a negligible effect on the total wall time
  Vc::float_v acc(0);
  Vc::float_v tv(0);
  const float* tp = times.data();
  size_t const m = times.size() / Vc::float_v::Size;
  for (size_t i = 0; i < m; ++i) {
    tv.load(tp);
    tp += Vc::float_v::Size;
    Vc::prefetchForOneRead(tp);
    acc += signalForm_integralVc(to - tv) - signalForm_integralVc(from - tv);
  }
  float result = acc.sum(); // horizontal sum
  // non-SIMD tail
  for (size_t i = Vc::float_v::Size * m; i < times.size(); ++i, ++tp) {
    result += signalForm_integral(to - (*tp)) - signalForm_integral(from - (*tp));
  }
  return result;
}

void Digitizer::process(const std::vector<o2::ft0::HitType>* hits,
                        std::vector<o2::ft0::Digit>& digitsBC,
                        std::vector<o2::ft0::ChannelData>& digitsCh,
                        std::vector<o2::ft0::DetTrigInput>& digitsTrig,
                        o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& label)
{
  ;
  // Calculating signal time, amplitude in mean_time +- time_gate --------------
  LOG(debug) << " process firstBCinDeque " << firstBCinDeque << " mIntRecord " << mIntRecord;
  if (firstBCinDeque != mIntRecord) {
    flush(digitsBC, digitsCh, digitsTrig, label);
  }

  Int_t parent = -10;
  for (auto const& hit : *hits) {
    if (hit.GetEnergyLoss() > 0) {
      continue;
    }

    Int_t hit_ch = hit.GetDetectorID();

    // If the dead channel map is used, and the channel with ID 'hit_ch' is dead, don't process this hit.
    if (mDeadChannelMap && !mDeadChannelMap->isChannelAlive(hit_ch)) {
      continue;
    }

    const auto& params = FT0DigParam::Instance();

    Bool_t is_A_side = (hit_ch < 4 * mGeometry.NCellsA);

    // Subtract time-of-flight from hit time
    const Float_t timeOfFlight = hit.GetPos().R() / o2::constants::physics::LightSpeedCm2NS;
    const Float_t timeOffset = is_A_side ? params.hitTimeOffsetA : params.hitTimeOffsetC;
    Double_t hit_time = hit.GetTime() - timeOfFlight + timeOffset;

    if (hit_time > 150) {
      continue; // not collect very slow particles
    }

    auto relBC = o2::InteractionRecord{hit_time};
    if (mCache.size() <= relBC.bc) {
      mCache.resize(relBC.bc + 1);
    }
    mCache[relBC.bc].hits.emplace_back(BCCache::particle{hit_ch, hit_time - relBC.bc2ns()});
    // charge particles in MCLabel
    Int_t parentID = hit.GetTrackID();
    if (parentID != parent) {
      mCache[relBC.bc].labels.emplace(parentID, mEventID, mSrcID, hit_ch);
      parent = parentID;
    }
  }
}

void Digitizer::storeBC(BCCache& bc,
                        std::vector<o2::ft0::Digit>& digitsBC,
                        std::vector<o2::ft0::ChannelData>& digitsCh,
                        std::vector<o2::ft0::DetTrigInput>& digitsTrig,
                        o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& labels)
{
  if (bc.hits.empty()) {
    return;
  }
  int n_hit_A = 0, n_hit_C = 0, mean_time_A = 0, mean_time_C = 0;
  int summ_ampl_A = 0, summ_ampl_C = 0;
  int vertex_time;
  const auto& params = FT0DigParam::Instance();
  int first = digitsCh.size(), nStored = 0;
  auto& particles = bc.hits;
  std::sort(std::begin(particles), std::end(particles));
  auto channel_end = particles.begin();
  std::vector<float> channel_times;
  for (Int_t ipmt = 0; ipmt < params.mMCPs; ++ipmt) {
    auto channel_begin = channel_end;
    channel_end = std::find_if(channel_begin, particles.end(),
                               [ipmt](BCCache::particle const& p) { return p.hit_ch != ipmt; });

    // The hits between 'channel_begin' and 'channel_end' now contains all hits for channel 'ipmt'

    if (channel_end - channel_begin < params.mAmp_trsh) {
      continue;
    }
    channel_times.resize(channel_end - channel_begin);
    std::transform(channel_begin, channel_end, channel_times.begin(), [](BCCache::particle const& p) { return p.hit_time; });
    int chain = (std::rand() % 2) ? 1 : 0;
    auto cfd = get_time(channel_times, mDeadTimes[ipmt].intrec.bc2ns() -
                                         firstBCinDeque.bc2ns() +
                                         mDeadTimes[ipmt].deadTime);
    mDeadTimes[ipmt].intrec = firstBCinDeque;
    mDeadTimes[ipmt].deadTime = cfd.deadTime;

    if (!cfd.particle) {
      continue;
    }
    // miscalibrate CFD with cahnnel offsets
    int miscalib = 0;
    if (mCalibOffset) {
      miscalib = mCalibOffset->mTimeOffsets[ipmt];
    }
    int smeared_time = 1000. * (*cfd.particle - params.mCfdShift) * params.mChannelWidthInverse + miscalib + int(1000. * mIntRecord.getTimeOffsetWrtBC() * params.mChannelWidthInverse);
    bool is_time_in_signal_gate = (smeared_time > -params.mTime_trg_gate && smeared_time < params.mTime_trg_gate);
    float charge = measure_amplitude(channel_times) * params.mCharge2amp;
    float amp = is_time_in_signal_gate ? params.mMV_2_Nchannels * charge : 0;
    if (amp > 4095) {
      amp = 4095;
    }

    LOG(debug) << mEventID << " bc " << firstBCinDeque.bc << " orbit " << firstBCinDeque.orbit << ", ipmt " << ipmt << ", smeared_time " << smeared_time << " nStored " << nStored << " offset " << miscalib;
    if (is_time_in_signal_gate) {
      chain |= (1 << o2::ft0::ChannelData::EEventDataBit::kIsCFDinADCgate);
      chain |= (1 << o2::ft0::ChannelData::EEventDataBit::kIsEventInTVDC);
    }
    digitsCh.emplace_back(ipmt, smeared_time, int(amp), chain);
    nStored++;

    // fill triggers

    Bool_t is_A_side = (ipmt < 4 * mGeometry.NCellsA);
    if (!is_time_in_signal_gate) {
      continue;
    }

    if (is_A_side) {
      n_hit_A++;
      summ_ampl_A += amp;
      mean_time_A += smeared_time;
    } else {
      n_hit_C++;
      summ_ampl_C += amp;
      mean_time_C += smeared_time;
    }
  }
  Bool_t is_A, is_C, isVertex, is_Central, is_SemiCentral = 0;
  is_A = n_hit_A > 0;
  is_C = n_hit_C > 0;
  is_Central = summ_ampl_A + summ_ampl_C >= params.mtrg_central_trh;
  is_SemiCentral = summ_ampl_A + summ_ampl_C >= params.mtrg_semicentral_trh;
  uint32_t amplA = is_A ? summ_ampl_A * 0.125 : -5000; // sum amplitude A side / 8 (hardware)
  uint32_t amplC = is_C ? summ_ampl_C * 0.125 : -5000; // sum amplitude C side / 8 (hardware)
  int timeA = is_A ? mean_time_A / n_hit_A : -5000;    // average time A side
  int timeC = is_C ? mean_time_C / n_hit_C : -5000;    // average time C side
  vertex_time = (timeC - timeA) * 0.5;
  isVertex = is_A && is_C && (vertex_time > -params.mTime_trg_gate && vertex_time < params.mTime_trg_gate);
  LOG(debug) << " A " << is_A << " timeA " << timeA << " mean_time_A " << mean_time_A << "  n_hit_A " << n_hit_A << " C " << is_C << " timeC " << timeC << " mean_time_C " << mean_time_C << "  n_hit_C " << n_hit_C << " vertex_time " << vertex_time;
  Triggers triggers;
  bool isLaser = false;
  bool isOutputsAreBlocked = false;
  bool isDataValid = true;
  if (nStored > 0) {
    triggers.setTriggers(is_A, is_C, isVertex, is_Central, is_SemiCentral, int8_t(n_hit_A), int8_t(n_hit_C),
                         amplA, amplC, timeA, timeC, isLaser, isOutputsAreBlocked, isDataValid);
    digitsBC.emplace_back(first, nStored, firstBCinDeque, triggers, mEventID - 1);
    digitsTrig.emplace_back(firstBCinDeque, is_A, is_C, isVertex, is_Central, is_SemiCentral);
    size_t const nBC = digitsBC.size();
    for (auto const& lbl : bc.labels) {
      labels.addElement(nBC - 1, lbl);
    }
  }
  // Debug output -------------------------------------------------------------

  LOG(debug) << "Event ID: " << mEventID << ", bc " << firstBCinDeque.bc << ", N hit " << bc.hits.size();
  LOG(debug) << "N hit A: " << int(triggers.getNChanA()) << " N hit C: " << int(triggers.getNChanC()) << " summ ampl A: " << int(triggers.getAmplA())
             << " summ ampl C: " << int(triggers.getAmplC()) << " mean time A: " << triggers.getTimeA()
             << " mean time C: " << triggers.getTimeC() << " nStored " << nStored;

  LOG(debug) << "IS A " << triggers.getOrA() << " IsC " << triggers.getOrC() << " vertex " << triggers.getVertex() << " is Central " << triggers.getCen() << " is SemiCentral " << triggers.getSCen();
}

//------------------------------------------------------------------------
void Digitizer::flush(std::vector<o2::ft0::Digit>& digitsBC,
                      std::vector<o2::ft0::ChannelData>& digitsCh,
                      std::vector<o2::ft0::DetTrigInput>& digitsTrig,
                      o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& labels)
{

  assert(firstBCinDeque <= mIntRecord);

  while (firstBCinDeque < mIntRecord && !mCache.empty()) {
    storeBC(mCache.front(), digitsBC, digitsCh, digitsTrig, labels);
    mCache.pop_front();
    ++firstBCinDeque;
  }
  firstBCinDeque = mIntRecord;
}

void Digitizer::flush_all(std::vector<o2::ft0::Digit>& digitsBC,
                          std::vector<o2::ft0::ChannelData>& digitsCh,
                          std::vector<o2::ft0::DetTrigInput>& digitsTrig,
                          o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& labels)
{

  assert(firstBCinDeque <= mIntRecord);
  ++mEventID;
  while (!mCache.empty()) {
    storeBC(mCache.front(), digitsBC, digitsCh, digitsTrig, labels);
    mCache.pop_front();
    ++firstBCinDeque;
  }
}

void Digitizer::initParameters()
{
  auto const sinc = [](double x) { x *= TMath::Pi(); return (std::abs(x) < 1e-12) ? 1.0 : std::sin(x) / x; };

  // number of noise samples in one BC
  const auto& params = FT0DigParam::Instance();
  mNumNoiseSamples = std::ceil(params.mBunchWidth / params.mNoisePeriod);
  mNoiseSamples.resize(mNumNoiseSamples);

  // set up tables with sinc function values (times noiseVar)
  for (size_t i = 0, n = mSincTable.size(); i < n; ++i) {
    float const time = i / float(n) * params.mNoisePeriod; // [0 .. 1/params.mNoisePeriod)
    LOG(debug) << "initParameters " << i << "/" << n << " " << time;
    // we make a table of sinc values between -num_noise_samples and 2*num_noise_samples
    mSincTable[i].resize(3 * mNumNoiseSamples);
    for (int j = -mNumNoiseSamples; j < 2 * mNumNoiseSamples; ++j) {
      mSincTable[i][mNumNoiseSamples + j] = params.mNoiseVar * sinc((time + 0.5f * params.mBunchWidth) / params.mNoisePeriod - j);
    }
  }
  // set up the lookup table for the signal form
  for (size_t i = 0; i < DP::SIGNAL_TABLE_SIZE; ++i) {
    float const x = float(i) / float(DP::SIGNAL_TABLE_SIZE) * params.mBunchWidth;
    mSignalTable[i] = signalForm_i(x);
  }

  // cache for signal time series used by the CFD -BC/2 .. +3BC/2
  mSignalCache.resize(std::lround(params.mBunchWidth / DP::SIGNAL_CACHE_DT));
}
//_______________________________________________________________________
void Digitizer::init()
{
  LOG(info) << " @@@ Digitizer::init " << std::endl;
  mDeadTimes.fill({InteractionRecord(0), -100.});
  printParameters();
}
//_______________________________________________________________________
void Digitizer::finish()
{
  printParameters();
}

//_______________________________________________________________________
void Digitizer::printParameters() const
{
  const auto& params = FT0DigParam::Instance();
  LOG(info) << " Run Digitzation with parametrs: \n"
            << " CFD amplitude threshold \n " << params.mCFD_trsh << " CFD signal gate in ps \n"
            << params.mTime_trg_gate << "shift to have signal around zero after CFD trancformation  \n"
            << params.mCfdShift << "CFD distance between 0.3 of max amplitude  to max \n"
            << params.mCFDShiftPos << "MIP -> mV " << params.mMip_in_V << " Pe in MIP \n"
            << params.mPe_in_mip << "noise level " << params.mNoiseVar << " noise frequency \n"
            << params.mNoisePeriod << " mMCPs " << params.mMCPs;
}

O2ParamImpl(FT0DigParam);
