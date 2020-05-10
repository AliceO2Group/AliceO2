// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FT0Simulation/Digitizer.h"
#include "FT0Simulation/DigitizationConstants.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <CommonDataFormat/InteractionRecord.h>

#include "TMath.h"
#include "TRandom.h"
#include <TH1F.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <optional>

using namespace o2::ft0;
//using o2::ft0::Geometry;

ClassImp(Digitizer);

Digitizer::CFDOutput Digitizer::get_time(const std::vector<float>& times, float deadTime)
{
  assert(std::is_sorted(begin(times), end(times)));

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
    int i = 0;
    for (; i < times.size() / Vc::float_v::size(); ++i) {
      tableVal.load(tp);
      tp += Vc::float_v::size();
      Vc::prefetchForOneRead(tp);
      acc += signalFormVc(time - tableVal);
    }
    val += acc.sum();
    // non-SIMD tail
    i *= Vc::float_v::size();
    for (; i < times.size(); ++i) {
      val += signalForm(time - times[i]);
    }
    // (2) add noise
    // find the right indices into the sinc table
    int timeIndex = std::lround(time / mParameters.mNoisePeriod * mSincTable.size());
    int timeOffset = timeIndex / mSincTable.size();
    timeIndex %= mSincTable.size();
    if (timeOffset >= mNumNoiseSamples) { // this happens when time >= 25 ns
      timeOffset = mNumNoiseSamples - 1;
      LOG(DEBUG) << "timeOffset >= mNumNoiseSamples";
    }
    if (timeOffset <= -mNumNoiseSamples) { // this happens when time <= -25 ns
      timeOffset = -mNumNoiseSamples + 1;
      LOG(DEBUG) << "timeOffset <= -mNumNoiseSamples";
    }
    Vc::float_v noiseVal(0);
    const float* np = mNoiseSamples.data();
    acc = 0;
    tp = mSincTable[timeIndex].data() + mNumNoiseSamples - timeOffset;
    i = 0;
    for (; i < mNumNoiseSamples / Vc::float_v::Size; ++i) {
      tableVal.load(tp);
      noiseVal.load(np);
      tp += Vc::float_v::Size;
      Vc::prefetchForOneRead(tp);
      np += Vc::float_v::Size;
      Vc::prefetchForOneRead(np);
      acc += noiseVal * tableVal;
    }
    val += acc.sum(); // horizontal sum
    // non-SIMD tail
    i *= Vc::float_v::Size;
    for (; i < mNumNoiseSamples; ++i) {
      val += mNoiseSamples[i] * *(tp++);
    }
    return val;
  };
  auto const min_time = std::max(deadTime, *std::min_element(std::begin(times),
                                                             std::end(times)));
  CFDOutput result{std::nullopt, -0.5 * mParameters.bunchWidth};
  bool is_positive = true;

  // reset the chache
  std::fill_n(mSignalCache.begin(), mSignalCache.size(), -1.0f);

  // we need double precision for time in order to match previous behaviour
  for (double time = min_time; time < 0.5 * mParameters.bunchWidth; time += DP::SIGNAL_CACHE_DT) {
    float const val = value_at(time);
    int const index = std::lround((time + 0.5 * mParameters.bunchWidth) / DP::SIGNAL_CACHE_DT);
    if (index >= 0 && index < mSignalCache.size()) { // save the value for later use
      mSignalCache[index] = val;
    }
    // look up the time-shifted signal value from the past
    float val_prev = 0.0;
    int const index_prev = std::lround((time - mParameters.mCFDShiftPos + 0.5 * mParameters.bunchWidth) / DP::SIGNAL_CACHE_DT);
    val_prev = ((index_prev < 0 || index_prev >= mSignalCache.size() || mSignalCache[index_prev] < 0.0f)
                  ? value_at(time - mParameters.mCFDShiftPos) //  was not computed before
                  : mSignalCache[index_prev]);                //  is available in the cache
    float const cfd_val = 5 * val_prev - val;
    if (std::abs(val) > mParameters.mCFD_trsh && !is_positive && cfd_val > 0) {
      if (!result.particle) {
        result.particle = time;
      }
      result.deadTime = time + mParameters.mCFDdeadTime;
      time += mParameters.mCFDdeadTime - DP::SIGNAL_CACHE_DT;
      is_positive = true;
    } else {
      is_positive = cfd_val > 0;
    }
  }
  if (!result.particle) {
    LOG(DEBUG) << "CFD failed to find peak ";
    for (auto t : times) {
      LOG(DEBUG) << t << ", dead time " << deadTime;
    }
  }
  return result;
}

double Digitizer::measure_amplitude(const std::vector<float>& times) const
{
  float const from = mParameters.mAmpRecordLow;
  float const to = from + mParameters.mAmpRecordUp;
  // SIMD version has a negligible effect on the total wall time
  Vc::float_v acc(0);
  Vc::float_v tv(0);
  const float* tp = times.data();
  int i = 0;
  for (; i < times.size() / Vc::float_v::Size; ++i) {
    tv.load(tp);
    tp += Vc::float_v::Size;
    Vc::prefetchForOneRead(tp);
    acc += signalForm_integralVc(to - tv) - signalForm_integralVc(from - tv);
  }
  float result = acc.sum(); // horizontal sum
  // non-SIMD tail
  i *= Vc::float_v::Size;
  for (; i < times.size(); ++i) {
    result += signalForm_integral(to - times[i]) - signalForm_integral(from - times[i]);
  }
  return result;
}

void Digitizer::process(const std::vector<o2::ft0::HitType>* hits,
                        std::vector<o2::ft0::Digit>& digitsBC,
                        std::vector<o2::ft0::ChannelData>& digitsCh,
                        o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& label)
{
  ;
  //Calculating signal time, amplitude in mean_time +- time_gate --------------
  flush(digitsBC, digitsCh, label);
  Int_t parent = -10;
  for (auto const& hit : *hits) {
    if (hit.GetEnergyLoss() > 0)
      continue;

    Int_t hit_ch = hit.GetDetectorID();
    Bool_t is_A_side = (hit_ch < 4 * mParameters.nCellsA);
    Float_t time_compensate = is_A_side ? mParameters.A_side_cable_cmps : mParameters.C_side_cable_cmps;
    Double_t hit_time = hit.GetTime() - time_compensate;
    auto relBC = o2::InteractionRecord{hit_time};
    if (mCache.size() <= relBC.bc) {
      mCache.resize(relBC.bc + 1);
    }
    mCache[relBC.bc].hits.emplace_back(BCCache::particle{hit_ch, hit_time - relBC.bc2ns()});

    //charge particles in MCLabel
    Int_t parentID = hit.GetTrackID();
    if (parentID != parent)
      mCache[relBC.bc].labels.emplace(parentID, mEventID, mSrcID, hit_ch);
    parent = parentID;
  }
}

void Digitizer::storeBC(BCCache& bc,
                        std::vector<o2::ft0::Digit>& digitsBC,
                        std::vector<o2::ft0::ChannelData>& digitsCh,
                        o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& labels)
{
  if (bc.hits.empty())
    return;
  int n_hit_A = 0, n_hit_C = 0, mean_time_A = 0, mean_time_C = 0;
  int summ_ampl_A = 0, summ_ampl_C = 0;
  int vertex_time;

  int first = digitsCh.size(), nStored = 0;
  auto& particles = bc.hits;
  std::sort(particles.begin(), particles.end());
  auto channel_end = particles.begin();
  std::vector<float> channel_times;
  for (Int_t ipmt = 0; ipmt < mParameters.mMCPs; ++ipmt) {
    auto channel_begin = channel_end;
    channel_end = std::find_if(channel_begin, particles.end(),
                               [ipmt](BCCache::particle const& p) { return p.hit_ch != ipmt; });
    if (channel_end - channel_begin < mParameters.mAmp_trsh)
      continue;
    channel_times.resize(channel_end - channel_begin);
    std::transform(channel_begin, channel_end, channel_times.begin(), [](BCCache::particle const& p) { return p.hit_time; });
    int chain = (std::rand() % 2) ? 1 : 0;

    auto cfd = get_time(channel_times, mDeadTimes[ipmt].intrec.bc2ns() -
                                         firstBCinDeque.bc2ns() +
                                         mDeadTimes[ipmt].deadTime);
    mDeadTimes[ipmt].intrec = firstBCinDeque;
    mDeadTimes[ipmt].deadTime = cfd.deadTime;

    if (!cfd.particle)
      continue;
    int smeared_time = 1000. * (*cfd.particle - mParameters.mCfdShift) * mParameters.ChannelWidthInverse;
    bool is_time_in_signal_gate = (smeared_time > -mParameters.mTime_trg_gate && smeared_time < mParameters.mTime_trg_gate);
    float charge = measure_amplitude(channel_times) * mParameters.charge2amp;
    float amp = is_time_in_signal_gate ? mParameters.mV_2_Nchannels * charge : 0;
    if (amp > 4095)
      amp = 4095;
    LOG(DEBUG) << "bc " << firstBCinDeque.bc << ", ipmt " << ipmt << ", smeared_time " << smeared_time << " nStored " << nStored;
    digitsCh.emplace_back(ipmt, smeared_time, int(amp), chain);
    nStored++;

    // fill triggers

    Bool_t is_A_side = (ipmt <= 4 * mParameters.nCellsA);
    if (smeared_time > mParameters.mTime_trg_gate || smeared_time < -mParameters.mTime_trg_gate)
      continue;

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
  is_Central = summ_ampl_A + summ_ampl_C >= mParameters.mtrg_central_trh;
  is_SemiCentral = summ_ampl_A + summ_ampl_C >= mParameters.mtrg_semicentral_trh;
  vertex_time = (mean_time_A - mean_time_C) * 0.5;
  isVertex = is_A && is_C && (vertex_time > -mParameters.mTime_trg_gate && vertex_time < mParameters.mTime_trg_gate);
  uint32_t amplA = is_A ? summ_ampl_A : 0;           // sum amplitude A side
  uint32_t amplC = is_C ? summ_ampl_C : 0;           // sum amplitude C side
  uint16_t timeA = is_A ? mean_time_A / n_hit_A : 0; // average time A side
  uint16_t timeC = is_C ? mean_time_C / n_hit_C : 0; // average time C side

  Triggers triggers;
  triggers.setTriggers(is_A, is_C, isVertex, is_Central, is_SemiCentral, int8_t(n_hit_A), int8_t(n_hit_C),
                       amplA, amplC, timeA, timeC);

  digitsBC.emplace_back(first, nStored, firstBCinDeque, triggers, mEventID);
  size_t const nBC = digitsBC.size();
  for (auto const& lbl : bc.labels)
    labels.addElement(nBC - 1, lbl);

  // Debug output -------------------------------------------------------------

  LOG(INFO) << "Event ID: " << mEventID << ", bc " << firstBCinDeque.bc << ", N hit " << bc.hits.size();
  LOG(INFO) << "N hit A: " << int(triggers.nChanA) << " N hit C: " << int(triggers.nChanC) << " summ ampl A: " << int(triggers.amplA)
            << " summ ampl C: " << int(triggers.amplC) << " mean time A: " << triggers.timeA
            << " mean time C: " << triggers.timeC;

  LOG(INFO) << "IS A " << triggers.getOrA() << " IsC " << triggers.getOrC() << " vertex " << triggers.getVertex() << " is Central " << triggers.getCen() << " is SemiCentral " << triggers.getSCen();
}

//------------------------------------------------------------------------
void Digitizer::flush(std::vector<o2::ft0::Digit>& digitsBC,
                      std::vector<o2::ft0::ChannelData>& digitsCh,
                      o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& labels)
{
  LOG(DEBUG) << "firstBCinDeque " << firstBCinDeque << " mIntRecord " << mIntRecord;
  assert(firstBCinDeque <= mIntRecord);
  while (firstBCinDeque < mIntRecord && !mCache.empty()) {
    storeBC(mCache.front(), digitsBC, digitsCh, labels);
    mCache.pop_front();
    ++firstBCinDeque;
  }
  firstBCinDeque = mIntRecord;
}

void Digitizer::flush_all(std::vector<o2::ft0::Digit>& digitsBC,
                          std::vector<o2::ft0::ChannelData>& digitsCh,
                          o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& labels)
{
  LOG(INFO) << "firstBCinDeque " << firstBCinDeque << " mIntRecord " << mIntRecord;
  assert(firstBCinDeque <= mIntRecord);
  while (!mCache.empty()) {
    storeBC(mCache.front(), digitsBC, digitsCh, labels);
    mCache.pop_front();
    ++firstBCinDeque;
  }
}

void Digitizer::initParameters()
{
  auto const sinc = [](double x) { x *= TMath::Pi(); return (std::abs(x) < 1e-12) ? 1.0 : std::sin(x) / x; };

  // number of noise samples in one BC
  mNumNoiseSamples = std::ceil(mParameters.bunchWidth / mParameters.mNoisePeriod);
  mNoiseSamples.resize(mNumNoiseSamples);

  // set up tables with sinc function values (times noiseVar)
  for (size_t i = 0, n = mSincTable.size(); i < n; ++i) {
    double const time = i / double(n) * mParameters.mNoisePeriod; // [0 .. 1/mParameters.mNoisePeriod)
    std::cout << "initParameters " << i << "/" << n << " " << time << std::endl;
    // we make a table of sinc values between -num_noise_samples and 2*num_noise_samples
    mSincTable[i].resize(3 * mNumNoiseSamples);
    for (int j = -mNumNoiseSamples; j < 2 * mNumNoiseSamples; ++j) {
      mSincTable[i][mNumNoiseSamples + j] = mParameters.mNoiseVar * sinc((time + 0.5 * mParameters.bunchWidth) / mParameters.mNoisePeriod - j);
    }
  }
  // set up the lookup table for the signal form
  for (int i = 0, n = mSignalTable.size(); i < n; ++i) {
    double const x = i / double(n) * mParameters.bunchWidth;
    mSignalTable[i] = signalForm_i(x);
  }

  // cache for signal time series used by the CFD -BC/2 .. +3BC/2
  mSignalCache.resize(std::lround(2 * mParameters.bunchWidth / DP::SIGNAL_CACHE_DT));
}
//_______________________________________________________________________
void Digitizer::init()
{
  LOG(INFO) << " @@@ Digitizer::init " << std::endl;
  mDeadTimes.fill({InteractionRecord(0), -100.});
}
//_______________________________________________________________________
void Digitizer::finish()
{
  printParameters();
}

void Digitizer::printParameters() const
{
  LOG(INFO) << " Run Digitzation with parametrs: \n"
            << " CFD amplitude threshold \n " << mParameters.mCFD_trsh << " CFD signal gate in ps \n"
            << mParameters.mSignalWidth << "shift to have signal around zero after CFD trancformation  \n"
            << mParameters.mCfdShift << "CFD distance between 0.3 of max amplitude  to max \n"
            << mParameters.mCFDShiftPos << "MIP -> mV " << mParameters.mMip_in_V << " Pe in MIP \n"
            << mParameters.mPe_in_mip << "noise level " << mParameters.mNoiseVar << " noise frequency \n"
            << mParameters.mNoisePeriod << std::endl;
}
