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

Digitizer::CFDOutput Digitizer::get_time(const std::vector<double>& times, double deadTime)
{
  double min_time = std::max(deadTime, *std::min_element(begin(times),
                                                         end(times)));
  assert(std::is_sorted(begin(times), end(times)));
  std::vector<double> noise(std::ceil(parameters.bunchWidth / parameters.mNoisePeriod));
  for (auto& n : noise)
    n = gRandom->Gaus(0, parameters.mNoiseVar);
  bool is_positive = true;
  auto period = parameters.mNoisePeriod;
  auto value_at = [&noise, &times, &period](double time) {
    double val = 0;
    for (double t : times)
      val += signalForm_i(time - t);
    for (size_t i = 0; i < noise.size(); ++i)
      val += noise[i] * sinc(TMath::Pi() * (time - 12.5) / period - i);
    return val;
  };
  //  for (double time = min_time; time < 0.5 * parameters.bunchWidth; time += 0.001 * parameters.channelWidth) {
  CFDOutput result{std::nullopt, -12.5};
  for (double time = min_time; time < 0.5 * parameters.bunchWidth; time += 0.005) {
    double val = value_at(time);
    double cfd_val = 5 * value_at(time - parameters.mCFDShiftPos) - val;
    if (std::abs(val) > parameters.mCFD_trsh && !is_positive && cfd_val > 0) {
      if (!result.particle) {
        result.particle = time;
      }
      result.deadTime = time + parameters.mCFDdeadTime;
      time += parameters.mCFDdeadTime - 0.005;
      is_positive = true;
    } else
      is_positive = cfd_val > 0;
  }
  if (!result.particle) {
    LOG(INFO) << "CFD failed to find peak ";
    // for (double t : times)
    // LOG(INFO) << t << ", dead time "<<deadTime;
  }
  return result;
}

double Digitizer::measure_amplitude(const std::vector<double>& times)
{
  double result = 0;
  double from = parameters.mAmpRecordLow;
  double to = from + parameters.mAmpRecordUp;
  for (double time : times) {
    result += signalForm_integral(to - time) - signalForm_integral(from - time);
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
    Bool_t is_A_side = (hit_ch < 4 * parameters.nCellsA);
    Float_t time_compensate = is_A_side ? parameters.A_side_cable_cmps : parameters.C_side_cable_cmps;
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
  std::vector<double> channel_times;
  for (Int_t ipmt = 0; ipmt < parameters.mMCPs; ++ipmt) {
    auto channel_begin = channel_end;
    channel_end = std::find_if(channel_begin, particles.end(),
                               [ipmt](BCCache::particle const& p) { return p.hit_ch != ipmt; });
    if (channel_end - channel_begin < parameters.mAmp_trsh)
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
    int smeared_time = 1000. * (*cfd.particle - parameters.mCfdShift) * parameters.ChannelWidthInverse;
    bool is_time_in_signal_gate = (smeared_time > -parameters.mTime_trg_gate && smeared_time < parameters.mTime_trg_gate);
    Float_t charge = measure_amplitude(channel_times) * parameters.charge2amp;
    float amp = is_time_in_signal_gate ? parameters.mV_2_Nchannels * charge : 0;
    if (amp > 4095)
      amp = 4095;
    LOG(INFO) << "bc " << firstBCinDeque.bc << ", ipmt " << ipmt << ", smeared_time " << smeared_time << " nStored " << nStored;
    digitsCh.emplace_back(ipmt, smeared_time, int(parameters.mV_2_Nchannels * amp), chain);
    nStored++;

    // fill triggers

    Bool_t is_A_side = (ipmt <= 4 * parameters.nCellsA);
    if (smeared_time > parameters.mTime_trg_gate || smeared_time < -parameters.mTime_trg_gate)
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
  is_Central = summ_ampl_A + summ_ampl_C >= parameters.mtrg_central_trh;
  is_SemiCentral = summ_ampl_A + summ_ampl_C >= parameters.mtrg_semicentral_trh;
  vertex_time = (mean_time_A - mean_time_C) * 0.5;
  isVertex = is_A && is_C && (vertex_time > -parameters.mTime_trg_gate && vertex_time < parameters.mTime_trg_gate);
  uint16_t amplA = is_A ? summ_ampl_A : 0;           // sum amplitude A side
  uint16_t amplC = is_C ? summ_ampl_C : 0;           // sum amplitude C side
  uint16_t timeA = is_A ? mean_time_A / n_hit_A : 0; // average time A side
  uint16_t timeC = is_C ? mean_time_C / n_hit_C : 0; // average time C side

  Triggers triggers;
  triggers.setTriggers(is_A, is_C, isVertex, is_Central, is_SemiCentral, n_hit_A, n_hit_C,
                       int(parameters.mV_2_Nchannels * amplA), int(parameters.mV_2_Nchannels * amplC), timeA, timeC);

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
  // mEventTime = 0;
  float signal_width = 0.5 * parameters.mSignalWidth;
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

void Digitizer::printParameters()
{
  LOG(INFO) << " Run Digitzation with parametrs: \n"
            << " CFD amplitude threshold \n " << parameters.mCFD_trsh << " CFD signal gate in ps \n"
            << parameters.mSignalWidth << "shift to have signal around zero after CFD trancformation  \n"
            << parameters.mCfdShift << "CFD distance between 0.3 of max amplitude  to max \n"
            << parameters.mCFDShiftPos << "MIP -> mV " << parameters.mMip_in_V << " Pe in MIP \n"
            << parameters.mPe_in_mip << "noise level " << parameters.mNoiseVar << " noise frequency \n"
            << parameters.mNoisePeriod << std::endl;
}
