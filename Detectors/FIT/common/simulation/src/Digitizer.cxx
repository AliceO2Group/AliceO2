// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FITSimulation/Digitizer.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <CommonDataFormat/InteractionRecord.h>

#include "TMath.h"
#include "TRandom.h"
#include <TH1F.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <optional>

using namespace o2::fit;
//using o2::fit::Geometry;

ClassImp(Digitizer);

double Digitizer::get_time(const std::vector<double>& times)
{
  mHist->Reset();
  mHistsum->Reset();
  mHistshift->Reset();

  /// Fill MHistrogram `mHist` with photoelectron induced voltage
  for (auto time : times) {
    for (int bin = mHist->FindBin(time); bin < mHist->GetSize(); ++bin)
      if (mHist->GetBinCenter(bin) > time)
        mHist->AddBinContent(bin, signalForm_i(mHist->GetBinCenter(bin) - time));
  }
  /// Add noise to `mHist`
  for (int c = 0; c < mHist->GetSize(); c += mNoisePeriod) {
    double val = gRandom->Gaus(0, parameters.mNoiseVar);
    for (int bin = 0; bin < mHist->GetSize(); ++bin)
      mHist->AddBinContent(bin, val * sinc(TMath::Pi() * (bin - c) / (double)mNoisePeriod));
    for (int bin = 0; bin < mBinshift; ++bin)
      mHistshift->AddBinContent(bin, val * sinc(TMath::Pi() * (bin - c) / (double)mNoisePeriod));
  }
  /// Shift `mHist` by 1.47 ns to `mHistshift`
  for (int bin = 0; bin < mHist->GetSize() - mBinshift; ++bin)
    mHistshift->SetBinContent(bin + mBinshift, mHist->GetBinContent(bin));

  /// Add the signal and its shifted version to `mHistsum`
  mHist->Scale(-1);
  mHistsum->Add(mHistshift, mHist, 5, 1);
  for (int bin = 1; bin < mHist->GetSize(); ++bin) {
    /// Find the point where zero is crossed in `mHistsum` ...
    if (mHistsum->GetBinContent(bin - 1) < 0 && mHistsum->GetBinContent(bin) >= 0) {
      /// ... and voltage is above 3 mV
      if (std::abs(mHist->GetBinContent(bin)) > 3) {
        //std::cout << "Amp high enough: " << mHist->GetBinContent(bin) << " mV of " << maxBin << " mV at " << mHist->GetBinCenter(bin) << " ns\n";
        return mHistsum->GetBinCenter(bin);
      }
    }
  }
  std::cout << "CFD failed to find peak\n";
  return 1e10;
}

void Digitizer::process(const std::vector<o2::ft0::HitType>* hits, o2::ft0::Digit* digit, std::vector<std::vector<double>>& channel_times)

{

  auto sorted_hits{ *hits };
  std::sort(sorted_hits.begin(), sorted_hits.end(), [](o2::ft0::HitType const& a, o2::ft0::HitType const& b) {
    return a.GetTrackID() < b.GetTrackID();
  });
  digit->setTime(mEventTime);
  digit->setInteractionRecord(mIntRecord);

  //Calculating signal time, amplitude in mean_time +- time_gate --------------
  std::vector<o2::ft0::ChannelData>& channel_data = digit->getChDgData();
  if (channel_data.size() == 0) {
    channel_data.reserve(parameters.mMCPs);
    for (int i = 0; i < parameters.mMCPs; ++i)
      channel_data.emplace_back(o2::ft0::ChannelData{ i, 0, 0, 0 });
  }
  if (channel_times.size() == 0)
    channel_times.resize(parameters.mMCPs);
  Int_t parent = -10;
  assert(digit->getChDgData().size() == parameters.mMCPs);
  for (auto& hit : sorted_hits) {
    if (hit.GetEnergyLoss() > 0)
      continue;
    Int_t hit_ch = hit.GetDetectorID();
    Bool_t is_A_side = (hit_ch < 4 * parameters.NCellsA);
    Float_t time_compensate = is_A_side ? A_side_cable_cmps : C_side_cable_cmps;
    Double_t hit_time = hit.GetTime() - time_compensate;

    Bool_t is_hit_in_signal_gate = (abs(hit_time) < parameters.mSignalWidth * .5);

    if (is_hit_in_signal_gate) {
      channel_data[hit_ch].numberOfParticles++;
      channel_data[hit_ch].QTCAmpl += hit.GetEnergyLoss(); //for FV0
      channel_times[hit_ch].push_back(hit_time);
    }

    //charge particles in MCLabel
    Int_t parentID = hit.GetTrackID();
    if (parentID != parent) {
      o2::ft0::MCLabel label(hit.GetTrackID(), mEventID, mSrcID, hit_ch);
      int lblCurrent;
      if (mMCLabels) {
        lblCurrent = mMCLabels->getIndexedSize(); // this is the size of mHeaderArray;
        mMCLabels->addElement(lblCurrent, label);
      }
      parent = parentID;
    }
  }
}

//------------------------------------------------------------------------
void Digitizer::smearCFDtime(o2::ft0::Digit* digit, std::vector<std::vector<double>> const& channel_times)
{
  //smeared CFD time for 50ps
  //  constexpr Float_t mMip_in_V = 7.;     // mV /250 ph.e.
  //  constexpr Float_t nPe_in_mip = 250.; // n ph. e. in one mip
  std::vector<o2::ft0::ChannelData> mChDgDataArr;
  for (const auto& d : digit->getChDgData()) {
    Int_t mcp = d.ChId;
    Float_t amp = parameters.mMip_in_V * d.numberOfParticles / parameters.mPe_in_mip;
    int numpart = d.numberOfParticles;
    if (amp > parameters.mCFD_trsh_mip) {
      double smeared_time = get_time(channel_times[mcp]) + parameters.mBC_clk_center + mEventTime - parameters.mCfdShift;
      if (smeared_time < 1e9)
        mChDgDataArr.emplace_back(o2::ft0::ChannelData{ mcp, smeared_time, amp, numpart });
    }
  }
  digit->setChDgData(std::move(mChDgDataArr));
}

//------------------------------------------------------------------------
void Digitizer::setTriggers(o2::ft0::Digit* digit)
{

  constexpr Double_t trg_central_trh = 100.;    // mip
  constexpr Double_t trg_semicentral_trh = 50.; // mip
  constexpr Double_t trg_vertex_min = -3.;      //ns
  constexpr Double_t trg_vertex_max = 3.;       //ns

  // Calculating triggers -----------------------------------------------------
  Int_t n_hit_A = 0, n_hit_C = 0;
  Float_t mean_time_A = 0.;
  Float_t mean_time_C = 0.;
  Float_t summ_ampl_A = 0.;
  Float_t summ_ampl_C = 0.;
  Float_t vertex_time;

  Double_t cfd[300] = {};
  Float_t amp[300] = {};
  for (const auto& d : digit->getChDgData()) {
    Int_t mcp = d.ChId;
    cfd[mcp] = d.CFDTime - parameters.mBC_clk_center - mEventTime /*+ parameters.mCfdShift*/;
    amp[mcp] = d.QTCAmpl;
    if (amp[mcp] < parameters.mCFD_trsh_mip)
      continue;
    if (cfd[mcp] < -parameters.mTime_trg_gate / 2. || cfd[mcp] > parameters.mTime_trg_gate / 2.)
      continue;

    Bool_t is_A_side = (mcp <= 4 * parameters.NCellsA);
    if (is_A_side) {
      n_hit_A++;
      summ_ampl_A += amp[mcp];
      mean_time_A += cfd[mcp];
    } else {
      n_hit_C++;
      summ_ampl_C += amp[mcp];
      mean_time_C += cfd[mcp];
    }
  }

  Bool_t is_A = n_hit_A > 0;
  Bool_t is_C = n_hit_C > 0;
  Bool_t is_Central = summ_ampl_A + summ_ampl_C >= trg_central_trh;
  Bool_t is_SemiCentral = summ_ampl_A + summ_ampl_C >= trg_semicentral_trh;

  mean_time_A = is_A ? mean_time_A / n_hit_A : 0;
  mean_time_C = is_C ? mean_time_C / n_hit_C : 0;
  vertex_time = (mean_time_A - mean_time_C) * .5;
  Bool_t is_Vertex = is_A && is_C && (vertex_time > trg_vertex_min) && (vertex_time < trg_vertex_max);

  //filling digit
  digit->setTriggers(is_A, is_C, is_Central, is_SemiCentral, is_Vertex);

  // Debug output -------------------------------------------------------------
  LOG(DEBUG) << "\n\nTest digizing data ===================" << FairLogger::endl;

  LOG(INFO) << "Event ID: " << mEventID << " Event Time " << mEventTime << FairLogger::endl;
  LOG(INFO) << "N hit A: " << n_hit_A << " N hit C: " << n_hit_C << " summ ampl A: " << summ_ampl_A
            << " summ ampl C: " << summ_ampl_C << " mean time A: " << mean_time_A
            << " mean time C: " << mean_time_C << FairLogger::endl;

  LOG(INFO) << "IS A " << is_A << " IS C " << is_C << " is Central " << is_Central
            << " is SemiCentral " << is_SemiCentral << " is Vertex " << is_Vertex << FairLogger::endl;

  LOG(DEBUG) << "======================================\n\n"
             << FairLogger::endl;
  // --------------------------------------------------------------------------
}

void Digitizer::initParameters()
{
  mEventTime = 0;
  float signal_width = 0.5 * parameters.mSignalWidth;
  mHist = new TH1F("time_histogram", "", 1000, -signal_width, signal_width);
  mHistsum = new TH1F("time_sum", "", 1000, -signal_width, signal_width);
  mHistshift = new TH1F("time_shift", "", 1000, -signal_width, signal_width);

  mNoisePeriod = parameters.mNoisePeriod / mHist->GetBinWidth(0);
  mBinshift = int(parameters.mCFDShiftPos / (parameters.mSignalWidth / 1000.));
}
//_______________________________________________________________________
void Digitizer::init()
{
  std::cout << " @@@ Digitizer::init " << std::endl;
}
//_______________________________________________________________________
void Digitizer::finish()
{
  printParameters();
}

void Digitizer::printParameters()
{
  std::cout << " Run Digitzation with parametrs: \n"
            << "mBC_clk_center \n"
            << parameters.mBC_clk_center << " CFD amplitude threshold \n " << parameters.mCFD_trsh_mip << " CFD signal gate in ns \n"
            << parameters.mSignalWidth << "shift to have signal around zero after CFD trancformation  \n"
            << parameters.mCfdShift << "CFD distance between 0.3 of max amplitude  to max \n"
            << parameters.mCFDShiftPos << "MIP -> mV " << parameters.mMip_in_V << " Pe in MIP \n"
            << parameters.mPe_in_mip << "noise level " << parameters.mNoiseVar << " noise frequency \n"
            << parameters.mNoisePeriod << std::endl;
}
