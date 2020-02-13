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

double Digitizer::get_time(const std::vector<double>& times)
{
  mHist->Reset();
  mHistsum->Reset();
  mHistshift->Reset();

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
        return mHistsum->GetBinCenter(bin);
      }
    }
  }
  std::cout << "CFD failed to find peak\n";
  return 1e10;
}

void Digitizer::process(const std::vector<o2::ft0::HitType>* hits)
{

  auto sorted_hits{*hits};
  std::sort(sorted_hits.begin(), sorted_hits.end(), [](o2::ft0::HitType const& a, o2::ft0::HitType const& b) {
    return a.GetTrackID() < b.GetTrackID();
  });

  //Calculating signal time, amplitude in mean_time +- time_gate --------------
  if (mChannel_times.size() == 0)
    mChannel_times.resize(parameters.mMCPs);
  Int_t parent = -10;
  for (auto& hit : sorted_hits) {
    if (hit.GetEnergyLoss() > 0)
      continue;
    Int_t hit_ch = hit.GetDetectorID();
    Bool_t is_A_side = (hit_ch < 4 * parameters.nCellsA);
    Float_t time_compensate = is_A_side ? parameters.A_side_cable_cmps : parameters.C_side_cable_cmps;
    Double_t hit_time = hit.GetTime() - time_compensate;

    Bool_t is_hit_in_signal_gate = (abs(hit_time) < parameters.mSignalWidth * .5);

    if (is_hit_in_signal_gate) {
      mNumParticles[hit_ch]++;
      mChannel_times[hit_ch].push_back(hit_time);
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
void Digitizer::setDigits(std::vector<o2::ft0::Digit>& digitsBC,
                          std::vector<o2::ft0::ChannelData>& digitsCh)
{

  int n_hit_A = 0, n_hit_C = 0, mean_time_A = 0, mean_time_C = 0;
  int summ_ampl_A = 0, summ_ampl_C = 0;
  int vertex_time;

  int first = digitsCh.size(), nStored = 0;
  for (Int_t ipmt = 0; ipmt < parameters.mMCPs; ++ipmt) {
    if (mNumParticles[ipmt] < parameters.mAmp_trsh)
      continue;
    Float_t amp = (parameters.mMip_in_V * mNumParticles[ipmt] * parameters.mPe_in_mip);
    int chain = (std::rand() % 2) ? 1 : 0;
    if (amp > parameters.mCFD_trsh) {
      int smeared_time = 1000. * (get_time(mChannel_times[ipmt]) - parameters.mCfdShift) * parameters.ChannelWidthInverse;
      if (smeared_time < 1e9) {
        digitsCh.emplace_back(ipmt, smeared_time, int(parameters.mV_2_Nchannels * amp), chain);
        nStored++;
      }
      // fill triggers
      Bool_t is_A_side = (ipmt <= 4 * parameters.nCellsA);
      if (std::abs(smeared_time) > 0.5 * parameters.mTime_trg_gate)
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
  }

  Bool_t is_A, is_C, isVertex, is_Central, is_SemiCentral = 0;
  is_A = n_hit_A > 0;
  is_C = n_hit_C > 0;
  is_Central = summ_ampl_A + summ_ampl_C >= parameters.mtrg_central_trh;
  is_SemiCentral = summ_ampl_A + summ_ampl_C >= parameters.mtrg_semicentral_trh;
  vertex_time = (mean_time_A - mean_time_C) * 0.5;
  isVertex = is_A && is_C && (std::abs(vertex_time) < parameters.mtrg_vertex);
  uint16_t amplA = is_A ? summ_ampl_A : 0;           // sum amplitude A side
  uint16_t amplC = is_C ? summ_ampl_C : 0;           // sum amplitude C side
  uint16_t timeA = is_A ? mean_time_A / n_hit_A : 0; // average time A side
  uint16_t timeC = is_C ? mean_time_C / n_hit_C : 0; // average time C side

  mTriggers.setTriggers(is_A, is_C, isVertex, is_Central, is_SemiCentral, n_hit_A, n_hit_C,
                        amplA, amplC, timeA, timeC);

  digitsBC.emplace_back(first, nStored, mIntRecord, mTriggers);

  // Debug output -------------------------------------------------------------

  LOG(INFO) << "Event ID: " << mEventID;
  LOG(INFO) << "N hit A: " << int(mTriggers.nChanA) << " N hit C: " << int(mTriggers.nChanC) << " summ ampl A: " << int(mTriggers.amplA)
            << " summ ampl C: " << int(mTriggers.amplC) << " mean time A: " << mTriggers.timeA
            << " mean time C: " << mTriggers.timeC;

  LOG(INFO) << "IS A " << mTriggers.getOrA() << " IsC " << mTriggers.getOrC() << " vertex " << mTriggers.getVertex() << " is Central " << mTriggers.getCen() << " is SemiCentral " << mTriggers.getSCen();
}

void Digitizer::initParameters()
{
  mEventTime = 0;
  float signal_width = 0.5 * parameters.mSignalWidth;
  mHist = new TH1F("time_histogram", "", 1000, -signal_width, signal_width);
  mHistsum = new TH1F("time_sum", "", 1000, -signal_width, signal_width);
  mHistshift = new TH1F("time_shift", "", 1000, -signal_width, signal_width);

  mNoisePeriod = parameters.mNoisePeriod / mHist->GetBinWidth(0);
  mBinshift = int(parameters.mCFDShiftPos / (0.001 * parameters.mSignalWidth));
}
//_______________________________________________________________________
void Digitizer::init()
{
  LOG(INFO) << " @@@ Digitizer::init " << std::endl;
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
