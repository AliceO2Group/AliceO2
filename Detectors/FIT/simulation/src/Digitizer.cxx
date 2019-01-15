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
#include "FITSimulation/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <CommonDataFormat/InteractionRecord.h>

#include "TMath.h"
#include "TRandom.h"
#include <algorithm>
#include <cassert>
#include <iostream>

using namespace o2::fit;
using o2::fit::Geometry;

ClassImp(Digitizer);

void Digitizer::process(const std::vector<HitType>* hits, Digit* digit)
{
  //parameters constants TO DO: move to class

  constexpr Float_t C_side_cable_cmps = 2.8; //ns
  constexpr Float_t A_side_cable_cmps = 11.; //ns
  constexpr Float_t signal_width = 5.;       // time gate for signal, ns
  constexpr Float_t nPe_in_mip = 250.;       // n ph. e. in one mip

  Int_t nlbl = 0; //number of MCtrues

  digit->setTime(mEventTime);
  digit->setBC(mBC);
  digit->setOrbit(mOrbit);

  //Calculating signal time, amplitude in mean_time +- time_gate --------------
  Double_t cfd[300] = {};
  Float_t amp[300] = {};
  Int_t ch_signal_nPe[300] = {};
  Double_t ch_signal_MIP[300] = {};
  Double_t ch_signal_time[300] = {};
  for (auto& hit : *hits) {
    Int_t hit_ch = hit.GetDetectorID();
    Double_t hit_time = hit.GetTime();
    Bool_t is_A_side = (hit_ch <= 4 * Geometry::NCellsA);
    Float_t time_compensate = is_A_side ? A_side_cable_cmps : C_side_cable_cmps;

    Bool_t is_hit_in_signal_gate = (hit_time > time_compensate - signal_width * .5) &&
                                   (hit_time < time_compensate + signal_width * .5);

    Double_t hit_time_corr = hit_time - time_compensate /* + mBC_clk_center + mEventTime*/;

    if (/*is_time_in_gate &&*/ is_hit_in_signal_gate) {
      ch_signal_nPe[hit_ch]++;
      ch_signal_time[hit_ch] += hit_time_corr;
    }

    //charge particles in MCLabel
    if (hit.GetEnergyLoss() > 0) {
      o2::fit::MCLabel label(hit.GetTrackID(), mEventID, mSrcID, hit_ch);
      int lblCurrent;
      if (mMCLabels) {
        lblCurrent = mMCLabels->getIndexedSize(); // this is the size of mHeaderArray;
        mMCLabels->addElement(lblCurrent, label);
        nlbl++;
      }
    }
  }

  // sum  different sources
  std::vector<ChannelData> mChDgDataArr;
  for (const auto& d : digit->getChDgData()) {
    Int_t mcp = d.ChId;
    cfd[mcp] = d.CFDTime;
    amp[mcp] = d.QTCAmpl;
  }

  for (Int_t ch_iter = 0; ch_iter < mMCPs; ch_iter++) {
    if (ch_signal_nPe[ch_iter] != 0) {
      ch_signal_MIP[ch_iter] = amp[ch_iter] + ch_signal_nPe[ch_iter] / nPe_in_mip;
      if (cfd[ch_iter] > 0) {
        cfd[ch_iter] = cfd[ch_iter] - mBC_clk_center - mEventTime;
        ch_signal_time[ch_iter] = ((cfd[ch_iter] + ch_signal_time[ch_iter] / (float)ch_signal_nPe[ch_iter]) / 2.) + mBC_clk_center + mEventTime;
      } else
        ch_signal_time[ch_iter] = (ch_signal_time[ch_iter] / (float)ch_signal_nPe[ch_iter]) + mBC_clk_center + mEventTime;

      if (ch_signal_MIP[ch_iter] > mCFD_trsh_mip) {
        LOG(DEBUG) << ch_iter << " : "
                   << " : " << ch_signal_time[ch_iter] - mBC_clk_center - mEventTime << " : "
                   << ch_signal_MIP[ch_iter] << " : " << mEventTime << " cfd " << cfd[ch_iter] << FairLogger::endl;
        mChDgDataArr.emplace_back(ChannelData{ ch_iter, ch_signal_time[ch_iter], ch_signal_MIP[ch_iter] });
      } else {
        ch_signal_MIP[ch_iter] = 0;
        ch_signal_time[ch_iter] = 0;
      }
    }
  }

  digit->setChDgData(std::move(mChDgDataArr));
}

//------------------------------------------------------------------------
void Digitizer::smearCFDtime(Digit* digit)
{
  //smeared CFD time for 50ps
  std::vector<ChannelData> mChDgDataArr;
  for (const auto& d : digit->getChDgData()) {
    Int_t mcp = d.ChId;
    Double_t cfd = d.CFDTime - mBC_clk_center - mEventTime;
    Float_t amp = d.QTCAmpl;
    if (amp > mCFD_trsh_mip) {
      Double_t smeared_time = gRandom->Gaus(cfd, 0.050) + mBC_clk_center + mEventTime;
      mChDgDataArr.emplace_back(ChannelData{ mcp, smeared_time, amp });
    }
  }
  digit->setChDgData(std::move(mChDgDataArr));
}

//------------------------------------------------------------------------
void Digitizer::setTriggers(Digit* digit)
{
  constexpr Double_t BC_clk_center = 12.5;      // clk center
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
    cfd[mcp] = d.CFDTime - mBC_clk_center - mEventTime;
    amp[mcp] = d.QTCAmpl;
    if (amp[mcp] < mCFD_trsh_mip)
      continue;
    if (cfd[mcp] < -mTime_trg_gate / 2. || cfd[mcp] > mTime_trg_gate / 2.)
      continue;

    Bool_t is_A_side = (mcp <= 4 * Geometry::NCellsA);
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
  vertex_time = (mean_time_A + mean_time_C) * .5;
  Bool_t is_Vertex = (vertex_time > trg_vertex_min) && (vertex_time < trg_vertex_max);

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
  mBC_clk_center = 12.5; // clk center
  mMCPs = (Geometry::NCellsA + Geometry::NCellsC) * 4;
  mCFD_trsh_mip = 0.4; // = 4[mV] / 10[mV/mip]
  mTime_trg_gate = 4.; // ns
  mAmpThreshold = 100;
  mEventTime = 0;
  // murmur
}
//_______________________________________________________________________
void Digitizer::init() {}

//_______________________________________________________________________
void Digitizer::finish() {}
/*
void Digitizer::printParameters()
{
  //murmur
}
*/
