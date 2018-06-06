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

#include "TCanvas.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLeaf.h"
#include "TMath.h"
#include "TProfile2D.h"
#include "TRandom.h"
#include <algorithm>
#include <cassert>

using namespace o2::fit;
using o2::fit::Geometry;

ClassImp(Digitizer);

void Digitizer::process(const std::vector<HitType>* hits, std::vector<Digit>* digits)
{
  //parameters constants TO DO: move to class
  constexpr Float_t TimeDiffAC = (Geometry::ZdetA - Geometry::ZdetC) * TMath::C();
  constexpr Double_t C_side_cable_cmps = 2.8; //ns
  constexpr Double_t A_side_cable_cmps = 11.; //ns
  constexpr Double_t BC_clk_center = 25./2.;
  constexpr Double_t signal_width = 5.; // time gate for signal, ns
  constexpr Double_t nPe_in_mip = 400.; // n ph. e. in one mip
  constexpr Double_t CFD_trsh_mip = 0.4; // = 4[mV] / 10[mV/mip]
  constexpr Int_t nMCPs = (Geometry::NCellsA + Geometry::NCellsC) * 4;

  mDigits = digits;

  Double_t digit_timeframe = mEventTime;
  Double_t digit_bc = mEventID;
  Int_t nClk = floor(mEventTime/25.);
  Double_t BCEventTime = mEventTime - 25.*nClk;

  LOG(DEBUG)<< FairLogger::endl<< FairLogger::endl << "Test digizing data ===================" << FairLogger::endl;
  LOG(DEBUG) << "Event ID: " << mEventID << " Event Time " << mEventTime << FairLogger::endl;
  LOG(DEBUG) << "nClk: " << nClk << " BC Event Time " << BCEventTime << " Time dif AC " << TimeDiffAC << FairLogger::endl;






  // Counting photo-electrons, mean time --------------------------------------
  Int_t ch_hit_nPe[nMCPs] = {};
  Double_t ch_hit_mean_time [nMCPs] = {};

  for (auto& hit : *hits) {
    Int_t hit_ch = hit.GetDetectorID();
    Double_t hit_time = hit.GetTime();

    //LOG(DEBUG) << "hit detector ID: " << hit_ch << " Time " << hit_time << FairLogger::endl;

    if(hit_time != 0.)
    {
	ch_hit_nPe[hit_ch]++;
	ch_hit_mean_time[hit_ch] += hit_time;
    }
  }

  for (Int_t ch_iter = 0; ch_iter < nMCPs; ch_iter++)
  {
    if(ch_hit_nPe[ch_iter] != 0)
      ch_hit_mean_time[ch_iter] = ch_hit_mean_time[ch_iter] / (float)ch_hit_nPe[ch_iter];

    //LOG(DEBUG) << "nMCP: " << ch_iter << " n Ph. e. " << ch_hit_nPe[ch_iter] << " mean time " << ch_hit_mean_time[ch_iter] << FairLogger::endl;

  }
  // --------------------------------------------------------------------------






  //Calculating signal time, amplitude in mean_time +- time_gate --------------
  Double_t ch_signal_nPe[nMCPs] = {};
  Double_t ch_signal_MIP[nMCPs] = {};
  Double_t ch_signal_time[nMCPs] = {};

  for (auto& hit : *hits) {
      Int_t hit_ch = hit.GetDetectorID();
      Double_t hit_time = hit.GetTime();
      Bool_t is_hit_in_signal_gate = (hit_time > ch_hit_mean_time[hit_ch] - signal_width*.5) &&
                                     (hit_time < ch_hit_mean_time[hit_ch] + signal_width*.5);

      Bool_t is_A_side = (hit_ch <= 4 * Geometry::NCellsA);
      Double_t time_compensate = is_A_side ? A_side_cable_cmps : C_side_cable_cmps;
      Double_t hit_time_corr = hit_time/1000. - time_compensate + BC_clk_center;// + BCEventTime;
      Double_t is_time_in_gate = (hit_time != 0.);//&&(hit_time_corr > -BC_clk_center)&&(hit_time_corr < BC_clk_center);

    if(is_time_in_gate && is_hit_in_signal_gate)
	{
	    ch_signal_nPe[hit_ch]++;
	    ch_signal_time[hit_ch] += hit_time_corr;
	}
  }

  for (Int_t ch_iter = 0; ch_iter < nMCPs; ch_iter++)
  {

    if(ch_signal_nPe[ch_iter] != 0)
    {
      ch_signal_MIP[ch_iter] = ch_signal_nPe[ch_iter] / nPe_in_mip;
      ch_signal_time[ch_iter] = ch_signal_time[ch_iter] / (float)ch_signal_nPe[ch_iter];
    }
  }
  // --------------------------------------------------------------------------






  //Filling digits array
  for (Int_t ch_iter = 0; ch_iter < nMCPs; ch_iter++)
  {
      if (ch_signal_nPe[ch_iter] > mAmpThreshold) {
	mDigits->emplace_back(digit_timeframe, ch_iter, ch_signal_time[ch_iter], Float_t(ch_signal_nPe[ch_iter]), digit_bc);
      }
    }



  LOG(DEBUG) << "nMCP : :IsA : hit nPe : hit mTime : sig nPe : sig mTime" << FairLogger::endl;
  for (Int_t ch_iter = 0; ch_iter < nMCPs; ch_iter++)
  {
      Bool_t is_A_side = (ch_iter <= 4 * Geometry::NCellsA);
      if(ch_hit_nPe[ch_iter] > 0)
      LOG(DEBUG) << ch_iter << " : " << is_A_side << " : " << ch_hit_nPe[ch_iter] << " : " << ch_hit_mean_time[ch_iter] << " : "
                    << ch_signal_nPe[ch_iter] << " : " << ch_signal_time[ch_iter] << FairLogger::endl;
  }
  LOG(DEBUG) << "======================================\n\n" << FairLogger::endl;





//  Int_t amp[nMCPs] = {};
//  Double_t cfd[nMCPs] = {};

//  for (auto& hit : *hits) {
//    // TODO: put timeframe counting/selection
//    // if (timeframe == mTimeFrameCurrent) {
//    // timeframe = Int_t((mEventTime + hit.GetTime())); // to be replaced with uncalibrated time
//    Int_t mcp = hit.GetDetectorID();
//    Double_t hittime = hit.GetTime();
//    if (mcp > 4 * Geometry::NCellsA)
//      hittime += mTimeDiffAC;
//    if (hittime > mLowTime && hittime < mHighTime) {
//      cfd[mcp] += hittime;
//      amp[mcp]++;
//    }
//  } // end of loop over hits


//  for (Int_t ipmt = 0; ipmt < nMCPs; ipmt++) {
//    if (amp[ipmt] > mAmpThreshold) {
//      cfd[ipmt] = cfd[ipmt] / Float_t(amp[ipmt]); //mean time on 1 quadrant
//      cfd[ipmt] = (gRandom->Gaus(cfd[ipmt], 50)); // Geometry::ChannelWidth;
//      mDigits->emplace_back(timeframe, ipmt, cfd[ipmt], Float_t(amp[ipmt]), bc);
//    }
//  } // end of loop over PMT

}

void Digitizer::initParameters()
{
  mAmpThreshold = 100;
  mLowTime = 10000;
  mHighTime = 12500;
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
