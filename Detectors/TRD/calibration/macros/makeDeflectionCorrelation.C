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

/// \file makeDeflectionCorrelation.C
/// \brief Plot the correlation of dy and slope for un/-calibrated tracklets
/// \author Felix Schlepper

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <array>
#include <bitset>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

// ROOT header
#include <TBranch.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TF1.h>
#include <TFile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TLegend.h>
#include <TGraphErrors.h>
#include <TH2.h>
#include <TH2F.h>
#include <THistPainter.h>
#include <TMultiGraph.h>
#include <TLatex.h>
#include <TPaveStats.h>
#include <TStyle.h>

// O2 header
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/NoiseCalibration.h"
#include "DataFormatsTRD/CalVdriftExB.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "TRDQC/Tracking.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"

#endif

#define DY_MAX 1.0
#define DY_MIN -1.0
#define DY_BINS 200
#define DY_CHAMBER_LEN 3.0
#define SNP_MAX 1.0
#define SNP_MIN -1.0
#define SNP_BINS DY_BINS
#define CHAMBER 200
#define NTRACKLETS 3
#define FIT_MIN 1000
#define TRACK_TYPE 0 // 0 = TPC-TRD, 1- ITS-TPC-TRD

using timePoint = o2::parameters::GRPECSObject::timePoint;
static std::vector<std::tuple<double, double, int>> vmap[540];
static bool good_ccdb_chambers[540];
static o2::trd::NoiseStatusMCM* noiseMapPtr;

static void ccdbDownload(unsigned int runNumber, std::string ccdb, timePoint queryInterval)
{
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbMgr.setURL("http://alice-ccdb.cern.ch/");
  auto runDuration = ccdbMgr.getRunDuration(runNumber);
  std::map<std::string, std::string> md;
  md["runNumber"] = std::to_string(runNumber);
  const auto* grp = ccdbMgr.getSpecific<o2::parameters::GRPECSObject>("GLO/Config/GRPECS", (runDuration.first + runDuration.second) / 2, md);
  grp->print();
  const auto startTime = grp->getTimeStart();
  const auto endTime = grp->getTimeEnd();
  ccdbMgr.setURL(ccdb);

  noiseMapPtr = ccdbMgr.get<o2::trd::NoiseStatusMCM>("TRD/Calib/NoiseStatusMCM");
}

static void find_good()
{
  for (int iDet = 0; iDet < 540; ++iDet) {
    good_ccdb_chambers[iDet] = true;
    if (vmap[iDet].size() == 0) {
      continue;
    }
    for (auto& e : vmap[iDet]) {
      if (std::get<0>(e) == 1.0 && std::get<1>(e) == 0.0) {
        good_ccdb_chambers[iDet] = false;
      }
    }
  }
}

inline bool cmpf(float A, float B, float epsilon = 0.0001f)
{
  return (fabs(A - B) < epsilon);
}

// Taken from Tracklet64.cxx
inline float getUncalibratedDy(int slope, int det, float nTbDrift = 19.4)
{
  float padWidth = 0.635f + 0.03f * (det % o2::trd::constants::NLAYER);
  return slope * o2::trd::constants::GRANULARITYTRKLSLOPE * padWidth *
         nTbDrift / o2::trd::constants::ADDBITSHIFTSLOPE;
}

void makeDeflectionCorrelation(unsigned int runNumber = 523677, std::string ccdb = "http://ccdb-test.cern.ch:8080", timePoint queryInterval = 900000)
{
  //----------------------------------------------------
  // Chain and Branch
  //----------------------------------------------------
  TChain chain("qc");
  chain.Add("trdQC*.root");
  std::vector<o2::trd::TrackQC> qc, *qcPtr{&qc};
  chain.SetBranchAddress("trackQC", &qcPtr);

  //----------------------------------------------------
  // CCDB
  //----------------------------------------------------
  ccdbDownload(runNumber, ccdb, queryInterval);
  find_good();

  //----------------------------------------------------
  // Noisemap
  //----------------------------------------------------
  o2::trd::NoiseStatusMCM noiseMap = *noiseMapPtr;

  //----------------------------------------------------
  // Out file
  //----------------------------------------------------
  std::unique_ptr<TFile> outFilePtr(
    TFile::Open("makeDeflectionCorrelation.root", "RECREATE"));

  //----------------------------------------------------
  // Histograms
  //----------------------------------------------------
  auto hcorrelation_cal =
    new TH2F("hcorrelation_cal",
             "Correlation of #psi (#angle Dy) and #phi (#angle Snp) - ALL",
             DY_BINS, DY_MIN, DY_MAX, SNP_BINS, SNP_MIN, SNP_MAX);
  hcorrelation_cal->GetXaxis()->SetTitle("#phi");
  hcorrelation_cal->GetYaxis()->SetTitle("#psi");
  hcorrelation_cal->GetZaxis()->SetTitle("entries");
  auto hcorrelation_uncal =
    new TH2F("hcorrelation_uncal", "Uncalibrated", DY_BINS, DY_MIN, DY_MAX,
             SNP_BINS, SNP_MIN, SNP_MAX);
  hcorrelation_uncal->GetXaxis()->SetTitle("#phi");
  hcorrelation_uncal->GetYaxis()->SetTitle("#psi");
  hcorrelation_uncal->GetZaxis()->SetTitle("entries");

  //----------------------------------------------------
  // Loop - Vars
  //----------------------------------------------------
  std::bitset<6> good;
  std::array<std::pair<TH2F*, TH2F*>, o2::trd::constants::MAXCHAMBER> hall;
  for (int i = 0; i < o2::trd::constants::MAXCHAMBER; ++i) {
    hall[i].first =
      new TH2F(TString::Format("hcorrelation_cal_ch_%d", i),
               TString::Format("Calibrated - Chamber %d", i), DY_BINS, DY_MIN,
               DY_MAX, SNP_BINS, SNP_MIN, SNP_MAX);
    hall[i].second =
      new TH2F(TString::Format("hcorrelation_uncal_ch_%d", i),
               TString::Format("Uncalibrated - Chamber %d", i), DY_BINS,
               DY_MIN, DY_MAX, SNP_BINS, SNP_MIN, SNP_MAX);
  }
  unsigned int chamber;
  std::array<bool, o2::trd::constants::MAXCHAMBER> good_chambers;
  good_chambers.fill(false);

  //----------------------------------------------------
  // Loop
  //----------------------------------------------------
  for (int iEntry = 0; iEntry < chain.GetEntries(); ++iEntry) {
    chain.GetEntry(iEntry);
    printf("Started loop %d: Entries: %ld\n", iEntry, qc.size());
    for (const auto& q : qc) {
      //----------------------------------------------------
      // Cuts
      //----------------------------------------------------
      good.set();
      if (q.type != TRACK_TYPE)
        continue; // type 0 = TPC-TRD, type 1 = ITS-TPC-TRD
      if (q.nTracklets < NTRACKLETS)
        continue;
      for (auto i = 0; i < 6; ++i) {
        if (abs(q.trackX[i]) < 10.0 ||
            noiseMap.getIsNoisy(q.trackletHCId[i], q.trackletRob[i],
                                q.trackletMcm[i]))
          good.reset(i);
      }

      //----------------------------------------------------
      // Plot
      //----------------------------------------------------
      for (Int_t j = 0; j < 6; ++j) {
        if (good[j]) {
          // calc chamber
          chamber = q.trackletHCId[j] / 2;

          // only do this for chambers which have ccdb data
          if (!good_ccdb_chambers[chamber])
            continue;

          float dy_cal = q.trackletDy[j];
          float psi_cal = atan(dy_cal / DY_CHAMBER_LEN);
          float dy_un =
            getUncalibratedDy(q.trackletSlopeSigned[j], q.trackletDet[j]);
          float psi_un = atan(dy_un / DY_CHAMBER_LEN);
          float snp = q.trackSnp[j];
          float phi = asin(snp);
          hcorrelation_cal->Fill(phi, psi_cal);
          hcorrelation_uncal->Fill(phi, psi_un);

          good_chambers[chamber] = true;

          // Fill all
          hall[chamber].first->Fill(phi, psi_cal);
          hall[chamber].second->Fill(phi, psi_un);
        }
      }
    }
  }

  // Find chambers with too few entries
  int i = 0;
  for (auto& h : hall) {
    if (h.first->GetEntries() < FIT_MIN)
      good_chambers[i] = false;
    ++i;
  }
  // ----------------------------------------------------
  // Fits
  // ----------------------------------------------------
  Double_t p0_cal[o2::trd::constants::MAXCHAMBER], p0e_cal[o2::trd::constants::MAXCHAMBER];
  Double_t p1_cal[o2::trd::constants::MAXCHAMBER], p1e_cal[o2::trd::constants::MAXCHAMBER];
  Double_t p0_uncal[o2::trd::constants::MAXCHAMBER], p0e_uncal[o2::trd::constants::MAXCHAMBER];
  Double_t p1_uncal[o2::trd::constants::MAXCHAMBER], p1e_uncal[o2::trd::constants::MAXCHAMBER];
  Double_t x_all[o2::trd::constants::MAXCHAMBER];
  for (int i = 0; i < o2::trd::constants::MAXCHAMBER; ++i) {
    x_all[i] = i;
    if (good_chambers[i] && good_ccdb_chambers[i]) {
      auto fitResCal = hall[i].first->Fit("pol1", "SQC", "", -0.6, -0.2);
      if (fitResCal.Get() != nullptr) {
        p0_cal[i] = fitResCal->Parameter(0);
        p0e_cal[i] = fitResCal->ParError(0);
        p1_cal[i] = fitResCal->Parameter(1);
        p1e_cal[i] = fitResCal->ParError(1);
      } else {
        p0_cal[i] = 0.0;
        p0e_cal[i] = 0.0;
        p1_cal[i] = 0.0;
        p1e_cal[i] = 0.0;
      }

      auto fitResUncal = hall[i].second->Fit("pol1", "SQC", "", -0.6, -0.2);
      if (fitResUncal.Get() != nullptr) {
        p0_uncal[i] = fitResUncal->Parameter(0);
        p0e_uncal[i] = fitResUncal->ParError(0);
        p1_uncal[i] = fitResUncal->Parameter(1);
        p1e_uncal[i] = fitResUncal->ParError(1);
      } else {
        p0_uncal[i] = 0.0;
        p0e_uncal[i] = 0.0;
        p1_uncal[i] = 0.0;
        p1e_uncal[i] = 0.0;
      }
    } else {
      p0_cal[i] = 0.0;
      p0e_cal[i] = 0.0;
      p1_cal[i] = 0.0;
      p1e_cal[i] = 0.0;
      p0_uncal[i] = 0.0;
      p0e_uncal[i] = 0.0;
      p1_uncal[i] = 0.0;
      p1e_uncal[i] = 0.0;
    }
  }
  auto gfit_slope_cal =
    new TGraphErrors(o2::trd::constants::MAXCHAMBER, x_all, p1_cal, nullptr, p1e_cal);
  gfit_slope_cal->SetTitle("Cal: Fit: Slope");
  gfit_slope_cal->GetXaxis()->SetTitle("Chamber");
  gfit_slope_cal->GetYaxis()->SetTitle("Slope");
  gfit_slope_cal->SetLineColor(kRed);
  auto gfit_slope_uncal =
    new TGraphErrors(o2::trd::constants::MAXCHAMBER, x_all, p1_uncal, nullptr, p1e_uncal);
  gfit_slope_uncal->SetTitle("Uncal: Fit: Slope");
  gfit_slope_uncal->GetXaxis()->SetTitle("Chamber");
  gfit_slope_uncal->GetYaxis()->SetTitle("Slope");
  gfit_slope_uncal->SetLineColor(kBlue);
  auto gfit_offset_cal =
    new TGraphErrors(o2::trd::constants::MAXCHAMBER, x_all, p0_cal, nullptr, p0e_cal);
  gfit_offset_cal->SetTitle("Cal: Fit: Offset");
  gfit_offset_cal->GetXaxis()->SetTitle("Chamber");
  gfit_offset_cal->GetYaxis()->SetTitle("Offset");
  gfit_offset_cal->SetLineColor(kRed);
  auto gfit_offset_uncal =
    new TGraphErrors(o2::trd::constants::MAXCHAMBER, x_all, p0_uncal, nullptr, p0e_uncal);
  gfit_offset_uncal->SetTitle("Uncal: Fit: Offset");
  gfit_offset_uncal->GetXaxis()->SetTitle("Chamber");
  gfit_offset_uncal->GetYaxis()->SetTitle("Offset");
  gfit_offset_uncal->SetLineColor(kBlue);

  // ----------------------------------------------------
  // Draw
  // ----------------------------------------------------
  // histos
  outFilePtr->mkdir("all");
  outFilePtr->cd("all");
  gStyle->SetOptStat("e");
  gStyle->SetOptFit(1000);
  auto tex = new TLatex(0.87, 0.98, TString::Format("RUN: %u", runNumber));
  tex->SetNDC();
  tex->SetTextSize(0.02);
  auto c = new TCanvas("PsiVsPhi", "correlation");
  c->Divide(2, 1);
  c->cd(1);
  hcorrelation_cal->Draw("colz1");
  gPad->SetLogz();
  c->cd(2);
  hcorrelation_uncal->Draw("colz1");
  gPad->SetLogz();
  tex->Draw();
  c->Write();

  c = new TCanvas("FitResults", "Fit Results");
  c->Divide(2, 1);
  c->cd();
  c->cd(1);
  gfit_slope_cal->Draw("AP");
  gfit_slope_uncal->Draw("SAME P");
  auto legend = new TLegend(0.1, 0.8, 0.3, 0.9);
  legend->AddEntry(gfit_slope_cal, "Calibrated");
  legend->AddEntry(gfit_slope_uncal, "Uncalibrated");
  legend->Draw();
  c->cd(2);
  gfit_offset_cal->Draw("AP");
  gfit_offset_uncal->Draw("SAME P");
  legend = new TLegend(0.1, 0.8, 0.3, 0.9);
  legend->AddEntry(gfit_offset_cal, "Calibrated");
  legend->AddEntry(gfit_offset_uncal, "Uncalibrated");
  legend->Draw();
  tex->Draw();
  c->Write();
  outFilePtr->cd("..");

  // ----------------------------------------------------
  // Write
  // ----------------------------------------------------
  outFilePtr->mkdir("chambers");
  outFilePtr->cd("chambers");
  for (auto& hist : hall) {
    hist.first->Write();
    hist.second->Write();
  }

  // ----------------------------------------------------
  // No tracklets in chamber
  // ----------------------------------------------------
  printf("'Bad' chambers:\n");
  for (int i = 0; i < o2::trd::constants::MAXCHAMBER; ++i)
    if (!good_chambers[i])
      printf("%i, ", i);
  printf("\n");
}
