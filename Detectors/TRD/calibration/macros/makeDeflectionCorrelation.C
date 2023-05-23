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
#include <tuple>
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
#include <TLine.h>

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
#define FIT_MIN 1000
#define TRACK_TYPE 15 // 1 = TPC-TRD, 15- ITS-TPC-TRD

using timePoint = o2::parameters::GRPECSObject::timePoint;
static std::vector<std::tuple<double, double, int>> vmap[540];
static bool good_ccdb_chambers[540];
static o2::trd::NoiseStatusMCM* noiseMapPtr;

static void ccdbDownload(unsigned int runNumber, std::string ccdb, std::string noiseMapPath, timePoint queryInterval)
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

  if (noiseMapPath.empty()) {
    noiseMapPtr = ccdbMgr.get<o2::trd::NoiseStatusMCM>("TRD/Calib/NoiseStatusMCM");
  } else {
    std::unique_ptr<TFile> fInNoiseMap(TFile::Open(noiseMapPath.c_str()));
    noiseMapPtr = (o2::trd::NoiseStatusMCM*)fInNoiseMap->Get("map");
  }
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
  ccdbDownload(runNumber, ccdb, noiseMapPath, queryInterval);
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
             SNP_BINS, SNP_MIN, SNP_MAX, DY_BINS, DY_MIN, DY_MAX);
  hcorrelation_cal->GetXaxis()->SetTitle("#phi");
  hcorrelation_cal->GetYaxis()->SetTitle("#psi");
  hcorrelation_cal->GetZaxis()->SetTitle("entries");
  auto hcorrelation_uncal =
    new TH2F("hcorrelation_uncal", "Uncalibrated",
             SNP_BINS, SNP_MIN, SNP_MAX, DY_BINS, DY_MIN, DY_MAX);
  hcorrelation_uncal->GetXaxis()->SetTitle("#phi");
  hcorrelation_uncal->GetYaxis()->SetTitle("#psi");
  hcorrelation_uncal->GetZaxis()->SetTitle("entries");

  //----------------------------------------------------
  // Loop - Vars
  //----------------------------------------------------
  std::bitset<6> good;
  std::array<std::tuple<TH2F*, TH2F*, TH2F*, TH2F*, TH2F*, TH2F*, TGraphErrors*, TGraphErrors*>, o2::trd::constants::MAXCHAMBER> hall;
  for (int i = 0; i < o2::trd::constants::MAXCHAMBER; ++i) {
    std::get<0>(hall[i]) =
      new TH2F(TString::Format("hcorrelation_cal_ch_%d", i),
               TString::Format("Calibrated - Chamber %d", i),
               SNP_BINS, SNP_MIN, SNP_MAX, DY_BINS, DY_MIN, DY_MAX);
    std::get<0>(hall[i])->GetXaxis()->SetTitle("#phi");
    std::get<0>(hall[i])->GetYaxis()->SetTitle("#psi");
    std::get<0>(hall[i])->GetZaxis()->SetTitle("entries");
    std::get<1>(hall[i]) =
      new TH2F(TString::Format("hcorrelation_uncal_ch_%d", i),
               TString::Format("Uncalibrated - Chamber %d", i),
               SNP_BINS, SNP_MIN, SNP_MAX, DY_BINS, DY_MIN, DY_MAX);
    std::get<1>(hall[i])->GetXaxis()->SetTitle("#phi");
    std::get<1>(hall[i])->GetYaxis()->SetTitle("#psi");
    std::get<1>(hall[i])->GetZaxis()->SetTitle("entries");
    std::get<2>(hall[i]) =
      new TH2F(TString::Format("hchi2_da30_cal_%d", i),
               TString::Format("Chi2 vs da Calibrated 30#pm2- Chamber %d", i), 100, -90.0, 90.0, 120,
               0.0, 12.0);
    std::get<2>(hall[i])->GetXaxis()->SetTitle("d#alpha");
    std::get<2>(hall[i])->GetYaxis()->SetTitle("chi2Red");
    std::get<2>(hall[i])->GetZaxis()->SetTitle("entries");
    std::get<3>(hall[i]) =
      new TH2F(TString::Format("hchi2_da30_uncal_%d", i),
               TString::Format("Chi2 vs da Uncalibrated 30#pm2- Chamber %d", i), 100, -90.0, 90.0, 120,
               0.0, 12.0);
    std::get<3>(hall[i])->GetXaxis()->SetTitle("d#alpha");
    std::get<3>(hall[i])->GetYaxis()->SetTitle("chi2Red");
    std::get<3>(hall[i])->GetZaxis()->SetTitle("entries");
    std::get<4>(hall[i]) =
      new TH2F(TString::Format("hchi2_da10_cal_%d", i),
               TString::Format("Chi2 vs da Calibrated 10#pm2- Chamber %d", i), 100, -90.0, 90.0, 120,
               0.0, 12.0);
    std::get<4>(hall[i])->GetXaxis()->SetTitle("d#alpha");
    std::get<4>(hall[i])->GetYaxis()->SetTitle("chi2Red");
    std::get<4>(hall[i])->GetZaxis()->SetTitle("entries");
    std::get<5>(hall[i]) =
      new TH2F(TString::Format("hchi2_da10_uncal_%d", i),
               TString::Format("Chi2 vs da Uncalibrated 10#pm2- Chamber %d", i), 100, -90.0, 90.0, 120,
               0.0, 12.0);
    std::get<5>(hall[i])->GetXaxis()->SetTitle("d#alpha");
    std::get<5>(hall[i])->GetYaxis()->SetTitle("chi2Red");
    std::get<5>(hall[i])->GetZaxis()->SetTitle("entries");
    std::get<6>(hall[i]) = new TGraphErrors();
    std::get<6>(hall[i])->SetNameTitle(TString::Format("gchi2_rms_30_%d", i), TString::Format("Chi2 vs RMS - Chamber %d", i));
    std::get<6>(hall[i])->GetXaxis()->SetTitle("chi2Red");
    std::get<6>(hall[i])->GetYaxis()->SetTitle("d#alpha RMS");
    std::get<7>(hall[i]) = new TGraphErrors();
    std::get<7>(hall[i])->SetNameTitle(TString::Format("gchi2_rms_10_%d", i), TString::Format("Chi2 vs RMS - Chamber %d", i));
    std::get<7>(hall[i])->GetXaxis()->SetTitle("chi2Red");
    std::get<7>(hall[i])->GetYaxis()->SetTitle("d#alpha RMS");
  }
  unsigned int chamber;
  std::array<bool, o2::trd::constants::MAXCHAMBER> good_chambers;
  good_chambers.fill(false);
  auto nEntries = chain.GetEntries();

  //----------------------------------------------------
  // Loop
  //----------------------------------------------------
  for (int iEntry = 0; iEntry < nEntries; ++iEntry) {
    chain.GetEntry(iEntry);
    printf("Started loop %d/%lld: Entries: %ld\n", iEntry, nEntries, qc.size());
    for (const auto& q : qc) {
      //----------------------------------------------------
      // Cuts
      //----------------------------------------------------
      good.set();
      if (q.refGlobalTrackId.getSource() != TRACK_TYPE) {
        continue;
      }
      if (q.trackTRD.getNtracklets() < NTRACKLETS) {
        continue;
      }
      for (auto i = 0; i < 6; ++i) {
        if (q.trackTRD.getTrackletIndex(i) < 0 || noiseMap.isTrackletFromNoisyMCM(q.trklt64[i])) {
          good.reset(i);
        }
      }

      //----------------------------------------------------
      // Plot
      //----------------------------------------------------
      for (Int_t j = 0; j < 6; ++j) {
        if (good[j]) {
          // calc chamber
          chamber = q.trklt64[j].getDetector();

          // only do this for chambers which have ccdb data
          if (!good_ccdb_chambers[chamber])
            continue;

          float dy_cal = q.trkltCalib[j].getDy();
          float psi_cal = atan(dy_cal / DY_CHAMBER_LEN);
          float dy_un = q.trklt64[j].getUncalibratedDy();
          float psi_un = atan(dy_un / DY_CHAMBER_LEN);
          float snp = q.trackProp[j].getSnp();
          float phi = asin(snp);
          hcorrelation_cal->Fill(phi, psi_cal);
          hcorrelation_uncal->Fill(phi, psi_un);

          float eAngle = psi_cal * TMath::RadToDeg();

          good_chambers[chamber] = true;

          // Fill all
          std::get<0>(hall[chamber])->Fill(phi, psi_cal);
          std::get<1>(hall[chamber])->Fill(phi, psi_un);
          if (eAngle >= 28.f && eAngle <= 32.f) {
            float da_cal = (psi_cal - phi) * TMath::RadToDeg();
            float da_un = (psi_un - phi) * TMath::RadToDeg();
            std::get<2>(hall[chamber])->Fill(da_cal, q.reducedChi2);
            std::get<3>(hall[chamber])->Fill(da_un, q.reducedChi2);
          } else if (eAngle >= 8.f && eAngle <= 12.f) {
            float da_cal = (psi_cal - phi) * TMath::RadToDeg();
            float da_un = (psi_un - phi) * TMath::RadToDeg();
            std::get<4>(hall[chamber])->Fill(da_cal, q.reducedChi2);
            std::get<5>(hall[chamber])->Fill(da_un, q.reducedChi2);
          }
        }
      }
    }
  }
  // Project onto X and get RMS
  for (int i = 0; i < o2::trd::constants::MAXCHAMBER; ++i) {
    auto nBins30 = std::get<2>(hall[i])->GetNbinsX();
    for (int iBin = 1; iBin < nBins30; ++iBin) {
      auto chi2 = std::get<2>(hall[i])->GetYaxis()->GetBinCenter(iBin);
      auto proj = std::get<2>(hall[i])->ProjectionX("projX", iBin, iBin + 1);
      auto rms = proj->GetRMS();
      auto rmsError = proj->GetRMSError();
      std::get<6>(hall[i])->AddPoint(chi2, rms);
      std::get<6>(hall[i])->SetPointError(iBin, 0, rmsError);
    }
    auto nBins10 = std::get<4>(hall[i])->GetNbinsX();
    for (int iBin = 1; iBin < nBins10; ++iBin) {
      auto chi2 = std::get<2>(hall[i])->GetYaxis()->GetBinCenter(iBin);
      auto proj = std::get<2>(hall[i])->ProjectionX("projX", iBin, iBin + 1);
      auto rms = proj->GetRMS();
      auto rmsError = proj->GetRMSError();
      std::get<7>(hall[i])->AddPoint(chi2, rms);
      std::get<7>(hall[i])->SetPointError(iBin, 0, rmsError);
    }
  }

  // Find chambers with too few entries
  int i = 0;
  for (auto& h : hall) {
    if (std::get<0>(h)->GetEntries() < FIT_MIN)
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
      auto fitResCal = std::get<0>(hall[i])->Fit("pol1", "SQC", "", -0.6, -0.2);
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

      auto fitResUncal = std::get<1>(hall[i])->Fit("pol1", "SQC", "", -0.6, -0.2);
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
  auto diagLine = new TLine(-1.0, -1.0, 1.0, 1.0);
  diagLine->SetLineColor(kBlue);
  outFilePtr->mkdir("chambers");
  outFilePtr->cd("chambers");
  for (int i = 0; i < o2::trd::constants::MAXCHAMBER; ++i) {
    c = new TCanvas(Form("chamber_%d_cor", i), Form("Chamber %d Correlation", i));
    c->Divide(2, 1);
    c->cd(1);
    std::get<0>(hall[i])->Draw("colz");
    gPad->SetLogz();
    diagLine->Draw();
    c->cd(2);
    std::get<1>(hall[i])->Draw("colz");
    gPad->SetLogz();
    diagLine->Draw();
    c->Write();

    c = new TCanvas(Form("chamber_%d_chi2", i), Form("Chamber %d Chi2", i));
    c->Divide(2, 2);
    c->cd(1);
    std::get<2>(hall[i])->Draw("colz");
    gPad->SetLogz();
    c->cd(2);
    std::get<3>(hall[i])->Draw("colz");
    gPad->SetLogz();
    c->cd(3);
    std::get<4>(hall[i])->Draw("colz");
    gPad->SetLogz();
    c->cd(4);
    std::get<5>(hall[i])->Draw("colz");
    gPad->SetLogz();
    c->Write();

    c = new TCanvas(Form("chamber_%d_rms", i), Form("Chamber %d RMS", i));
    c->Divide(2, 1);
    c->cd(1);
    std::get<6>(hall[i])->Draw("A");
    c->cd(2);
    std::get<7>(hall[i])->Draw("A");

    c->Write();
  }
  outFilePtr->cd("..");

  // ----------------------------------------------------
  // No tracklets in chamber
  // ----------------------------------------------------
  printf("'Bad' chambers:\n");
  for (int i = 0; i < o2::trd::constants::MAXCHAMBER; ++i)
    if (!good_chambers[i])
      printf("%i, ", i);
  printf("\n");
}
