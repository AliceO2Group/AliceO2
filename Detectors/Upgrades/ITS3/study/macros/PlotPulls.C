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

/// \file PlotPulls.C
/// \brief Simple macro to plot ITS3 pulls

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <TROOT.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TF1.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/Track.h"
#include "SimulationDataFormat/MCTrack.h"
#endif

// chi2 PDF with amplitude A, degrees of freedom k, scale s
Double_t chi2_pdf(Double_t* x, Double_t* par)
{
  const Double_t xx = x[0];
  const Double_t A = par[0];
  const Double_t k = par[1];
  const Double_t s = par[2];
  if (xx <= 0.0 || k <= 0.0 || s <= 0.0) {
    return 0.0;
  }
  const Double_t coef = 1.0 / (TMath::Power(2.0 * s, k * 0.5) * TMath::Gamma(k * 0.5));
  return A * coef * TMath::Power(xx, (k * 0.5) - 1.0) * TMath::Exp(-xx / (2.0 * s));
}

void PlotPulls(const char* fName = "its3TrackStudy.root")
{
  TH1::SetDefaultSumw2();
  std::unique_ptr<TFile> inFile(TFile::Open(fName));
  if (!inFile || inFile->IsZombie()) {
    return;
  }
  auto tree = inFile->Get<TTree>("pull");
  if (!tree) {
    return;
  }

  uint8_t src; // track type
  tree->SetBranchAddress("src", &src);
  o2::track::TrackParCov* trk{nullptr};
  tree->SetBranchAddress("trk", &trk);
  o2::track::TrackPar* mcTrk{nullptr};
  tree->SetBranchAddress("mcTrk", &mcTrk);
  o2::MCTrack* part{nullptr};
  tree->SetBranchAddress("mcPart", &part);
  const int nPtBins = 35;
  const double ptLimits[nPtBins] = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.2, 2.5, 3., 4., 5., 6., 8., 10., 15., 20.};
  const int yBins{100}, yRange{5};
  const char* pNames[5] = {"Y", "Z", "Snp", "Tgl", "Q2Pt"};
  auto fGaus = new TF1("fGaus", "[0]*exp(-0.5*((x-[1])/[2])**2)", -3., 3.);

  std::array<TH2F*, o2::track::kNParams> pulls{
    new TH2F("hPullY", "", nPtBins - 1, ptLimits, yBins, -yRange, yRange),
    new TH2F("hPullZ", "", nPtBins - 1, ptLimits, yBins, -yRange, yRange),
    new TH2F("hPullSnp", "", nPtBins - 1, ptLimits, yBins, -yRange, yRange),
    new TH2F("hPullTgl", "", nPtBins - 1, ptLimits, yBins, -yRange, yRange),
    new TH2F("hPullQ2Pt", "", nPtBins - 1, ptLimits, yBins, -yRange, yRange)};

  std::array<TH1F*, o2::track::kNParams> means{
    new TH1F("hPullYMean", "", nPtBins - 1, ptLimits),
    new TH1F("hPullZMean", "", nPtBins - 1, ptLimits),
    new TH1F("hPullSnpMean", "", nPtBins - 1, ptLimits),
    new TH1F("hPullTglMean", "", nPtBins - 1, ptLimits),
    new TH1F("hPullQ2PtMean", "", nPtBins - 1, ptLimits)};

  std::array<TH1F*, o2::track::kNParams> sigmas{
    new TH1F("hPullYSigma", "", nPtBins - 1, ptLimits),
    new TH1F("hPullZSigma", "", nPtBins - 1, ptLimits),
    new TH1F("hPullSnpSigma", "", nPtBins - 1, ptLimits),
    new TH1F("hPullTglSigma", "", nPtBins - 1, ptLimits),
    new TH1F("hPullQ2PtSigma", "", nPtBins - 1, ptLimits)};

  auto calcMahalanobisDist2 = [&](const auto* trk, const auto* mc) -> float {
    o2::math_utils::SMatrix<float, o2::track::kNParams, o2::track::kNParams, o2::math_utils::MatRepSym<float, o2::track::kNParams>> cov;
    cov(o2::track::kY, o2::track::kY) = trk->getSigmaY2();
    cov(o2::track::kZ, o2::track::kY) = trk->getSigmaZY();
    cov(o2::track::kZ, o2::track::kZ) = trk->getSigmaZ2();
    cov(o2::track::kSnp, o2::track::kY) = trk->getSigmaSnpY();
    cov(o2::track::kSnp, o2::track::kZ) = trk->getSigmaSnpZ();
    cov(o2::track::kSnp, o2::track::kSnp) = trk->getSigmaSnp2();
    cov(o2::track::kTgl, o2::track::kY) = trk->getSigmaTglY();
    cov(o2::track::kTgl, o2::track::kZ) = trk->getSigmaTglZ();
    cov(o2::track::kTgl, o2::track::kSnp) = trk->getSigmaTglSnp();
    cov(o2::track::kTgl, o2::track::kTgl) = trk->getSigmaTgl2();
    cov(o2::track::kQ2Pt, o2::track::kY) = trk->getSigma1PtY();
    cov(o2::track::kQ2Pt, o2::track::kZ) = trk->getSigma1PtZ();
    cov(o2::track::kQ2Pt, o2::track::kSnp) = trk->getSigma1PtSnp();
    cov(o2::track::kQ2Pt, o2::track::kTgl) = trk->getSigma1PtTgl();
    cov(o2::track::kQ2Pt, o2::track::kQ2Pt) = trk->getSigma1Pt2();
    if (!cov.Invert()) {
      return -1.f;
    }
    o2::math_utils::SVector<float, o2::track::kNParams> trkPar(trk->getParams(), o2::track::kNParams), mcPar(mc->getParams(), o2::track::kNParams);
    auto res = trkPar - mcPar;
    return ROOT::Math::Similarity(cov, res);
  };

  auto hMahDist2 = new TH1F("hMahDist2", ";Mahalanobis distance 2;n. entries", 100, 0, 10);

  auto getIndex = [](int i) -> int { return i * (i + 3) / 2; };

  for (int iEntry = 0; tree->LoadTree(iEntry) >= 0; ++iEntry) {
    tree->GetEntry(iEntry);
    if (src != o2::dataformats::GlobalTrackID::ITS || std::abs(part->GetPdgCode()) != 211) {
      continue;
    }
    for (int i{0}; i < o2::track::kNParams; ++i) {
      pulls[i]->Fill(part->GetPt(), (trk->getParam(i) - mcTrk->getParam(i)) / std::sqrt(trk->getCov()[getIndex(i)]));
    }
    if (part->GetPt() >= 1.0 && part->GetPt() < 2) {
      if (auto dist = calcMahalanobisDist2(trk, mcTrk); dist >= 0.) {
        hMahDist2->Fill(dist);
      }
    }
  }

  std::vector<TH1D*> projs;
  const char* fitOpt{"QWMERSB"};
  for (int i{0}; i < o2::track::kNParams; ++i) {
    for (auto iPt{0}; iPt < nPtBins - 1; ++iPt) {
      auto hProj = pulls[i]->ProjectionY(Form("%s_%d", pulls[i]->GetName(), iPt), iPt + 1, iPt + 1);
      hProj->SetName(Form("p%s_pt%d", pNames[i], iPt));
      hProj->SetTitle(Form("Pull %s #it{p}_{T}#in[%.2f, %.2f)", pNames[i], ptLimits[iPt], ptLimits[iPt + 1]));
      projs.push_back(hProj);
      if (hProj->GetEntries() < 100) {
        continue;
      }
      fGaus->SetParameter(1, 0);
      fGaus->SetParameter(2, 1);
      auto fRes = hProj->Fit(fGaus, fitOpt);
      if (fRes->IsValid() && fGaus->GetParameter(2) > 0) {
        means[i]->SetBinContent(iPt + 1, fGaus->GetParameter(1));
        means[i]->SetBinError(iPt + 1, fGaus->GetParError(1));
        sigmas[i]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
        sigmas[i]->SetBinError(iPt + 1, fGaus->GetParError(2));
      }
    }
  }

  TF1* fchi2Fit = new TF1("fchi2_fit", chi2_pdf, 0.1, 6, 3);
  fchi2Fit->SetParNames("A", "k", "s");
  fchi2Fit->SetParameter(0, 1);
  fchi2Fit->SetParameter(1, 5);
  fchi2Fit->SetParameter(2, 1);
  if (hMahDist2->Integral("width") > 0.) {
    hMahDist2->Scale(1. / hMahDist2->Integral("width"));
    auto fitres = hMahDist2->Fit(fchi2Fit, "RMQS");
    if (fitres.Get()) {
      fitres->Print();
    }
  }

  TFile outFile("plotPulls.root", "RECREATE");
  for (int i{0}; i < o2::track::kNParams; ++i) {
    pulls[i]->Write();
    means[i]->Write();
    sigmas[i]->Write();
  }
  for (const auto& p : projs) {
    p->Write();
  }
  hMahDist2->Write();
  fchi2Fit->Write();
}
