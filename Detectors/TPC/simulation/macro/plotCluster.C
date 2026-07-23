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

/// \file plotCluster.C
/// \brief Plots the merged 83mKr charge spectrum from tpcBoxClusters.root with a staged multi-Gaussian fit
/// \author Ankur Yadav <ankur.yadav@cern.ch>

// Reads tpcBoxClusters.root (produced by o2-tpc-krypton-clusterer) and plots
// the merged 83mKr charge spectrum with a staged multi-Gaussian fit.
//
// Usage:
//   root -b -q 'plotCluster.C("tpcBoxClusters.root")'
//   root -b -q 'plotCluster.C("tpcBoxClusters.root","IROC")'
//
// rocSel:      "IROC" | "OROC1" | "OROC2" | "OROC3" | "OROC" | "ALL"
// sectorSel:   -1 = all sectors merged (default); 0-35 = one sector
// qualityCuts: true (default) -- sigma-based cluster-shape quality cuts
// outputPdf:   "" = auto-named kr_<roc>_<sector>.pdf
//
// ROC row boundaries (O2 TPC, local pad row within sector):
//   IROC  rows   0 -  62  (63 rows)
//   OROC1 rows  63 -  96  (34 rows)
//   OROC2 rows  97 - 126  (30 rows)
//   OROC3 rows 127 - 151  (25 rows)

#include "DataFormatsTPC/KrCluster.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TFile.h"
#include "TFitResult.h"
#include "TH1F.h"
#include "TLatex.h"
#include "TLine.h"
#include "TMath.h"
#include "TSystem.h"
#include "TTree.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

static const int kIrocLast = 62;
static const int kOroc1Last = 96;
static const int kOroc2Last = 126;
static const int kOroc3Last = 151;

static int getRocIndex(float meanRow)
{
  int r = TMath::Nint(meanRow);
  if (r <= kIrocLast) {
    return 0;
  }
  if (r <= kOroc1Last) {
    return 1;
  }
  if (r <= kOroc2Last) {
    return 2;
  }
  if (r <= kOroc3Last) {
    return 3;
  }
  return -1;
}

static bool inRocSel(int rocIdx, const std::string& roc)
{
  if (roc == "IROC") {
    return rocIdx == 0;
  }
  if (roc == "OROC1") {
    return rocIdx == 1;
  }
  if (roc == "OROC2") {
    return rocIdx == 2;
  }
  if (roc == "OROC3") {
    return rocIdx == 3;
  }
  if (roc == "OROC") {
    return rocIdx >= 1 && rocIdx <= 3;
  }
  return true; // ALL
}

// Staged multi-Gaussian + exponential+erfc background fit of the 83mKr
// spectrum's six characteristic peaks (9.4, 12.6, 19.6, 29.1, 32.2, 41.6 keV).
// Parameters [0..15]:
//   [0],[1]     expo background (ln A, slope)
//   [2]         erfc shoulder amplitude (centre/width tied to the 41.6 keV peak)
//   [3],[4]      9.4 keV Gaussian  (amp, mu)
//   [5],[6]     12.6 keV Gaussian  (amp, mu)
//   [7],[8]     19.6 keV Gaussian  (amp, mu)
//   [9],[10]    29.1 keV Gaussian  (amp, mu)
//   [11],[12]   32.2 keV Gaussian  (amp, mu)
//   [13..15]    41.6 keV Gaussian (main; amp, mu=p[14], sigma=p[15])
// Width constraint: sigma_i = p[15] * sqrt(E_i/41.6); only p[15] is free
// (Fano-limited resolution: sigma/E ~ 1/sqrt(E), so sigma ~ sqrt(E)).
static void fitKrSpectrum(TH1F* h, double mu41)
{
  const double ratio[6] = {9.4 / 41.6, 12.6 / 41.6, 19.6 / 41.6, 29.1 / 41.6, 32.2 / 41.6, 1.0};
  const char* lbl[6] = {"T2#gamma 9.4 keV", "K#alpha 12.6 keV", "19.6 keV", "29.1 keV", "32.2 keV", "41.6 keV (main)"};
  const int col[6] = {kOrange + 2, kGreen + 2, kViolet + 1, kMagenta, kCyan + 2, kRed};
  // Satellite i (0..4) lives at parameters [ampIdx(i)], [muIdx(i)];
  // the main (41.6 keV) peak is amp=p[13], mu=p[14], sigma=p[15].
  auto ampIdx = [](int i) { return 3 + 2 * i; };
  auto muIdx = [](int i) { return 4 + 2 * i; };
  const int kAmpMain = 13, kMuMain = 14, kSigMain = 15;
  const int kNPar = 16;

  double mu[6], sig0[6];
  for (int i = 0; i < 6; i++) {
    mu[i] = mu41 * ratio[i];
    sig0[i] = mu[i] * 0.08;
  }

  double amp41 = TMath::Max(h->GetBinContent(h->FindBin(mu41)), 1.);
  const double alo[6] = {amp41 / 50., amp41 / 50., amp41 / 50., amp41 / 50., amp41 / 50., amp41 * 0.30};
  const double ahi[6] = {amp41 * 1.0, amp41 * 1.0, amp41 * 1.0, amp41 * 1.0, amp41 * 1.0, amp41 * 3.00};

  const double slo[6] = {mu[0] * 0.05, mu[1] * 0.05, mu[2] * 0.05, mu[3] * 0.04, mu[4] * 0.04, mu[5] * 0.03};
  const double shi[6] = {mu[0] * 0.30, mu[1] * 0.28, mu[2] * 0.30, mu[3] * 0.25, mu[4] * 0.25, mu[5] * 0.20};

  const double xlo = mu41 * 0.11, xhi = mu41 * 1.20;

  TF1* fbg = new TF1("fbg_pre", "expo", mu41 * 0.07, mu41 * 0.17);
  h->Fit(fbg, "RQN0");

  const double rlo[6] = {mu41 * 0.175, mu41 * 0.265, mu41 * 0.370, mu41 * 0.625, mu41 * 0.720, mu41 * 0.865};
  const double rhi[6] = {mu41 * 0.280, mu41 * 0.365, mu41 * 0.570, mu41 * 0.760, mu41 * 0.835, mu41 * 1.180};
  TF1* fg[6];
  for (int i = 0; i < 6; i++) {
    fg[i] = new TF1(Form("fg_pre%d", i), "gaus", rlo[i], rhi[i]);
    fg[i]->SetParameters(TMath::Max(h->GetBinContent(h->FindBin(mu[i])), 1.), mu[i], sig0[i]);
    fg[i]->SetParLimits(0, alo[i], ahi[i]);
    fg[i]->SetParLimits(1, mu[i] * 0.90, mu[i] * 1.10);
    fg[i]->SetParLimits(2, slo[i], shi[i]);
    h->Fit(fg[i], "RQN0");
  }

  TF1* total = new TF1(
    "fitTotal",
    [](double* x, double* p) -> double {
      static const double r[5] = {9.4 / 41.6, 12.6 / 41.6, 19.6 / 41.6, 29.1 / 41.6, 32.2 / 41.6};
      double val = TMath::Exp(p[0] + p[1] * x[0]);
      val += p[2] * TMath::Erfc((x[0] - p[14]) / (TMath::Sqrt2() * p[15]));
      for (int i = 0; i < 5; i++) {
        val += p[3 + 2 * i] * TMath::Gaus(x[0], p[4 + 2 * i], p[15] * TMath::Sqrt(r[i]), false);
      }
      val += p[13] * TMath::Gaus(x[0], p[14], p[15], false);
      return val;
    },
    xlo, xhi, kNPar);
  total->SetNpx(3000);
  total->SetParameter(0, fbg->GetParameter(0));
  total->SetParameter(1, fbg->GetParameter(1));
  double erfcSeed = h->GetBinContent(h->FindBin(mu41 * 0.905)) * 0.4;
  total->SetParameter(2, erfcSeed > 1. ? erfcSeed : 1.);
  for (int i = 0; i < 6; i++) {
    double aSeed = TMath::Min(TMath::Max(fg[i]->GetParameter(0), alo[i]), ahi[i]);
    total->SetParameter(i < 5 ? ampIdx(i) : kAmpMain, aSeed);
    total->SetParameter(i < 5 ? muIdx(i) : kMuMain, fg[i]->GetParameter(1));
  }
  total->SetParameter(kSigMain, TMath::Abs(fg[5]->GetParameter(2)));

  total->SetParLimits(1, -0.02, 0.);
  total->SetParLimits(2, 0., 1e9);
  for (int i = 0; i < 5; i++) {
    total->SetParLimits(ampIdx(i), alo[i], ahi[i]);
    total->SetParLimits(muIdx(i), mu[i] * 0.90, mu[i] * 1.10);
  }
  total->SetParLimits(kAmpMain, alo[5], ahi[5]);
  total->SetParLimits(kMuMain, mu[5] * 0.90, mu[5] * 1.10);
  total->SetParLimits(kSigMain, mu[5] * 0.03, mu[5] * 0.20);

  total->SetLineColor(kBlack);
  total->SetLineWidth(2);
  TFitResultPtr r = h->Fit(total, "RS");

  // Explicitly draw the combined total fit (sum of background+erfc+all
  // Gaussians) as its own visible curve -- the individual component draws
  // below are mathematically identical pieces of this same function, but
  // without this the combined shape isn't directly checkable by eye.
  total->SetNpx(3000);
  total->Draw("SAME");

  TF1* fbgDraw = new TF1("fbg_draw", "exp([0]+[1]*x)+[2]*erfc((x-[3])/(sqrt(2.0)*[4]))", xlo, xhi);
  fbgDraw->SetParameters(total->GetParameter(0), total->GetParameter(1), total->GetParameter(2),
                         total->GetParameter(kMuMain), total->GetParameter(kSigMain));
  fbgDraw->SetNpx(3000);
  fbgDraw->SetLineColor(kGray + 1);
  fbgDraw->SetLineStyle(7);
  fbgDraw->SetLineWidth(2);
  fbgDraw->Draw("SAME");

  const double sigma41 = TMath::Abs(total->GetParameter(kSigMain));
  for (int i = 0; i < 6; i++) {
    int ai = (i < 5) ? ampIdx(i) : kAmpMain;
    int mi = (i < 5) ? muIdx(i) : kMuMain;
    double sigmaI = sigma41 * TMath::Sqrt(ratio[i]); // = sigma41 * sqrt(E_i/41.6)
    TF1* gc = new TF1(Form("gc%d", i), "[0]*exp(-0.5*pow((x-[1])/[2],2))", xlo, xhi);
    gc->SetParameters(total->GetParameter(ai), total->GetParameter(mi), sigmaI);
    gc->SetNpx(3000);
    gc->SetLineColor(col[i]);
    gc->SetLineStyle(2);
    gc->SetLineWidth(2);
    gc->Draw("SAME");
    double px = total->GetParameter(mi), py = total->GetParameter(ai);
    if (py > h->GetMaximum() * 0.015) {
      TLatex* tx = new TLatex(px + mu41 * 0.008, py * 0.65, lbl[i]);
      tx->SetTextSize(0.028);
      tx->SetTextColor(col[i]);
      tx->Draw();
    }
  }

  double chi2ndf = (r->Ndf() > 0) ? r->Chi2() / r->Ndf() : -1.;
  printf("\n==== Kr-83m spectrum fit [%s] ====\n", h->GetTitle());
  printf("  41.6 keV seed  : %.1f ADC\n", mu41);
  printf("  sigma_41 (free): %.1f +/- %.1f ADC  (%.2f%%)\n", sigma41, total->GetParError(kSigMain),
         100. * sigma41 / total->GetParameter(kMuMain));
  printf("  chi2/ndf       : %.2f\n", chi2ndf);
  printf("  %-22s  %14s  %14s  %8s\n", "Peak", "mu_fit [ADC]", "sigma (sqrtE)", "reso [%]");
  printf("  %-22s  %14s  %14s  %8s\n", "----", "------------", "-------------", "--------");
  for (int i = 0; i < 6; i++) {
    int mi = (i < 5) ? muIdx(i) : kMuMain;
    double muI = total->GetParameter(mi);
    double sigmaI = sigma41 * TMath::Sqrt(ratio[i]);
    double reso = (muI > 0.) ? 100. * sigmaI / muI : -1.;
    printf("  %-22s  %7.1f +/- %4.1f  %12.1f  %7.2f%%\n", lbl[i], muI, total->GetParError(mi), sigmaI, reso);
  }
  printf("==============================================\n\n");
}

void plotCluster(const char* inputFile = "tpcBoxClusters.root", const char* rocSel = "IROC", int sectorSel = -1,
                 bool qualityCuts = true, const char* outputPdf = "")
{
  gSystem->Load("libO2TPCReconstruction");
  gSystem->Load("libO2DataFormatsTPC");
  gStyle->SetOptStat(0); // stat box hides the peaks otherwise

  std::string roc(rocSel);
  for (auto& c : roc) {
    c = toupper(c);
  }
  if (roc != "IROC" && roc != "OROC1" && roc != "OROC2" && roc != "OROC3" && roc != "OROC" && roc != "ALL") {
    std::cout << "Invalid rocSel '" << rocSel << "'. Choose: IROC OROC1 OROC2 OROC3 OROC ALL" << std::endl;
    return;
  }
  if (sectorSel < -1 || sectorSel > 35) {
    std::cout << "Invalid sectorSel " << sectorSel << ". Use -1 (all) or 0-35." << std::endl;
    return;
  }

  auto f = TFile::Open(inputFile);
  if (!f || f->IsZombie()) {
    std::cout << "Cannot open " << inputFile << std::endl;
    return;
  }
  auto t = (TTree*)f->Get("Clusters");
  if (!t) {
    std::cout << "No 'Clusters' tree in " << inputFile << std::endl;
    return;
  }
  if (!t->GetBranch("TPCBoxCluster_0")) {
    std::cout << "Expected DPL branches (TPCBoxCluster_N) not found in " << inputFile << std::endl;
    return;
  }

  std::cout << "Input    : " << inputFile << std::endl;
  std::cout << "TFs      : " << t->GetEntries() << std::endl;
  std::cout << "ROC sel  : " << roc << std::endl;
  std::cout << "Sector   : " << (sectorSel < 0 ? "all" : Form("%d", sectorSel)) << std::endl;
  std::cout << "QC cuts  : " << (qualityCuts ? "ON" : "OFF") << std::endl;

  auto passQC = [&](const o2::tpc::KrCluster& c) -> bool {
    if (!qualityCuts) {
      return true;
    }
    return (c.sigmaTime > 0.1f && c.sigmaTime < 1.8f) &&
           (c.sigmaRow > 0.2f && c.sigmaRow < 0.6f + c.totCharge / 4000.f) && (c.sigmaPad > 0.1f && c.sigmaPad < 1.2f);
  };

  std::vector<o2::tpc::KrCluster>* secCls[36] = {};
  for (int s = 0; s < 36; s++) {
    t->SetBranchAddress(Form("TPCBoxCluster_%d", s), &secCls[s]);
  }

  const int nBins = 400;
  const double xMax = 6000.;
  const double binW = xMax / nBins; // 15 ADC
  std::string mergeTitle =
    (sectorSel >= 0) ? Form("Sector %d -- %s", sectorSel, roc.c_str()) : Form("All sectors -- %s", roc.c_str());
  TH1F* hMerged = new TH1F("hMerged", Form("%s;Total cluster charge (ADC counts);Entries / %.0f ADC", mergeTitle.c_str(), binW),
                           nBins, 0., xMax);
  hMerged->SetDirectory(nullptr);

  long long nTotal = 0, nQCPass = 0, nMerged = 0;
  for (Long64_t ev = 0; ev < t->GetEntries(); ++ev) {
    t->GetEntry(ev);
    for (int s = 0; s < 36; s++) {
      if (!secCls[s]) {
        continue;
      }
      for (auto& c : *secCls[s]) {
        ++nTotal;
        if (!passQC(c)) {
          continue;
        }
        int rocIdx = getRocIndex(c.meanRow);
        if (rocIdx < 0) {
          continue;
        }
        ++nQCPass;
        if (inRocSel(rocIdx, roc) && (sectorSel < 0 || s == sectorSel)) {
          hMerged->Fill(c.totCharge);
          ++nMerged;
        }
      }
    }
  }

  printf("Total clusters read : %lld\n", nTotal);
  printf("QC-passed clusters  : %lld (%.1f%%)\n", nQCPass, nTotal > 0 ? 100. * nQCPass / nTotal : 0.);
  printf("In canvas selection : %lld\n", nMerged);

  if (nMerged == 0) {
    std::cout << "No clusters in canvas selection." << std::endl;
    return;
  }

  double mu41 = -1.;
  {
    int mBin = hMerged->FindBin(1000.);
    for (int b = mBin + 1; b <= hMerged->GetNbinsX(); b++) {
      if (hMerged->GetBinContent(b) > hMerged->GetBinContent(mBin)) {
        mBin = b;
      }
    }
    mu41 = hMerged->GetBinCenter(mBin);
  }
  const bool goodSpectrum = (mu41 >= 1200.);
  printf("41.6 keV seed peak  : %.0f ADC%s\n", mu41, goodSpectrum ? " [OK]" : " [WRONG SPECTRUM]");

  std::string pdfName;
  if (std::string(outputPdf).empty()) {
    std::string secStr = (sectorSel < 0) ? "allsec" : Form("s%02d", sectorSel);
    pdfName = Form("kr_%s_%s.pdf", roc.c_str(), secStr.c_str());
    for (auto& c : pdfName) {
      c = tolower(c);
    }
  } else {
    pdfName = outputPdf;
  }

  TCanvas* cfit = new TCanvas("cfit", mergeTitle.c_str(), 1100, 700);
  cfit->SetLeftMargin(0.10);
  cfit->SetRightMargin(0.05);
  cfit->SetBottomMargin(0.12);

  hMerged->SetLineColor(kBlue + 1);
  hMerged->SetLineWidth(2);
  hMerged->Draw("HIST");

  const double ratio[6] = {9.4 / 41.6, 12.6 / 41.6, 19.6 / 41.6, 29.1 / 41.6, 32.2 / 41.6, 1.0};
  const int pcol[6] = {kOrange + 2, kGreen + 2, kViolet + 1, kMagenta, kCyan + 2, kRed};
  if (goodSpectrum) {
    fitKrSpectrum(hMerged, mu41);
  } else {
    for (int i = 0; i < 6; i++) {
      double xexp = mu41 * ratio[i];
      if (xexp < 50 || xexp > 5900) {
        continue;
      }
      TLine* l = new TLine(xexp, 0, xexp, hMerged->GetMaximum() * 0.8);
      l->SetLineColor(pcol[i]);
      l->SetLineStyle(3);
      l->SetLineWidth(1);
      l->Draw();
    }
    TLatex* msg = new TLatex(0.15, 0.85, Form("WRONG SPECTRUM: max at %.0f ADC (expected > 1200)", mu41));
    msg->SetNDC();
    msg->SetTextColor(kRed);
    msg->SetTextSize(0.038);
    msg->Draw();
  }

  TLatex* tlab = new TLatex(0.12, 0.92, Form("#bf{%s}", mergeTitle.c_str()));
  tlab->SetNDC();
  tlab->SetTextSize(0.038);
  tlab->Draw();
  TLatex* nlab = new TLatex(0.88, 0.92, Form("N_{sel}=%lld", nMerged));
  nlab->SetNDC();
  nlab->SetTextSize(0.033);
  nlab->SetTextAlign(31);
  nlab->Draw();

  cfit->SaveAs(pdfName.c_str());
  printf("Saved %s\n", pdfName.c_str());
}
