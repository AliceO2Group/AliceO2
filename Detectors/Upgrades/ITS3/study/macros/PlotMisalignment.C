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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TF1.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile2D.h>
#include <TCanvas.h>
#include <TTree.h>
#include <TStyle.h>
#include <TMath.h>
#include "ReconstructionDataFormats/Track.h"
#include <cmath>
#endif

constexpr int kNLay = 8; // slots: PV(-1) + layers 0-6
constexpr int kNVar = 2; // dY, dZ
constexpr int kNPar = 5; // Y, Z, Snp, Tgl, Q2Pt
// constexpr int kNPtBins = 6; // integrated + 5 differential
// const float kPtEdges[kNPtBins] = {0., 0.3, 0.8, 2., 5., 10.};
// const char* kPtLabels[kNPtBins] = {"", "0.0<p_{T}<0.3", "0.3<p_{T}<0.8", "0.8<p_{T}<2.0", "2.0<p_{T}<5.0", "5.0<p_{T}<10.0"};
// const char* kPtTags[kNPtBins] = {"", "_pt0_3", "_pt3_8", "_pt8_20", "_pt20_50", "_pt50_100"};
constexpr int kNPtBins = 1; // integrated + 5 differential
const float kPtEdges[kNPtBins] = {0.};
const char* kPtLabels[kNPtBins] = {""};
const char* kPtTags[kNPtBins] = {""};
const char* kVarNames[kNVar] = {"dY", "dZ"};
const char* kVarTitles[kNVar] = {"d_{Y} (#mum)", "d_{Z} (#mum)"};
const char* kParNames[kNPar] = {"Y", "Z", "Snp", "Tgl", "Q2Pt"};
const int kCovIdx[kNPar] = {0, 2, 5, 9, 14};

int getPtBin(float pt)
{
  for (int i = 0; i < kNPtBins - 1; i++) {
    if (pt >= kPtEdges[i] && pt < kPtEdges[i + 1]) {
      return i + 1; // 1-indexed, 0 = integrated
    }
  }
  return -1;
}

void processTree(TFile* f, const char* treeName)
{
  auto* tree = f->Get<TTree>(treeName);
  if (!tree) {
    return;
  }

  // branch variables
  float dY, dZ, phi, eta;
  int lay;
  auto* trk = new o2::track::TrackParCov;
  auto* mcTrk = new o2::track::TrackPar;
  tree->SetBranchAddress("dY", &dY);
  tree->SetBranchAddress("dZ", &dZ);
  tree->SetBranchAddress("phi", &phi);
  tree->SetBranchAddress("eta", &eta);
  tree->SetBranchAddress("lay", &lay);
  tree->SetBranchAddress("trk", &trk);
  tree->SetBranchAddress("mcTrk", &mcTrk);

  // --- book histograms ---
  // [ptBin][lay] for each plot type
  // dY/dZ vs phi
  TH2F* hVsPhi[kNVar][kNPtBins][kNLay];
  // dY/dZ vs eta
  TH2F* hVsEta[kNVar][kNPtBins][kNLay];
  // profile2D phi vs eta
  TProfile2D* hProf[kNVar][kNPtBins][kNLay];
  // pulls
  TH1F* hPull[kNPar][kNPtBins][kNLay];

  for (int ipt = 0; ipt < kNPtBins; ipt++) {
    for (int ilay = 0; ilay < kNLay; ilay++) {
      for (int iv = 0; iv < kNVar; iv++) {
        hVsPhi[iv][ipt][ilay] = new TH2F(
          Form("%s_%s_phi%s_l%d", treeName, kVarNames[iv], kPtTags[ipt], ilay),
          Form("Layer %d %s;#phi (rad);%s", ilay - 1, kPtLabels[ipt], kVarTitles[iv]),
          100, 0, 2 * TMath::Pi(), 100, -100, 100);
        hVsEta[iv][ipt][ilay] = new TH2F(
          Form("%s_%s_eta%s_l%d", treeName, kVarNames[iv], kPtTags[ipt], ilay),
          Form("Layer %d %s;#eta;%s", ilay - 1, kPtLabels[ipt], kVarTitles[iv]),
          100, -1.5, 1.5, 100, -100, 100);
        hProf[iv][ipt][ilay] = new TProfile2D(
          Form("%s_%s_prof%s_l%d", treeName, kVarNames[iv], kPtTags[ipt], ilay),
          Form("Layer %d %s;#phi (rad);#eta;#LT%s#GT", ilay - 1, kPtLabels[ipt], kVarTitles[iv]),
          50, 0, 2 * TMath::Pi(), 50, -1.5, 1.5);
      }
      for (int ip = 0; ip < kNPar; ip++) {
        hPull[ip][ipt][ilay] = new TH1F(
          Form("%s_pull_%s%s_l%d", treeName, kParNames[ip], kPtTags[ipt], ilay),
          Form("Layer %d %s;pull_{%s};counts", ilay - 1, kPtLabels[ipt], kParNames[ip]),
          100, -5, 5);
      }
    }
  }

  // --- fill loop ---
  const Long64_t nEntries = tree->GetEntries();
  for (Long64_t i = 0; i < nEntries; i++) {
    tree->GetEntry(i);
    if (i % 100000 == 0) {
      std::cout << "Progress: " << i << "/" << nEntries << " (" << (100.0 * i / nEntries) << "%)" << std::endl;
    }

    int ilay = lay + 1;
    float pt = trk->getPt();
    float dYum = dY * 10000.f;
    float dZum = dZ * 10000.f;

    // integrated (ipt=0) + differential
    int iptDiff = getPtBin(pt);
    for (int ipt : {0, iptDiff}) {
      if (ipt < 0) {
        continue;
      }
      for (int iv = 0; iv < kNVar; iv++) {
        float val = (iv == 0) ? dYum : dZum;
        hVsPhi[iv][ipt][ilay]->Fill(phi, val);
        hVsEta[iv][ipt][ilay]->Fill(eta, val);
        hProf[iv][ipt][ilay]->Fill(phi, eta, val);
      }
      for (int ip = 0; ip < kNPar; ip++) {
        float sigma2 = trk->getDiagError2(ip);
        if (sigma2 > 0) {
          hPull[ip][ipt][ilay]->Fill((trk->getParam(ip) - mcTrk->getParam(ip)) / std::sqrt(sigma2));
        }
      }
    }
  }

  // --- draw & save ---
  auto drawSliceFits = [](TH2F* h) {
    h->FitSlicesY(nullptr, 0, -1, 5);
    auto* hMean = (TH1D*)gDirectory->Get(Form("%s_1", h->GetName()));
    auto* hSigma = (TH1D*)gDirectory->Get(Form("%s_2", h->GetName()));
    if (hMean && hSigma) {
      for (auto* hh : {hMean, hSigma}) {
        hh->SetMarkerStyle(20);
        hh->SetMarkerSize(0.6);
      }
      hMean->SetMarkerColor(kRed);
      hMean->SetLineColor(kRed);
      hSigma->SetMarkerColor(kOrange + 1);
      hSigma->SetLineColor(kOrange + 1);
      hMean->Draw("same");
      hSigma->Draw("same");
    }
  };

  for (int ipt = 0; ipt < kNPtBins; ipt++) {
    // dY/dZ vs phi
    for (int iv = 0; iv < kNVar; iv++) {
      auto* c = new TCanvas(Form("%s_%s_vs_phi%s", treeName, kVarNames[iv], kPtTags[ipt]),
                            Form("%s vs #phi %s", kVarNames[iv], kPtLabels[ipt]), 800, 1600);
      c->Divide(2, 4);
      for (int ilay = 0; ilay < kNLay; ilay++) {
        c->cd(ilay + 1);
        gPad->SetRightMargin(0.13);
        hVsPhi[iv][ipt][ilay]->Draw("col");
        drawSliceFits(hVsPhi[iv][ipt][ilay]);
      }
      c->SaveAs(Form("%s.png", c->GetName()));
    }

    // dY/dZ vs eta
    for (int iv = 0; iv < kNVar; iv++) {
      auto* c = new TCanvas(Form("%s_%s_vs_eta%s", treeName, kVarNames[iv], kPtTags[ipt]),
                            Form("%s vs #eta %s", kVarNames[iv], kPtLabels[ipt]), 800, 1600);
      c->Divide(2, 4);
      for (int ilay = 0; ilay < kNLay; ilay++) {
        c->cd(ilay + 1);
        gPad->SetRightMargin(0.13);
        hVsEta[iv][ipt][ilay]->Draw("col");
        drawSliceFits(hVsEta[iv][ipt][ilay]);
      }
      c->SaveAs(Form("%s.png", c->GetName()));
    }

    // profile2D
    for (int iv = 0; iv < kNVar; iv++) {
      auto* c = new TCanvas(Form("%s_%s_prof2d%s", treeName, kVarNames[iv], kPtTags[ipt]),
                            Form("%s #phi vs #eta %s", kVarNames[iv], kPtLabels[ipt]), 800, 1600);
      c->Divide(2, 4);
      for (int ilay = 0; ilay < kNLay; ilay++) {
        c->cd(ilay + 1);
        gPad->SetRightMargin(0.15);
        hProf[iv][ipt][ilay]->Draw("colz");
        hProf[iv][ipt][ilay]->GetZaxis()->SetRangeUser(-100, 100);
      }
      c->SaveAs(Form("%s.png", c->GetName()));
    }

    // pulls
    for (int ilay = 0; ilay < kNLay; ilay++) {
      auto* c = new TCanvas(Form("%s_pulls_l%d%s", treeName, ilay, kPtTags[ipt]),
                            Form("Pulls layer %d %s", ilay - 1, kPtLabels[ipt]), 1200, 800);
      c->Divide(3, 2);
      for (int ip = 0; ip < kNPar; ip++) {
        c->cd(ip + 1);
        hPull[ip][ipt][ilay]->Draw();
        if (hPull[ip][ipt][ilay]->GetEntries() > 20) {
          hPull[ip][ipt][ilay]->Fit("gaus", "Q");
        }
      }
      c->SaveAs(Form("%s.png", c->GetName()));
    }
  }
}

void PlotMisalignment(const char* fname = "its3TrackStudy.root")
{
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(1);
  auto f = TFile::Open(fname);
  processTree(f, "idealRes");
  processTree(f, "misRes");
}
