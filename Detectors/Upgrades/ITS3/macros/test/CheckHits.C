// Copyright 2020-2022 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CheckHits.C
/// \brief analyze its3 hits
/// \author felix.schlepper@cern.ch

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TH2F.h>
#include <TLegend.h>
#include <TMultiGraph.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TTree.h>
#include <TSystem.h>

#include <array>
#include <cmath>
#include <vector>

#define ENABLE_UPGRADES
#include "CommonConstants/MathConstants.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "ITS3Base/SpecsV2.h"
#include "ITSMFTSimulation/Hit.h"
#include "SimulationDataFormat/MCTrack.h"
#endif

namespace it3c = o2::its3::constants;
namespace it3d = it3c::detID;
using SSAlpide = o2::its3::SegmentationSuperAlpide;
using o2::itsmft::Hit;

constexpr double interaction_rate = 50e3;                                // Hz
constexpr double integration_time = 10e-6;                               // s
constexpr double qedXSection = 34962.1;                                  // barn
constexpr double hadXSection = 8.f;                                      // barn
constexpr double qedRate = qedXSection / hadXSection * interaction_rate; // Hz
constexpr double qedFactor = qedRate * integration_time;                 // a.u.
constexpr std::array<int, 3> colors{
  kRed, kOrange, kBlue};

void CheckHits(bool qed = false,
               std::string hitFileName = "o2sim_HitsIT3.root",
               bool batch = false)
{
  TH1::SetDefaultSumw2();
  gStyle->SetOptStat(0);
  gROOT->SetBatch(batch);
  // Vars
  if (qed) {
    printf("----\n");
    printf("QEDXSection=%f;HadronXSection=%f;QEDRate=%f;QEDFactor=%f\n",
           qedXSection, hadXSection, qedRate, qedFactor);
  }

  // Hits
  std::unique_ptr<TFile> hitFile(TFile::Open(hitFileName.data(), "READ"));
  auto hitTree = hitFile->Get<TTree>("o2sim");
  std::vector<Hit> hitArray, *hitArrayPtr{&hitArray};
  hitTree->SetBranchAddress("IT3Hit", &hitArrayPtr);

  // Hit Plots
  std::array<TH1F*, 3> mHits;
  for (int iLayer{0}; iLayer < 3; ++iLayer) {
    mHits[iLayer] = new TH1F(Form("h_hit_%d", iLayer), Form("L%d;Z (cm);Hit Density (cm^{-2})", iLayer),
                             it3c::segment::nRSUs * 6,
                             -it3c::segment::lengthSensitive / 2.,
                             +it3c::segment::lengthSensitive / 2.);
    mHits[iLayer]->SetMarkerStyle(20 + iLayer);
    mHits[iLayer]->SetLineColor(colors[iLayer]);
    mHits[iLayer]->SetMarkerColor(colors[iLayer]);
  }

  // Hits Loop
  const auto nEvents = hitTree->GetEntriesFast();
  for (int iEntry = 0; hitTree->LoadTree(iEntry) >= 0; ++iEntry) {
    hitTree->GetEntry(iEntry);

    // Hits
    for (const auto& hit : hitArray) {
      if (!it3d::isDetITS3(hit.GetDetectorID())) { // outside of Inner Barrel, e.g. Outer Barrel
        continue;
      }
      auto layer = it3d::getDetID2Layer(hit.GetDetectorID());
      mHits[layer]->Fill(hit.GetZ());
    }
  }

  std::unique_ptr<TFile> oFile(TFile::Open("CheckHits.root", "RECREATE"));
  auto c = new TCanvas();
  c->cd();
  auto leg = new TLegend();
  for (int iLayer{0}; iLayer < 3; ++iLayer) {
    mHits[iLayer]->Scale(1. / (it3c::pixelarray::area * 100));
    mHits[iLayer]->Draw("same");
    mHits[iLayer]->Write();
    leg->AddEntry(mHits[iLayer], Form("Layer %d", iLayer));
  }
  leg->Draw();
  c->Draw();
  c->Write();
  c->SaveAs("it3hits.pdf");
}

// void Plot()
// {
//   std::vector<std::string> centralities{
//     "0.00", "1.57", "2.22", "2.71", "3.13", "3.50", "4.94",
//     "6.05", "6.98", "7.81", "8.55", "9.23", "9.88", "10.47",
//     "11.04", "11.58", "12.09", "12.58", "13.05", "13.52", "13.97",
//     "14.43", "14.96", "15.67", "20.00"};
//   std::vector<std::array<Data, 3>> data;
//   for (auto it = centralities.cbegin(); it != centralities.cend() - 1; ++it) {
//     auto path = "./" + *it + "_" + *(it + 1) + "/";
//     gSystem->cd(path.c_str());
//     gSystem->Exec("pwd");
//     data.push_back(CheckHits());
//     gSystem->cd("..");
//   }
//   for (const auto& elem : data) {
//     std::cout << "+++++++++++++++++++++++++++++++++++++++++\n";
//     for (int i = 0; i < 3; ++i) {
//       std::cout << "===\nLayer " << i << "\n";
//       std::cout << elem[i].max << " +/- " << elem[i].maxE << "\n";
//       std::cout << elem[i].mean << " +/- " << elem[i].meanRMS << "\n";
//     }
//     std::cout << "-----------------------------------------" << std::endl;
//   }
//   const Int_t n = 24;
//   char const* range[n] = {
//     "0-1", "1-2", "2-3", "3-4", "4-5", "5-10", "10-15", "15-20",
//     "20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60",
//     "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95-100"};
//   int i;
//   double x[n];
//   double y0[n], y1[n], y2[n];
//   double vy0[n], vy1[n], vy2[n];
//   double y0M[n], y1M[n], y2M[n];
//   double vy0M[n], vy1M[n], vy2M[n];
//   for (i = 0; i < n; ++i) {
//     x[i] = i;
//     y0[i] = data[i][0].max;
//     y1[i] = data[i][1].max;
//     y2[i] = data[i][2].max;
//     vy0[i] = data[i][0].maxE;
//     vy1[i] = data[i][1].maxE;
//     vy2[i] = data[i][2].maxE;
//     y0M[i] = data[i][0].mean;
//     y1M[i] = data[i][1].mean;
//     y2M[i] = data[i][2].mean;
//     vy0M[i] = data[i][0].meanRMS;
//     vy1M[i] = data[i][1].meanRMS;
//     vy2M[i] = data[i][2].meanRMS;
//   }
//
//   auto c1 = new TCanvas("c1", "A Simple Graph Example", 200, 10, 700, 500);
//   auto h = new TH1F("h", "", n, x[0] - 0.5, x[n - 1] + 0.5);
//   h->SetTitle("Mean hit density per centrality class");
//   h->GetYaxis()->SetTitleOffset(1.);
//   h->GetXaxis()->SetTitleOffset(1.);
//   h->GetYaxis()->SetTitle("hit density (cm^{-2})");
//   h->GetXaxis()->SetTitle("centrality (%)");
//   h->GetXaxis()->SetNdivisions(-10);
//   for (i = 1; i <= n; i++)
//     h->GetXaxis()->SetBinLabel(i, range[i - 1]);
//   h->SetMaximum(70);
//   h->SetMinimum(0);
//   h->SetStats(0);
//   h->Draw("");
//   auto gr0 = new TGraphErrors(n, x, y0, nullptr, vy0);
//   gr0->SetMarkerStyle(4);
//   gr0->SetMarkerSize(0.5);
//   gr0->SetTitle("L0");
//   auto gr1 = new TGraphErrors(n, x, y1, nullptr, vy1);
//   gr1->SetMarkerStyle(4);
//   gr1->SetMarkerSize(0.5);
//   gr1->SetTitle("L1");
//   auto gr2 = new TGraphErrors(n, x, y2, nullptr, vy2);
//   gr2->SetMarkerStyle(4);
//   gr2->SetMarkerSize(0.5);
//   gr2->SetTitle("L2");
//   auto mg = new TMultiGraph("mg", "");
//   mg->Add(gr0);
//   mg->Add(gr1);
//   mg->Add(gr2);
//   mg->Draw("P pmc plc");
//   auto leg = new TLegend(0.75, 0.75, 0.9, 0.9);
//   leg->AddEntry(gr0);
//   leg->AddEntry(gr1);
//   leg->AddEntry(gr2);
//   leg->Draw();
//   c1->SaveAs("its3HitsCentralityMean.pdf");
//
//   c1 = new TCanvas("c1", "A Simple Graph Example", 200, 10, 700, 500);
//   auto hh = new TH1F("hh", "", n, x[0] - 0.5, x[n - 1] + 0.5);
//   hh->SetTitle("Maximum hit density per centrality class per event with RMS");
//   hh->GetYaxis()->SetTitleOffset(1.);
//   hh->GetXaxis()->SetTitleOffset(1.);
//   hh->GetYaxis()->SetTitle("hit density (cm^{-2})");
//   hh->GetXaxis()->SetTitle("centrality (%)");
//   hh->GetXaxis()->SetNdivisions(-10);
//   for (i = 1; i <= n; i++) {
//     hh->GetXaxis()->SetBinLabel(i, range[i - 1]);
//   }
//   hh->SetMaximum(130);
//   hh->SetMinimum(0);
//   hh->SetStats(false);
//   hh->Draw("");
//   auto gr0M = new TGraphErrors(n, x, y0M, nullptr, vy0M);
//   gr0M->SetMarkerStyle(4);
//   gr0M->SetMarkerSize(0.5);
//   gr0M->SetTitle("L0");
//   auto gr1M = new TGraphErrors(n, x, y1M, nullptr, vy1M);
//   gr1M->SetMarkerStyle(4);
//   gr1M->SetMarkerSize(0.5);
//   gr1M->SetTitle("L1");
//   auto gr2M = new TGraphErrors(n, x, y2M, nullptr, vy2M);
//   gr2M->SetMarkerStyle(4);
//   gr2M->SetMarkerSize(0.5);
//   gr2M->SetTitle("L2");
//   auto mgM = new TMultiGraph("mg", "");
//   mgM->Add(gr0M);
//   mgM->Add(gr1M);
//   mgM->Add(gr2M);
//   mgM->Draw("P pmc plc");
//   auto legM = new TLegend(0.75, 0.75, 0.9, 0.9);
//   legM->AddEntry(gr0M);
//   legM->AddEntry(gr1M);
//   legM->AddEntry(gr2M);
//   legM->Draw();
//   c1->SaveAs("its3HitsCentralityMaximum.pdf");
// }
