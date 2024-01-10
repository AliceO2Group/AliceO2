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
#include <numeric>
#include <utility>
#include <vector>

#endif
#define ENABLE_UPGRADES
#include "CommonConstants/MathConstants.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "ITSMFTSimulation/Hit.h"
#include "SimulationDataFormat/MCTrack.h"

using o2::itsmft::Hit;

constexpr int nLayers = 3;
constexpr double interaction_rate = 50e3;                                // Hz
constexpr double integration_time = 10e-6;                               // s
constexpr double qedXSection = 34962.1;                                  // barn
constexpr double hadXSection = 8.f;                                      // barn
constexpr double qedRate = qedXSection / hadXSection * interaction_rate; // Hz
constexpr double qedFactor = qedRate * integration_time;                 // a.u.
static constexpr std::array<double, 4> radii{1.9, 2.52, 3.15, 999};
static const o2::its3::SegmentationSuperAlpide ssAlpide(0, 22.e-4, 22.e-4, 66.e-4, 26., radii.data());

struct Maximum {
  double y;
  double vy;
  Maximum(double y, double vy) : y{y}, vy{vy} {}
};

struct Data {
  double max;
  double maxE;
  double mean;
  double meanRMS;
};

void checkFile(const std::unique_ptr<TFile>& file);
TH2D* makeMap(int layer, std::string n);

std::array<Data, 3> CheckHits(bool qed = false, std::string hitFileName = "o2sim_HitsIT3.root",
                              std::string kineFileName = "o2sim_Kine.root",
                              bool batch = true)
{
  gROOT->SetBatch(batch);
  // Vars
  const int nHemis = nLayers * 2;

  if (qed) {
    printf("----\n");
    printf("QEDXSection=%f;HadronXSection=%f;QEDRate=%f;QEDFactor=%f\n",
           qedXSection, hadXSection, qedRate, qedFactor);
  }

  // Hits
  std::unique_ptr<TFile> hitFile(TFile::Open(hitFileName.data()));
  checkFile(hitFile);
  auto hitTree = hitFile->Get<TTree>("o2sim");
  std::vector<Hit> hitArray, *hitArrayPtr{&hitArray};
  hitTree->SetBranchAddress("IT3Hit", &hitArrayPtr);

  // Kine
  std::unique_ptr<TFile> kineFile(TFile::Open(kineFileName.data()));
  checkFile(kineFile);
  auto kineTree = kineFile->Get<TTree>("o2sim");
  std::vector<o2::MCTrack> trackArray, *trackArrayPtr{&trackArray};
  kineTree->SetBranchAddress("MCTrack", &trackArrayPtr);

  // Hit Plots
  auto hitsXY =
    new TH2F("h_hitsXY", "Hits XY", 1600, -5.f, 5.f, 1600, -5.f, 5.f);
  hitsXY->GetXaxis()->SetTitle("x  (cm)");
  hitsXY->GetYaxis()->SetTitle("y  (cm)");
  auto hitsZY =
    new TH2F("h_hitsZY", "Hits ZY", 1600, -30.f, 30.f, 1600, -5.f, 5.f);
  hitsZY->GetXaxis()->SetTitle("z (cm)");
  hitsZY->GetYaxis()->SetTitle("y (cm)");
  std::array<TH2D*, nLayers> avgHitMaps{};
  for (int i = 0; i < nLayers; ++i) {
    avgHitMaps[i] = makeMap(i, "");
  }
  std::array<TH1D*, nLayers> avgHitMapsProj{};
  auto densMG = new TMultiGraph("g_densMG", "Hit Density along z");
  std::vector<std::array<TH2D*, nLayers>> eventHitMaps;
  std::vector<std::array<TH1D*, nLayers>> eventHitMapsProj;
  std::array<std::vector<double>, nLayers> eventMaxes;
  // Hits Loop
  const auto nEvents = hitTree->GetEntriesFast();
  for (int iEntry = 0; hitTree->LoadTree(iEntry) >= 0; ++iEntry) {
    if (!qed) {
      std::array<TH2D*, nLayers> maps{};
      for (int i = 0; i < nLayers; ++i) {
        maps[i] = makeMap(i, Form("_event_%d", iEntry));
      }
      eventHitMaps.push_back(maps);
    }
    hitTree->GetEntry(iEntry);

    // Hits
    for (const auto& hit : hitArray) {
      if (hit.GetDetectorID() >=
          nHemis) { // outside of Inner Barrel, e.g. Outer Barrel
        continue;
      }
      hitsXY->Fill(hit.GetX(), hit.GetY());
      hitsZY->Fill(hit.GetZ(), hit.GetY());
      auto z = hit.GetZ();
      auto phi = std::atan(hit.GetX() / (hit.GetY())) +
                 o2::constants::math::PI / 2.f;
      if (hit.GetDetectorID() == 0) {
        auto rphi = phi * ssAlpide.mRadii[0];
        if (!qed) {
          eventHitMaps.back()[0]->Fill(z, rphi);
        }
        avgHitMaps[0]->Fill(z, rphi);
      } else if (hit.GetDetectorID() == 2) {
        auto rphi = phi * ssAlpide.mRadii[1];
        if (!qed) {
          eventHitMaps.back()[1]->Fill(z, rphi);
        }
        avgHitMaps[1]->Fill(z, rphi);
      } else if (hit.GetDetectorID() == 4) {
        auto rphi = phi * ssAlpide.mRadii[2];
        if (!qed) {
          eventHitMaps.back()[2]->Fill(z, rphi);
        }
        avgHitMaps[2]->Fill(z, rphi);
      }
    }

    if (!qed) {
      std::array<TH1D*, nLayers> projs{};
      for (int iLayer = 0; iLayer < nLayers; ++iLayer) {
        double binArea = eventHitMaps.back()[iLayer]->GetXaxis()->GetBinWidth(1) *
                         eventHitMaps.back()[iLayer]->GetYaxis()->GetBinWidth(1);
        eventHitMaps.back()[iLayer]->Scale(1.f / binArea);
        projs[iLayer] = eventHitMaps.back()[iLayer]->ProjectionX("_px", 0, -1, "e");
        projs[iLayer]->Scale(1. / eventHitMaps.back()[iLayer]->GetNbinsY());
        projs[iLayer]->SetMarkerStyle(21);
        projs[iLayer]->SetTitle(Form("L%d", iLayer));
        projs[iLayer]->GetXaxis()->SetTitle("z (cm)");
        projs[iLayer]->GetYaxis()->SetTitle("hit density (cm^{-2})");
        eventMaxes[iLayer].push_back(projs[iLayer]->GetMaximum());
      }
      eventHitMapsProj.push_back(projs);
    }
  }

  // Kinematic Plots
  auto kineN =
    new TH1D("h_kineN", "Multiplicity Stable particles", 100, -5.f, 5.f);
  kineN->GetXaxis()->SetTitle("#eta");
  kineN->GetYaxis()->SetTitle("#frac{dN}{d#eta}");
  auto kineN05 = new TH1D("h_kineN05", "#||{#eta}<0.5", 50, -0.5f, 0.5f);
  kineN05->GetXaxis()->SetTitle("#eta");
  kineN05->GetYaxis()->SetTitle("#frac{dN}{d#eta}");

  // Kinematics Loop
  for (int iEntry = 0; kineTree->LoadTree(iEntry) >= 0; ++iEntry) {
    kineTree->GetEntry(iEntry);

    for (const auto& track : trackArray) { // MCTrack
      // primary particles
      auto vx = track.Vx();
      auto vy = track.Vy();
      if (vx * vx + vy * vy > 0.01) {
        continue;
      }
      auto id = TMath::Abs(track.GetPdgCode());
      if (id != 2212 && // proton
          id != 211 &&  // pion
          id != 321     // kaon
      ) {
        continue;
      }
      kineN->Fill(track.GetEta());
      if (TMath::Abs(track.GetEta()) < 0.5) {
        kineN05->Fill(track.GetEta());
      }
    }
  }
  kineN->Scale(1.f / (kineN->GetXaxis()->GetBinWidth(1) * nEvents));
  kineN05->Scale(1.f / (kineN05->GetXaxis()->GetBinWidth(1) * nEvents));
  printf("----\n");
  printf("Kinematics: Abs(eta)<0.5=%.2f\n", kineN05->Integral("width"));

  // Projections
  for (int i = 0; i < nLayers; ++i) {
    double binArea = avgHitMaps[i]->GetXaxis()->GetBinWidth(1) *
                     avgHitMaps[i]->GetYaxis()->GetBinWidth(1);
    avgHitMaps[i]->Scale(1.f / (binArea * nEvents));
    if (qed) {
      avgHitMaps[i]->Scale(qedFactor);
    }
    avgHitMapsProj[i] = avgHitMaps[i]->ProjectionX("_px", 0, -1, "e");
    avgHitMapsProj[i]->Scale(1. / avgHitMaps[i]->GetNbinsY());
    avgHitMapsProj[i]->SetMarkerStyle(21);
    avgHitMapsProj[i]->SetTitle(Form("L%d", i));
  }

  std::vector<TH1D> hMaxes;
  hMaxes.reserve(nLayers);
  for (int i = 0; i < nLayers; ++i) {
    hMaxes.emplace_back(Form("hMaxes_%d", i), Form("RMS in Layer %d", i), 600, 0, 150);
  }

  std::array<Data, 3> ret;
  // Report maximum
  printf("----\n");
  printf("Max local hit density:\n");
  for (int i = 0; i < nLayers; ++i) {
    printf("L%i =  %f +- %f\n", i, avgHitMapsProj[i]->GetMaximum(), avgHitMapsProj[i]->GetBinError(avgHitMapsProj[i]->GetMaximumBin()));

    if (!qed) {
      for (const auto& max : eventMaxes[i]) {
        hMaxes[i].Fill(max);
      }

      printf("Per Event Mean=%f with RMS=%f\n", hMaxes[i].GetMean(),
             hMaxes[i].GetRMS());
    }

    Data r{};
    r.max = avgHitMapsProj[i]->GetMaximum();
    r.maxE = avgHitMapsProj[i]->GetBinError(avgHitMapsProj[i]->GetMaximumBin());
    if (!qed) {
      r.mean = hMaxes[i].GetMean();
      r.meanRMS = hMaxes[i].GetRMS();
    }
    ret[i] = r;
  }

  std::unique_ptr<TFile> oFile(TFile::Open("checkHits.root", "RECREATE"));
  checkFile(oFile);
  auto canvXY = new TCanvas("canvHits", "", 1600, 800);
  canvXY->Divide(2, 1);
  canvXY->cd(1);
  hitsXY->Draw("colz");
  canvXY->cd(2);
  hitsZY->Draw("colz");
  canvXY->Write();
  for (int i = 0; i < nLayers; ++i) {
    auto canvDens =
      new TCanvas(Form("c%s", avgHitMaps[i]->GetName()), "", 1600, 800);
    canvDens->Divide(2, 1);
    canvDens->cd(1);
    avgHitMaps[i]->Draw("colz");
    canvDens->cd(2);
    avgHitMapsProj[i]->GetXaxis()->SetTitle("z (cm)");
    avgHitMapsProj[i]->GetYaxis()->SetTitle("hit density (cm^{-2})");
    avgHitMapsProj[i]->Draw("AP");
    canvDens->Write();
  }

  auto canvDensLayers = new TCanvas("canvDensLayers", "", 1000, 800);
  canvDensLayers->cd();
  for (int i = 0; i < nLayers; ++i) {
    auto g = new TGraphErrors(avgHitMapsProj[i]);
    densMG->Add(g);
    g->Write();
  }
  densMG->GetXaxis()->SetTitle("z (cm)");
  densMG->GetYaxis()->SetTitle("hit density (cm^{-2})");
  densMG->Draw("AP pmc plc");
  canvDensLayers->BuildLegend(0.75, 0.75, 0.9, 0.9);
  canvDensLayers->Write();
  canvDensLayers->SaveAs("cHitsDens.pdf");

  auto canvKineN = new TCanvas("canvKineN", "", 2000, 800);
  canvKineN->Divide(2, 1);
  auto pKineN = canvKineN->cd(1);
  pKineN->SetLeftMargin(0.12);
  kineN->Draw("hist");
  pKineN = canvKineN->cd(2);
  pKineN->SetLeftMargin(0.12);
  kineN05->Draw("hist");
  canvKineN->Write();

  for (auto& h : eventHitMaps) {
    for (auto& pp : h) {
      pp->Write();
    }
  }
  for (auto& h : hMaxes) {
    h.Write();
  }
  return ret;
}

void checkFile(const std::unique_ptr<TFile>& file)
{
  if (!file || file->IsZombie()) {
    printf("Could not open %s!\n", file->GetName());
    std::exit(1);
  }
}

TH2D* makeMap(int layer, std::string n)
{
  const double half_length = ssAlpide.mLength / 2.f;
  auto map =
    new TH2D(Form("h_densityZL%d%s", layer, n.data()), "Local Hit Density",
             static_cast<int>(ssAlpide.mLength), -half_length, half_length,
             static_cast<int>(ssAlpide.mRadii[layer] * o2::constants::math::PI /
                              (ssAlpide.mPitchRow * 100)),
             0.f, ssAlpide.mRadii[layer] * o2::constants::math::PI);
  map->GetXaxis()->SetTitle("z (cm)");
  map->GetYaxis()->SetTitle("r#varphi");
  return map;
}

void Plot()
{
  std::vector<std::string> centralities{
    "0.00", "1.57", "2.22", "2.71", "3.13", "3.50", "4.94",
    "6.05", "6.98", "7.81", "8.55", "9.23", "9.88", "10.47",
    "11.04", "11.58", "12.09", "12.58", "13.05", "13.52", "13.97",
    "14.43", "14.96", "15.67", "20.00"};
  std::vector<std::array<Data, 3>> data;
  for (auto it = centralities.cbegin(); it != centralities.cend() - 1; ++it) {
    auto path = "./" + *it + "_" + *(it + 1) + "/";
    gSystem->cd(path.c_str());
    gSystem->Exec("pwd");
    data.push_back(CheckHits());
    gSystem->cd("..");
  }
  for (const auto& elem : data) {
    std::cout << "+++++++++++++++++++++++++++++++++++++++++\n";
    for (int i = 0; i < 3; ++i) {
      std::cout << "===\nLayer " << i << "\n";
      std::cout << elem[i].max << " +/- " << elem[i].maxE << "\n";
      std::cout << elem[i].mean << " +/- " << elem[i].meanRMS << "\n";
    }
    std::cout << "-----------------------------------------" << std::endl;
  }
  const Int_t n = 24;
  char const* range[n] = {
    "0-1", "1-2", "2-3", "3-4", "4-5", "5-10", "10-15", "15-20",
    "20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60",
    "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95-100"};
  int i;
  double x[n];
  double y0[n], y1[n], y2[n];
  double vy0[n], vy1[n], vy2[n];
  double y0M[n], y1M[n], y2M[n];
  double vy0M[n], vy1M[n], vy2M[n];
  for (i = 0; i < n; ++i) {
    x[i] = i;
    y0[i] = data[i][0].max;
    y1[i] = data[i][1].max;
    y2[i] = data[i][2].max;
    vy0[i] = data[i][0].maxE;
    vy1[i] = data[i][1].maxE;
    vy2[i] = data[i][2].maxE;
    y0M[i] = data[i][0].mean;
    y1M[i] = data[i][1].mean;
    y2M[i] = data[i][2].mean;
    vy0M[i] = data[i][0].meanRMS;
    vy1M[i] = data[i][1].meanRMS;
    vy2M[i] = data[i][2].meanRMS;
  }

  auto c1 = new TCanvas("c1", "A Simple Graph Example", 200, 10, 700, 500);
  auto h = new TH1F("h", "", n, x[0] - 0.5, x[n - 1] + 0.5);
  h->SetTitle("Mean hit density per centrality class");
  h->GetYaxis()->SetTitleOffset(1.);
  h->GetXaxis()->SetTitleOffset(1.);
  h->GetYaxis()->SetTitle("hit density (cm^{-2})");
  h->GetXaxis()->SetTitle("centrality (%)");
  h->GetXaxis()->SetNdivisions(-10);
  for (i = 1; i <= n; i++)
    h->GetXaxis()->SetBinLabel(i, range[i - 1]);
  h->SetMaximum(70);
  h->SetMinimum(0);
  h->SetStats(0);
  h->Draw("");
  auto gr0 = new TGraphErrors(n, x, y0, nullptr, vy0);
  gr0->SetMarkerStyle(4);
  gr0->SetMarkerSize(0.5);
  gr0->SetTitle("L0");
  auto gr1 = new TGraphErrors(n, x, y1, nullptr, vy1);
  gr1->SetMarkerStyle(4);
  gr1->SetMarkerSize(0.5);
  gr1->SetTitle("L1");
  auto gr2 = new TGraphErrors(n, x, y2, nullptr, vy2);
  gr2->SetMarkerStyle(4);
  gr2->SetMarkerSize(0.5);
  gr2->SetTitle("L2");
  auto mg = new TMultiGraph("mg", "");
  mg->Add(gr0);
  mg->Add(gr1);
  mg->Add(gr2);
  mg->Draw("P pmc plc");
  auto leg = new TLegend(0.75, 0.75, 0.9, 0.9);
  leg->AddEntry(gr0);
  leg->AddEntry(gr1);
  leg->AddEntry(gr2);
  leg->Draw();
  c1->SaveAs("its3HitsCentralityMean.pdf");

  c1 = new TCanvas("c1", "A Simple Graph Example", 200, 10, 700, 500);
  auto hh = new TH1F("hh", "", n, x[0] - 0.5, x[n - 1] + 0.5);
  hh->SetTitle("Maximum hit density per centrality class per event with RMS");
  hh->GetYaxis()->SetTitleOffset(1.);
  hh->GetXaxis()->SetTitleOffset(1.);
  hh->GetYaxis()->SetTitle("hit density (cm^{-2})");
  hh->GetXaxis()->SetTitle("centrality (%)");
  hh->GetXaxis()->SetNdivisions(-10);
  for (i = 1; i <= n; i++)
    hh->GetXaxis()->SetBinLabel(i, range[i - 1]);
  hh->SetMaximum(130);
  hh->SetMinimum(0);
  hh->SetStats(false);
  hh->Draw("");
  auto gr0M = new TGraphErrors(n, x, y0M, nullptr, vy0M);
  gr0M->SetMarkerStyle(4);
  gr0M->SetMarkerSize(0.5);
  gr0M->SetTitle("L0");
  auto gr1M = new TGraphErrors(n, x, y1M, nullptr, vy1M);
  gr1M->SetMarkerStyle(4);
  gr1M->SetMarkerSize(0.5);
  gr1M->SetTitle("L1");
  auto gr2M = new TGraphErrors(n, x, y2M, nullptr, vy2M);
  gr2M->SetMarkerStyle(4);
  gr2M->SetMarkerSize(0.5);
  gr2M->SetTitle("L2");
  auto mgM = new TMultiGraph("mg", "");
  mgM->Add(gr0M);
  mgM->Add(gr1M);
  mgM->Add(gr2M);
  mgM->Draw("P pmc plc");
  auto legM = new TLegend(0.75, 0.75, 0.9, 0.9);
  legM->AddEntry(gr0M);
  legM->AddEntry(gr1M);
  legM->AddEntry(gr2M);
  legM->Draw();
  c1->SaveAs("its3HitsCentralityMaximum.pdf");
}
