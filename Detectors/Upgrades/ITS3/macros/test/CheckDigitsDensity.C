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

/// \file CheckDigitsDensity.C
/// \brief  analyze ITS3 digit density
/// \author felix.schlepper@cern.ch

#include <algorithm>
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TH2F.h>
#include <TLegend.h>
#include <TMath.h>
#include <TMultiGraph.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TTree.h>
#include <TSystem.h>

#include <array>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

#define ENABLE_UPGRADES
#include "CommonConstants/MathConstants.h"
#include "DataFormatsITSMFT/Digit.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#endif

/*
 * How to simulate:
 * $o2-sim-digitizer-workflow -b
 */

constexpr double interaction_rate = 50e3;                                // Hz
constexpr double integration_time = 10e-6;                               // s
constexpr double qedXSection = 34346.9;                                  // barn
constexpr double hadXSection = 8.f;                                      // barn
constexpr double qedRate = qedXSection / hadXSection * interaction_rate; // Hz
constexpr double qedFactor = qedRate * integration_time;                 // a.u.
using o2::itsmft::Digit;
using densMap = std::pair<TH2F*, TGraphErrors*>;
using densMaps = std::vector<densMap>;

void checkFile(const std::unique_ptr<TFile>& file);
densMaps makeDensMaps(int nLayers, std::string n = "");
struct Maximum {
  double y;
  double vy;
  Maximum(double y, double vy) : y{y}, vy{vy} {}
};

std::vector<Maximum> densityProjection(densMaps& maps, int nEvents, bool qed);
std::vector<TCanvas*> makeCanvasProjection(const densMaps& maps);

std::vector<Maximum> CheckDigitsDensity(int nLayers = 3, int nEvents = 100, float pitchCol = 22.e-4f,
                                        float pitchRow = 22.e-4f,
                                        std::string digitFileName = "it3digits.root",
                                        bool qed = false, bool batch = true)
{
  gROOT->SetBatch(batch);

  // Vars
  const int nHemis = nLayers * 2;
  o2::its3::SegmentationSuperAlpide temp;
  std::vector<o2::its3::SegmentationSuperAlpide> seg;
  std::vector<double> totArea;
  for (auto i = 0; i < nLayers; ++i) {
    seg.emplace_back(i, pitchCol, pitchRow, 50.e-4, 26.f, temp.mRadii);
    totArea.emplace_back(seg[i].mRadii[i] * seg[i].mLength * TMath::Pi());
  }

  // Digits
  std::unique_ptr<TFile> digitFile(TFile::Open(digitFileName.data()));
  checkFile(digitFile);
  auto digitTree = digitFile->Get<TTree>("o2sim");
  std::vector<Digit> digitArray;
  std::vector<Digit>* digitArrayPtr{&digitArray};
  digitTree->SetBranchAddress("IT3Digit", &digitArrayPtr);
  std::vector<TH2I> hist;
  auto densityMaps = makeDensMaps(nLayers);
  auto densMG = new TMultiGraph("g_densMG", "Digit Density along z");
  for (auto i = 0; i < nLayers; ++i) {
    hist.emplace_back(Form("h_L%d", i), Form("Digits L%i", i), seg[i].mNCols, 0,
                      seg[i].mNCols, seg[i].mNRows, 0, seg[i].mNRows);
  }

  // Digits Loop
  for (int iEntry = 0; digitTree->LoadTree(iEntry) >= 0; ++iEntry) {
    digitTree->GetEntry(iEntry);

    // Digit
    for (const auto& digit : digitArray) {
      auto id = digit.getChipIndex();
      if (id >= nHemis) { // outside of Inner Barrel
        continue;
      }
      auto col = digit.getColumn();
      auto row = digit.getRow();
      auto z = static_cast<double>(seg[0].mLength) /
                 static_cast<double>(seg[0].mNCols) * col -
               seg[0].mLength / 2.f;
      hist[id / 2].Fill(col, row);
      if (id == 0 || id == 1) {
        auto r = static_cast<double>(TMath::Pi()) * seg[0].mRadii[0] /
                 static_cast<double>(seg[0].mNRows) * row;
        densityMaps[0].first->Fill(z, r);
      } else if (id == 2 || id == 3) {
        auto r = static_cast<double>(TMath::Pi()) * seg[1].mRadii[1] /
                 static_cast<double>(seg[1].mNRows) * row;
        densityMaps[1].first->Fill(z, r);
      } else if (id == 4 || id == 5) {
        auto r = static_cast<double>(TMath::Pi()) * seg[2].mRadii[2] /
                 static_cast<double>(seg[2].mNRows) * row;
        densityMaps[2].first->Fill(z, r);
      }
    }
  }

  auto maxes = densityProjection(densityMaps, nEvents, qed);

  printf("----\n");
  printf("NEvents %d\n", nEvents);
  for (auto i = 0; i < nLayers; ++i) {
    printf("Layer %d\n", i);
    auto nDigits = hist[i].Integral();
    printf("nDigits %f\n", nDigits);
    printf("Number of digits per Area (cm^-2) per Event: %.2f\n",
           nDigits / static_cast<double>(totArea[i] * nEvents * 2));
    double nPixels =
      seg[i].mNCols * seg[i].mNRows * 2; // both hemispheres combined
    printf("Total number of Pixel %f\n", nPixels);
    printf("Occupancy: %.6f\n", nDigits / (nPixels * nEvents));
    printf("L%i =  %f +- %f\n", i, maxes[i].y, maxes[i].vy);
  }

  std::unique_ptr<TFile> oFile(TFile::Open("checkDigitsDensity.root", "RECREATE"));
  checkFile(oFile);
  for (auto& h : hist) {
    h.Scale(1.f / (22.e-4 * 22.e-4 * nEvents * 2));
    h.Write();
  }

  auto canvDens = makeCanvasProjection(densityMaps);
  for (auto& canvDen : canvDens) {
    canvDen->Write();
  }

  auto canvDensLayers = new TCanvas("canvDensLayers", "", 1000, 800);
  canvDensLayers->cd();
  for (const auto& [_, g] : densityMaps) {
    densMG->Add(g);
    g->Write();
  }
  densMG->GetXaxis()->SetTitle("z (cm)");
  densMG->GetYaxis()->SetTitle("digit density (cm^{-2})");
  densMG->Draw("AP pmc plc");
  canvDensLayers->BuildLegend(0.75, 0.75, 0.9, 0.9);
  canvDensLayers->Write();
  canvDensLayers->SaveAs("cDigits.pdf");

  return maxes;
}

void checkFile(const std::unique_ptr<TFile>& file)
{
  if (!file || file->IsZombie()) {
    printf("Could not open %s!\n", file->GetName());
    std::exit(1);
  }
}

// Make density plots for nLayers
densMaps makeDensMaps(int nLayers, std::string n)
{
  const o2::its3::SegmentationSuperAlpide ssAlpide;
  const double half_length = ssAlpide.mLength / 2.f;
  densMaps map;
  for (int i = 0; i < nLayers; ++i) {
    auto densityZ =
      new TH2F(Form("h_densityZL%d%s", i, n.data()), "Local Digit Density",
               static_cast<int>(ssAlpide.mLength), -half_length, half_length,
               static_cast<int>(ssAlpide.mRadii[i] * o2::constants::math::PI /
                                (ssAlpide.mPitchRow * 100)),
               0.f, ssAlpide.mRadii[i] * o2::constants::math::PI);
    densityZ->GetXaxis()->SetTitle("z (cm)");
    densityZ->GetYaxis()->SetTitle("#varphi");
    auto densityZErr = new TGraphErrors(densityZ->GetNbinsX());
    densityZErr->SetMarkerStyle(21);
    densityZErr->SetTitle(Form("L%d", i));
    densityZErr->GetXaxis()->SetTitle("z (cm)");
    densityZErr->GetYaxis()->SetTitle("hit density (cm^{-2})");
    map.emplace_back(std::move(densityZ), std::move(densityZErr));
  }
  return map;
}
std::vector<Maximum> densityProjection(densMaps& maps, int nEvents, bool qed)
{
  std::vector<Maximum> maxes;
  for (auto& [densityZ, densityProZErr] : maps) {
    double binArea = densityZ->GetXaxis()->GetBinWidth(1) *
                     densityZ->GetYaxis()->GetBinWidth(1);
    double maxHitDensity = 0.f;
    double maxHitDensityErr = 0.f;
    // Calculate Hit density for each pixel
    // For PbPb just per event
    densityZ->Scale(
      1.f /
      (2 * binArea *
       nEvents)); // to increase statistics we looked at both hemispheres
    if (qed) {
      densityZ->Scale(qedFactor);
    }
    // Fill hit density projection
    for (int x = 1; x <= densityZ->GetNbinsX(); ++x) {
      std::vector<double> v;
      for (int y = 1; y <= densityZ->GetNbinsY(); ++y) {
        if (densityZ->GetBinContent(x, y) > 0) {
          v.push_back(densityZ->GetBinContent(x, y));
        }
      }
      double sum = std::accumulate(v.begin(), v.end(), 0.0);
      double mean = sum / v.size();
      std::vector<double> diff(v.size());
      std::transform(v.begin(), v.end(), diff.begin(),
                     [mean](double x) { return x - mean; });
      double sq_sum =
        std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
      double stdev = std::sqrt(sq_sum / v.size());
      // Set point
      densityProZErr->SetPoint(x - 1, densityZ->GetXaxis()->GetBinCenter(x),
                               mean);
      densityProZErr->SetPointError(
        x - 1, densityZ->GetXaxis()->GetBinWidth(x) / 2.f, stdev);
      // update max
      if (mean > maxHitDensity) {
        maxHitDensity = mean;
        maxHitDensityErr = stdev;
      }
    }
    maxes.emplace_back(maxHitDensity, maxHitDensityErr);
  }
  return maxes;
}

std::vector<TCanvas*> makeCanvasProjection(const densMaps& maps)
{
  std::vector<TCanvas*> canvases;
  for (const auto& [densityZ, densityProZErr] : maps) {
    auto canvDens =
      new TCanvas(Form("c%s", densityZ->GetName()), "", 1600, 800);
    canvDens->Divide(2, 1);
    canvDens->cd(1);
    densityZ->Draw("colz");
    canvDens->cd(2);
    densityProZErr->Draw("AP");
    canvases.push_back(canvDens);
  }
  return canvases;
}

void Plot()
{
  int i = 0;
  int j = 0;
  std::vector<std::string> centralities{
    "0.00", "1.57", "2.22", "2.71", "3.13", "3.50", "4.94",
    "6.05", "6.98", "7.81", "8.55", "9.23", "9.88", "10.47",
    "11.04", "11.58", "12.09", "12.58", "13.05", "13.52", "13.97",
    "14.43", "14.96", "15.67", "20.00"};
  std::vector<std::vector<Maximum>> data;
  for (auto it = centralities.cbegin(); it != centralities.cend() - 1; ++it) {
    auto path = "./" + *it + "_" + *(it + 1) + "/";
    gSystem->cd(path.c_str());
    gSystem->Exec("pwd");
    data.push_back(CheckDigitsDensity());
    gSystem->cd("..");
  }
  for (const auto& elem : data) {
    std::cout << "+++++++++++++++++++++++++++++++++++++++++\n";
    std::cout << ++j << std::endl;
    for (int i = 0; i < 3; ++i) {
      std::cout << "===\nLayer " << i << "\n";
      std::cout << elem[i].y << " +/- " << elem[i].vy << "\n";
    }
    std::cout << "-----------------------------------------" << std::endl;
  }
  auto c1 = new TCanvas("c1", "A Simple Graph Example", 200, 10, 700, 500);
  const Int_t n = 24;
  char const* range[n] = {
    "0-1", "1-2", "2-3", "3-4", "4-5", "5-10", "10-15", "15-20",
    "20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60",
    "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95-100"};
  double x[n];
  double y0[n], y1[n], y2[n];
  double vy0[n], vy1[n], vy2[n];
  for (i = 0; i < n; ++i) {
    x[i] = i;
    y0[i] = data[i][0].y;
    y1[i] = data[i][1].y;
    y2[i] = data[i][2].y;
    vy0[i] = data[i][0].vy;
    vy1[i] = data[i][1].vy;
    vy2[i] = data[i][2].vy;
  }

  auto h = new TH1F("h", "", n, x[0] - 0.5, x[n - 1] + 0.5);
  h->SetTitle("Mean digit density per centrality class");
  h->GetYaxis()->SetTitleOffset(1.);
  h->GetXaxis()->SetTitleOffset(1.);
  h->GetYaxis()->SetTitle("digit density (cm^{-2})");
  h->GetXaxis()->SetTitle("centrality (%)");
  h->GetXaxis()->SetNdivisions(-10);
  for (i = 1; i <= n; i++)
    h->GetXaxis()->SetBinLabel(i, range[i - 1]);
  h->SetMaximum(600);
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

  c1->SaveAs("its3DigitDensityCentrality.pdf");
}
