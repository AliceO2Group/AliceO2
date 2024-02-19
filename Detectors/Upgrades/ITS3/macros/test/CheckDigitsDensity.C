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
#include <algorithm>

#define ENABLE_UPGRADES
#include "ITS3Base/SpecsV2.h"
#include "CommonConstants/MathConstants.h"
#include "DataFormatsITSMFT/Digit.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSBase/GeometryTGeo.h"
#include "fairlogger/Logger.h"
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
namespace its3 = o2::its3;
using SSAlpide = its3::SegmentationSuperAlpide;

void checkFile(const std::unique_ptr<TFile>& file);

void CheckDigitsDensity(int nEvents = 10000, std::string digitFileName = "it3digits.root", std::string geomFileName = "o2sim_geometry.root", bool qed = false, bool batch = true)
{
  gROOT->SetBatch(batch);
  LOGP(debug, "Checking Digit ITS3 Density");
  // Vars

  // Geometry
  o2::base::GeometryManager::loadGeometry(geomFileName);
  auto* gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  // Digits
  LOGP(debug, "Opening tree in {}", digitFileName);
  std::unique_ptr<TFile> digitFile(TFile::Open(digitFileName.data()));
  checkFile(digitFile);
  auto digitTree = digitFile->Get<TTree>("o2sim");
  std::vector<Digit> digitArray, *digitArrayPtr{&digitArray};
  digitTree->SetBranchAddress("IT3Digit", &digitArrayPtr);
  std::array<TH2F*, 3> hists;
  for (int i{3}; i--;) {
    double rmin = its3::constants::radii[i] - its3::constants::thickness;
    double rmax = its3::constants::radii[i] + its3::constants::thickness;
    hists[i] = new TH2F(Form("h_digits_dens_L%d", i), Form("Digit Density L%d in %d Events; Z_{Glo} [cm]; R_{Glo} [cm]", i, nEvents), 100, -15, 15, 100, rmin, rmax);
  }

  // Digits Loop
  LOGP(debug, "Starting tree loop");
  for (int iEntry = 0; digitTree->LoadTree(iEntry) >= 0; ++iEntry) {
    digitTree->GetEntry(iEntry);

    // Digit
    for (const auto& digit : digitArray) {
      auto id = digit.getChipIndex();
      auto layer = its3::constants::detID::getDetID2Layer(id);
      bool isIB = its3::constants::detID::isDetITS3(id);
      if (!isIB) { // outside of Inner Barrel, e.g. Outer Barrel
        continue;
      }
      auto col = digit.getColumn();
      auto row = digit.getRow();
      // goto curved coordinates
      float x{0.f}, y{0.f}, z{0.f};
      float xFlat{0.f}, yFlat{0.f};
      its3::SuperSegmentations[layer].detectorToLocal(row, col, xFlat, z);
      its3::SuperSegmentations[layer].flatToCurved(xFlat, 0., x, y);
      const o2::math_utils::Point3D<double> locD(x, y, z);
      const auto gloD = gman->getMatrixL2G(id)(locD); // convert to global
      const auto R = std::hypot(gloD.X(), gloD.Y());
      hists[layer]->Fill(gloD.Z(), R);
    }
  }

  std::unique_ptr<TFile> oFile(TFile::Open("checkDigitsDensity.root", "RECREATE"));
  checkFile(oFile);
  for (const auto& h : hists) {
    h->Scale(1. / (SSAlpide::mPitchCol * SSAlpide::mPitchRow * nEvents));
    h->ProjectionX()->Write();
    h->Write();
  }
}

void checkFile(const std::unique_ptr<TFile>& file)
{
  if (!file || file->IsZombie()) {
    printf("Could not open %s!\n", file->GetName());
    std::exit(1);
  }
}

// void Plot()
// {
//   int i = 0;
//   int j = 0;
//   constexpr std::array<std::string_view, 25> centralities{
//     "0.00", "1.57", "2.22", "2.71", "3.13", "3.50", "4.94",
//     "6.05", "6.98", "7.81", "8.55", "9.23", "9.88", "10.47",
//     "11.04", "11.58", "12.09", "12.58", "13.05", "13.52", "13.97",
//     "14.43", "14.96", "15.67", "20.00"};
//   std::vector<std::vector<Maximum>> maxData;
//   for (auto it = centralities.cbegin(); it != centralities.cend() - 1; ++it) {
//     auto path = "./" + *it + "_" + *(it + 1) + "/";
//     gSystem->cd(path.c_str());
//     gSystem->Exec("pwd");
//     data.push_back(CheckDigitsDensity());
//     gSystem->cd("..");
//   }
//   for (const auto& elem : data) {
//     std::cout << "+++++++++++++++++++++++++++++++++++++++++\n";
//     std::cout << ++j << std::endl;
//     for (int i = 0; i < 3; ++i) {
//       std::cout << "===\nLayer " << i << "\n";
//       std::cout << elem[i].y << " +/- " << elem[i].vy << "\n";
//     }
//     std::cout << "-----------------------------------------" << std::endl;
//   }
//   auto c1 = new TCanvas("c1", "A Simple Graph Example", 200, 10, 700, 500);
//   const Int_t n = 24;
//   char const* range[n] = {
//     "0-1", "1-2", "2-3", "3-4", "4-5", "5-10", "10-15", "15-20",
//     "20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55", "55-60",
//     "60-65", "65-70", "70-75", "75-80", "80-85", "85-90", "90-95", "95-100"};
//   double x[n];
//   double y0[n], y1[n], y2[n];
//   double vy0[n], vy1[n], vy2[n];
//   for (i = 0; i < n; ++i) {
//     x[i] = i;
//     y0[i] = data[i][0].y;
//     y1[i] = data[i][1].y;
//     y2[i] = data[i][2].y;
//     vy0[i] = data[i][0].vy;
//     vy1[i] = data[i][1].vy;
//     vy2[i] = data[i][2].vy;
//   }
//
//   auto h = new TH1F("h", "", n, x[0] - 0.5, x[n - 1] + 0.5);
//   h->SetTitle("Mean digit density per centrality class");
//   h->GetYaxis()->SetTitleOffset(1.);
//   h->GetXaxis()->SetTitleOffset(1.);
//   h->GetYaxis()->SetTitle("digit density (cm^{-2})");
//   h->GetXaxis()->SetTitle("centrality (%)");
//   h->GetXaxis()->SetNdivisions(-10);
//   for (i = 1; i <= n; i++)
//     h->GetXaxis()->SetBinLabel(i, range[i - 1]);
//   h->SetMaximum(600);
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
//
//   c1->SaveAs("its3DigitDensityCentrality.pdf");
// }
