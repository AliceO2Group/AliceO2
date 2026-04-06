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
#include <TStyle.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TH2F.h>
#include <TProfile2D.h>
#include <TTree.h>

#include "MathUtils/Utils.h"
#include "ITSMFTSimulation/Hit.h"
#include "ITS3Base/SpecsV2.h"
#endif

void CheckHitResiduals(const std::string& hitOFile = "o2sim_HitsIT3_Orig.root",
                       const std::string& hitRFile = "o2sim_HitsIT3.root")
{
  gStyle->SetOptStat(0);

  TFile fileO(hitOFile.data());
  auto* hitOTree = dynamic_cast<TTree*>(fileO.Get("o2sim"));
  std::vector<o2::itsmft::Hit>* hitOArray = nullptr;
  hitOTree->SetBranchAddress("IT3Hit", &hitOArray);

  TFile fileR(hitRFile.data());
  auto* hitRTree = dynamic_cast<TTree*>(fileR.Get("o2sim"));
  std::vector<o2::itsmft::Hit>* hitRArray = nullptr;
  hitRTree->SetBranchAddress("IT3Hit", &hitRArray);

  struct Hits {
    o2::itsmft::Hit oHit, rHit;
    int oEve{-1}, rEve{-1};
    bool hasBoth() const noexcept { return oEve >= 0 && rEve >= 0 && oEve == rEve && oHit.GetDetectorID() == rHit.GetDetectorID() && rHit.GetTrackID() == oHit.GetTrackID(); }
  };
  std::unordered_map<uint64_t, Hits> hits;

  size_t total{0};
  for (int iEntry{0}; iEntry < hitOTree->GetEntries(); ++iEntry) {
    hitOTree->GetEntry(iEntry);
    total += hitOArray->size();
    for (const auto& h : *hitOArray) {
      if (!o2::its3::constants::detID::isDetITS3(h.GetDetectorID())) {
        continue;
      }
      uint64_t key = (uint64_t(iEntry) << 48) | (uint64_t(h.GetTrackID()) << 24) | uint64_t(h.GetDetectorID());
      auto& hh = hits[key];
      hh.oHit = h;
      hh.oEve = iEntry;
    }
  }
  printf("placed %zu hits saw %zu\n", hits.size(), total);
  for (int iEntry{0}; iEntry < hitRTree->GetEntries(); ++iEntry) {
    hitRTree->GetEntry(iEntry);
    for (const auto& h : *hitRArray) {
      if (!o2::its3::constants::detID::isDetITS3(h.GetDetectorID())) {
        continue;
      }
      uint64_t key = (uint64_t(iEntry) << 48) | (uint64_t(h.GetTrackID()) << 24) | uint64_t(h.GetDetectorID());
      auto& hh = hits[key];
      hh.rHit = h;
      hh.rEve = iEntry;
    }
  }

  // plot the residuals in dRPhi, dZ against ideal phi,z for each layer
  std::array<TProfile2D*, 3> mDRPhi{}, mDZ{};
  for (int i{0}; i < 3; ++i) {
    mDRPhi[i] = new TProfile2D(Form("hDRPhi_%d", i), Form("#Delta_{r#varphi};z (cm); r#varphi (cm)"), 100, -15, 15, 100, 0, 2 * TMath::Pi());
    mDRPhi[i]->SetDirectory(nullptr);
    mDZ[i] = new TProfile2D(Form("hDZ_%d", i), Form("#Delta_{z};z (cm); r#varphi (cm)"), 100, -15, 15, 100, 0, 2 * TMath::Pi());
    mDZ[i]->SetDirectory(nullptr);
  }
  std::array<int, 3> cIB{};
  for (const auto& [_, h] : hits) {
    if (h.hasBoth()) {
      int chip = o2::its3::constants::detID::getSensorID(h.oHit.GetDetectorID()) / 2;
      ++cIB[chip];
      auto gloO = h.oHit.GetPosStart();
      auto gloR = h.rHit.GetPosStart();
      // ideal (original) cluster: phi, z, rphi
      float phiO = std::atan2(gloO.Y(), gloO.X());
      o2::math_utils::bringTo02Pi(phiO);
      const float rO = std::hypot(gloO.X(), gloO.Y());

      // deformed (reconstructed) cluster
      float phiR = std::atan2(gloR.Y(), gloR.X());
      o2::math_utils::bringTo02Pi(phiR);
      const float rR = std::hypot(gloR.X(), gloR.Y());

      // residuals
      const float dRPhi = rO * (phiR - phiO); // or use average r, doesn't matter at this precision
      const float dZ = gloR.Z() - gloO.Z();

      // fill vs (z, phi) of ideal position
      mDRPhi[chip]->Fill(gloO.Z(), phiO, dRPhi * 1e4);
      mDZ[chip]->Fill(gloO.Z(), phiO, dZ * 1e4);
    }
  }
  for (int lay{0}; lay < 3; ++lay) {
    printf("\t%d has %d\n", lay, cIB[lay]);
  }
  auto c1 = new TCanvas;
  c1->Divide(3, 1);
  for (int i{0}; i < 3; ++i) {
    c1->cd(1 + i);
    gPad->SetRightMargin(3);
    mDRPhi[i]->GetZaxis()->SetRangeUser(-200, 200);
    mDRPhi[i]->Draw("colz");
  }
  c1->Draw();
  c1->SaveAs("its3_clus_res_rphi.pdf");
  auto c2 = new TCanvas;
  c2->Divide(3, 1);
  for (int i{0}; i < 3; ++i) {
    c2->cd(1 + i);
    gPad->SetRightMargin(3);
    mDZ[i]->GetZaxis()->SetRangeUser(-200, 200);
    mDZ[i]->Draw("colz");
  }
  c2->Draw();
  c2->SaveAs("its3_clus_res_z.pdf");
}
