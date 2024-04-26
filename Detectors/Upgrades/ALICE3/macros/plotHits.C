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
//
// Author: M. Concas

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "ITSMFTSimulation/Hit.h"

#include <TCanvas.h>
#include <TFile.h>
#include <TTree.h>
#include <TVector3.h>
#include <TVector2.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TNtuple.h>

#include <vector>
#endif

void plotHits(const size_t nevents = 100)
{
  TNtuple* tuple = new TNtuple("xyz", "xyz", "x:y:z:color");
  TH2* ep = new TH2F("etaph", "hist_etaph;#varphi;#eta", 150, 0., TMath::TwoPi(), 150, -5, 5);
  TH2* zp = new TH2F("zph", "hist_zph;;#varphi;z(cm) ", 150, 0., TMath::TwoPi(), 300, -200, 200);
  TH2* zr = new TH2F("zr", "hist_zr;z(cm);r(cm) ", 500, -400, 400, 500, 0, 110);
  TH3F* xyz_trk = new TH3F("xyz_trk", "hist_xyz;x(cm);y(cm);z(cm)", 300, -100, 100, 300, -100, 100, 300, -400, 400);
  TH3F* xyz_ft3 = new TH3F("xyz_ft3", "hist_xyz;x(cm);y(cm);z(cm)", 300, -100, 100, 300, -100, 100, 300, -400, 400);
  TH3F* xyz_tof = new TH3F("xyz_tf3", "hist_xyz;x(cm);y(cm);z(cm)", 300, -100, 100, 300, -100, 100, 300, -400, 400);

  std::vector<TFile*> hitFiles;
  hitFiles.push_back(TFile::Open("o2sim_HitsTF3.root"));
  hitFiles.push_back(TFile::Open("o2sim_HitsFT3.root"));
  hitFiles.push_back(TFile::Open("o2sim_HitsTRK.root"));

  TTree* trkTree = hitFiles[2] ? (TTree*)hitFiles[2]->Get("o2sim") : nullptr;
  TTree* ft3Tree = hitFiles[1] ? (TTree*)hitFiles[1]->Get("o2sim") : nullptr;
  TTree* tf3Tree = hitFiles[0] ? (TTree*)hitFiles[0]->Get("o2sim") : nullptr;

  // TRK
  std::vector<o2::itsmft::Hit>* trkHit = nullptr;
  trkTree->SetBranchAddress("TRKHit", &trkHit);

  // FT3
  std::vector<o2::itsmft::Hit>* ft3Hit = nullptr;
  ft3Tree->SetBranchAddress("FT3Hit", &ft3Hit);

  // TF3
  std::vector<o2::itsmft::Hit>* tf3Hit = nullptr;
  tf3Tree->SetBranchAddress("TF3Hit", &tf3Hit);

  for (size_t iev = 0; iev < std::min(nevents, (size_t)trkTree->GetEntries()); iev++) {
    trkTree->GetEntry(iev);
    for (const auto& h : *trkHit) {
      TVector3 posvec(h.GetX(), h.GetY(), h.GetZ());
      ep->Fill(TVector2::Phi_0_2pi(posvec.Phi()), posvec.Eta());
      zp->Fill(TVector2::Phi_0_2pi(posvec.Phi()), posvec.Z());
      zr->Fill(posvec.Z(), TMath::Hypot(posvec.X(), posvec.Y()));
      xyz_trk->Fill(posvec.X(), posvec.Y(), posvec.Z());
      tuple->Fill(posvec.X(), posvec.Y(), posvec.Z(), 40);
    }
    ft3Tree->GetEntry(iev);
    for (const auto& h : *ft3Hit) {
      TVector3 posvec(h.GetX(), h.GetY(), h.GetZ());
      ep->Fill(TVector2::Phi_0_2pi(posvec.Phi()), posvec.Eta());
      zp->Fill(TVector2::Phi_0_2pi(posvec.Phi()), posvec.Z());
      zr->Fill(posvec.Z(), TMath::Hypot(posvec.X(), posvec.Y()));
      xyz_ft3->Fill(posvec.X(), posvec.Y(), posvec.Z());
      tuple->Fill(posvec.X(), posvec.Y(), posvec.Z(), 30);
    }
    tf3Tree->GetEntry(iev);
    for (const auto& h : *tf3Hit) {
      TVector3 posvec(h.GetX(), h.GetY(), h.GetZ());
      ep->Fill(TVector2::Phi_0_2pi(posvec.Phi()), posvec.Eta());
      zp->Fill(TVector2::Phi_0_2pi(posvec.Phi()), posvec.Z());
      zr->Fill(posvec.Z(), TMath::Hypot(posvec.X(), posvec.Y()));
      xyz_tof->Fill(posvec.X(), posvec.Y(), posvec.Z());
      tuple->Fill(posvec.X(), posvec.Y(), posvec.Z(), 20);
    }
  }

  auto* EPcanvas = new TCanvas("EtaPhi", "EP", 1000, 800);
  EPcanvas->cd();
  ep->Draw();
  EPcanvas->SaveAs("EtaPhi.png");
  auto* ZPcanvas = new TCanvas("ZPhi", "ZP", 1000, 800);
  ZPcanvas->cd();
  zp->Draw();
  ZPcanvas->SaveAs("ZPhi.png");
  auto* RZcanvas = new TCanvas("RZ", "RZ", 1000, 800);
  RZcanvas->cd();
  zr->Draw();
  RZcanvas->SaveAs("RZ.png");
  auto* XYZcanvas = new TCanvas("XYZ", "XYZ", 1000, 800);
  XYZcanvas->cd();
  xyz_trk->Draw("p");
  xyz_ft3->Draw("same");
  xyz_tof->Draw("same");
  XYZcanvas->SaveAs("XYZ.png");
  auto* XYZ3Dcanvas = new TCanvas("XYZ3D", "XYZ3D", 1000, 800);
  XYZ3Dcanvas->cd();
  tuple->Draw("x:y:z:color");
  XYZ3Dcanvas->SaveAs("XYZ3D.png");
}