// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CheckHits.C
/// \brief Simple macro to check TRD hits

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TCanvas.h>

#include "FairLogger.h"
#include "TRDSimulation/Detector.h"
#include "TRDBase/TRDGeometry.h"
#include "TRDBase/Calibrations.h"
#endif

void CheckHits(const int detector = 50, // 354, 14, 242, 50
               std::string digifile = "trddigits.root",
               std::string hitfile = "o2sim.root",
               std::string inputGeom = "O2geometry.root",
               std::string paramfile = "o2sim_par.root")
{
  // o2::trd::Calibrations calib;
  // calib.setCCDBForSimulation(297595);

  TFile* fin = TFile::Open(hitfile.data());
  TTree* hitTree = (TTree*)fin->Get("o2sim");
  std::vector<o2::trd::HitType>* hits = nullptr;
  hitTree->SetBranchAddress("TRDHit", &hits);
  int nev = hitTree->GetEntries();

  TH1F* hlocC = new TH1F("hlocC", ";locC (cm);Counts", 100, -60, 60);
  TH1F* hlocR = new TH1F("hlocR", ";locR (cm);Counts", 100, -80, 80);
  TH1F* hlocT = new TH1F("hlocT", ";locT (cm);Counts", 100, -3.5, 0.5);
  TH1F* hnEl = new TH1F("hnEl", ";Number of Electrons;Counts", 100, 0, 5000);
  TH1F* hnElPhoton = new TH1F("hnElPhoton", ";Number of Electrons;Counts", 100, 0, 1000);

  TH2F* h2locClocT = new TH2F("h2locClocT", ";locC (cm);locT(cm)", 100, -60, 60, 100, -3.5, 0.5);
  TH2F* h2locClocTnEl = new TH2F("h2locClocTnEl", "nEl;locC (cm);locT(cm)", 100, -60, 60, 100, -3.5, 0.5);

  LOG(INFO) << nev << " entries found";
  for (int iev = 0; iev < nev; ++iev) {
    hitTree->GetEvent(iev);
    for (const auto& hit : *hits) {
      int det = hit.GetDetectorID();
      // if (calib.isChamberNoData(det) || o2::trd::TRDGeometry::getStack(det)!=2 || o2::trd::TRDGeometry::getSector(det) != 4) {
      // if (calib.isChamberNoData(det)) {
      //   continue;
      // }
      if (det != detector) {
        // LOG(INFO) << "REJECTED Detector = " << det <<"\t Stack = " << o2::trd::TRDGeometry::getStack(det) << "\t Sector = " << o2::trd::TRDGeometry::getSector(det);
        continue;
      }
      LOG(INFO) << "ACCEPTED Detector = " << det << "\t Stack = " << o2::trd::TRDGeometry::getStack(det) << "\t Sector = " << o2::trd::TRDGeometry::getSector(det);
      // loop over det, pad, row?
      double locC = hit.getLocalC(); // col direction in amplification or drift volume
      double locR = hit.getLocalR(); // row direction in amplification or drift volume
      double locT = hit.getLocalT(); // time direction in amplification or drift volume
      int nEl = hit.GetHitValue();
      hlocC->Fill(locC);
      hlocR->Fill(locR);
      hlocT->Fill(locT);
      if (nEl > 0) {
        hnEl->Fill(nEl);
      } else {
        hnElPhoton->Fill(-nEl);
      }
      h2locClocT->Fill(locC, locT);
      auto bin = h2locClocTnEl->FindBin(locC, locT);
      h2locClocTnEl->SetBinContent(bin, nEl);
    }
  }
  TCanvas* c = new TCanvas("c", "trd hits analysis", 800, 600);
  c->Divide(2, 2);
  c->cd(1);
  hlocC->Draw();
  c->cd(2);
  hlocR->Draw();
  c->cd(3);
  hlocT->Draw();
  c->cd(4);
  c->cd(4)->SetLogy();
  hnEl->Draw();
  hnElPhoton->SetLineColor(kRed);
  hnElPhoton->Draw("SAME");
  c->SaveAs("testCheckHits_1D.pdf");

  TCanvas* c2 = new TCanvas("c2", "trd hits analysis", 800, 800);
  h2locClocT->Draw("COL");
  c2->SaveAs("testCheckHits_2D.pdf");

  TCanvas* c3 = new TCanvas("c3", "trd hits analysis", 800, 800);
  h2locClocTnEl->Draw("COL");
  c3->SaveAs("testCheckHits_2D_nEl.pdf");
}