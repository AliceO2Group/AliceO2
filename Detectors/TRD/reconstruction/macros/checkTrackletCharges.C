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

/// \file checkTrackletCharges.C
/// \brief Simple macro to display the tracklet charge information.
//
//  outputs a png
//

#if !defined(__CLING__) || defined(__ROOTCLING__)
// ROOT header
#include <TROOT.h>
#include <TChain.h>
#include <TH2.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TProfile.h>
#include <TFile.h>
#include <TTree.h>
#include <TAxis.h>
#include <TLine.h>
// O2 header
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Constants.h"
#endif

using namespace o2::trd;

std::vector<TH1F*> createTrdChargeHists()
{
  std::vector<TH1F*> hCharges;
  hCharges.push_back(new TH1F("Charge0", "Charge window 0;charge;counts", 128, -0.5, 127.5));
  hCharges.push_back(new TH1F("Charge1", "Charge window 1;charge;counts", 128, -0.5, 127.5));
  hCharges.push_back(new TH1F("Charge2", "Charge window 2;charge;counts", 64, -0.5, 63.5));
  return hCharges;
}

void checkTrackletCharges(int sector = -1)
{
  TChain chain("o2sim");
  chain.AddFile("trdtracklets.root");
  std::vector<TriggerRecord> trigIn, *trigInPtr{&trigIn};
  std::vector<Tracklet64> tracklets, *trackletsInPtr{&tracklets};
  chain.SetBranchAddress("TrackTrg", &trigInPtr);
  chain.SetBranchAddress("Tracklet", &trackletsInPtr);

  //auto fOut = new TFile("trackletsOutput.root", "recreate");
  auto hDet = new TH1F("det", "Detector number for tracklet;detector;counts", 540, -0.5, 539.5);
  auto hTrigger = new TH1F("trigger", "Number of TRD triggers per TF;# trigger;counts", 200, 0, 200);
  auto hPosition = new TH1F("position", "Tracklet position uncalibrated;position;counts", 120, -60, 60);
  auto hSlope = new TH1F("slope", "Tracklet slope uncalibrated;slope;counts", 200, -2, 2);

  auto hNTracklets = new TH1F("nTracklets", "Number of tracklets for triggers with digits;nTracklets;counts", 100, 0, 1000);

  auto hCharges = createTrdChargeHists();

  int countEntries = 0;
  int countTrigger = 0;
  int nTriggerWithDigits = 0;

  for (int iEntry = 0; iEntry < chain.GetEntries(); ++iEntry) {
    chain.GetEntry(iEntry); // for each TimeFrame there is one tree entry
    ++countEntries;
    countTrigger += trigIn.size();
    hTrigger->Fill(trigIn.size());
    for (const auto& tracklet : tracklets) {
      int det = tracklet.getHCID() / 2;
      int side = tracklet.getHCID() % 2; // 0: A-side, 1: B-side
      hDet->Fill(det);
      int layer = det % 6;
      int stack = (det % 30) / 6;
      int sec = det / 30;
      hSlope->Fill(tracklet.getUncalibratedDy());
      hPosition->Fill(tracklet.getUncalibratedY());
      if (sector > -1) {

        if (sector == sec) {
          if (tracklet.getQ0() > 0)
            hCharges[0]->Fill(tracklet.getQ0());
          if (tracklet.getQ1() > 0)
            hCharges[1]->Fill(tracklet.getQ1());
          if (tracklet.getQ2() > 0)
            hCharges[2]->Fill(tracklet.getQ2());
        }
      } else {
        if (tracklet.getQ0() > 0)
          hCharges[0]->Fill(tracklet.getQ0());
        if (tracklet.getQ1() > 0)
          hCharges[1]->Fill(tracklet.getQ1());
        if (tracklet.getQ2() > 0)
          hCharges[2]->Fill(tracklet.getQ2());
      }
    }
  }

  //printf("Found in total %i collisions with digits\n", nTriggerWithDigits);
  auto c = new TCanvas("c", "c", 1400, 1000);
  auto line = new TLine();
  c->Divide(3, 3);
  c->cd(1);
  hDet->Draw();
  c->cd(2);
  hTrigger->Draw();
  c->cd(3);
  hNTracklets->Draw();
  auto pada = c->cd(4);
  pada->SetLogy();
  hPosition->Draw();
  auto padb = c->cd(5);
  padb->SetLogy();
  hSlope->Draw();

  for (int charge = 0; charge < 3; ++charge) {
    auto pad = c->cd(charge + 7);
    pad->SetRightMargin(0.15);
    hCharges[charge]->Draw();
    //pad->SetLogz();
  }
  c->Update();
  c->SaveAs(Form("tracklet_charges_sector%d.png", sector));

  printf("Got in total %i trigger from %i TFs\n", countTrigger, countEntries);
}
