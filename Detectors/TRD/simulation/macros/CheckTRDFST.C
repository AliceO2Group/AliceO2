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

/// \file CheckTRDFST.C
/// \brief Simple macro to check TRD digits and tracklets post sim to post reconstruction

// a couple of steps are not included in this:
// It is assumed that the raw-to-tf is run in the directory you run the fst.
// you reconstruct to trddigits and trdtracklets in the fst/raw/timeframe directory
//
// alienv enter O2PDPSuite/latest-o2 Readout/latest-o2
// DISABLE_PROCESSING=1 NEvents=20 NEventsQED=100 SHMSIZE=128000000000 TPCTRACKERSCRATCHMEMORY=40000000000 SPLITTRDDIGI=0 GENERATE_ITSMFT_DICTIONARIES=1 $O2_ROOT/prodtests/full_system_test.sh
// $O2_ROOT/prodtests/full-system-test/convert-raw-to-tf-file.sh
// cd raw/timeframe
// o2-raw-tf-reader-workflow --input-data o2_rawtf_run00000000_tf00000001_???????.tf | o2-trd-datareader --fixsm1617 --enable-root-output | o2-dpl-run --run -b
// Then run this script.
// the convert-raw-to-tf-file.sh must be run on a machine with >200G the rest can be run anywhere.
//

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TCanvas.h>
#include <TLegend.h>

#include <fairlogger/Logger.h>
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"
#endif

using namespace o2::trd;

constexpr int kMINENTRIES = 100;

void CheckTRDFST(std::string fstbasedir = "./",
                 std::string digitfile = "trddigits.root", std::string trackletfile = "trdtracklets.root",
                 std::string recodigitfile = "trddigits.root", std::string recotrackletfile = "trdtracklets.root")
{
  TFile* dfin = TFile::Open(Form("%s%s", fstbasedir.data(), digitfile.data()));
  TTree* digitTree = (TTree*)dfin->Get("o2sim");
  std::vector<Digit>* digits = nullptr;
  digitTree->SetBranchAddress("TRDDigit", &digits);
  int ndigitev = digitTree->GetEntries();

  TFile* tfin = TFile::Open(Form("%s%s", fstbasedir.data(), trackletfile.data()));
  TTree* trackletTree = (TTree*)tfin->Get("o2sim");
  std::vector<Tracklet64>* tracklets = nullptr;
  trackletTree->SetBranchAddress("Tracklet", &tracklets);
  int ntrackletev = trackletTree->GetEntries();

  TFile* dfinreco = TFile::Open(Form("%s/raw/timeframe/%s", fstbasedir.data(), recodigitfile.data()));
  TTree* digitTreereco = (TTree*)dfinreco->Get("o2sim");
  std::vector<Digit>* digitsreco = nullptr;
  digitTreereco->SetBranchAddress("TRDDigit", &digitsreco);
  int ndigitevreco = digitTreereco->GetEntries();

  TFile* tfinreco = TFile::Open(Form("%s/raw/timeframe/%s", fstbasedir.data(), recotrackletfile.data()));
  TTree* trackletTreereco = (TTree*)tfinreco->Get("o2sim");
  std::vector<Tracklet64>* trackletsreco = nullptr;
  trackletTreereco->SetBranchAddress("Tracklet", &trackletsreco);
  int ntrackletevreco = trackletTreereco->GetEntries();

  TH2F* hDigitsPerLayer[6];
  TH2F* hTrackletsPerLayer[6];
  TH2F* hDigitsPerLayer_reco[6];
  TH2F* hTrackletsPerLayer_reco[6];
  TH2F* hDigitsPerLayer_diff[6];
  TH2F* hTrackletsPerLayer_diff[6];
  for (int layer = 0; layer < 6; layer++) {
    hDigitsPerLayer[layer] = new TH2F(Form("Digit_layer%d", layer), ";stack;sector", 5, 0, 5, 35, 0, 35);
    hDigitsPerLayer[layer]->SetTitle(Form("Layer %d", layer));
    hTrackletsPerLayer[layer] = new TH2F(Form("Tracklets_layer%d", layer), ";stack;sector", 5, 0, 5, 35, 0, 35);
    hTrackletsPerLayer[layer]->SetTitle(Form("Layer %d", layer));
    hDigitsPerLayer_reco[layer] = new TH2F(Form("Digit_layer%d_reco", layer), ";stack;sector", 5, 0, 5, 35, 0, 35);
    hDigitsPerLayer_reco[layer]->SetTitle(Form("Layer %d", layer));
    hTrackletsPerLayer_reco[layer] = new TH2F(Form("Tracklets_layer%d_reco", layer), ";stack;sector", 5, 0, 5, 35, 0, 35);
    hTrackletsPerLayer_reco[layer]->SetTitle(Form("Layer %d", layer));
    hDigitsPerLayer_diff[layer] = new TH2F(Form("Digit_layer%d_diff", layer), ";stack;sector", 5, 0, 5, 35, 0, 35);
    hDigitsPerLayer_diff[layer]->SetTitle(Form("Layer %d difference", layer));
    hTrackletsPerLayer_diff[layer] = new TH2F(Form("Tracklets_layer%d_diff", layer), ";stack;sector", 5, 0, 5, 35, 0, 35);
    hTrackletsPerLayer_diff[layer]->SetTitle(Form("Layer %d difference", layer));
  }

  LOG(info) << ndigitev << " digits entries found";
  LOG(info) << ntrackletev << " tracklet entries found";
  for (int iev = 0; iev < ntrackletev; ++iev) {
    digitTree->GetEvent(iev);
    trackletTree->GetEvent(iev);
    for (const auto& digit : *digits) {
      int det = digit.getDetector(); // chamber
      int row = digit.getPadRow();   // pad row
      int col = digit.getPadCol();   // pad column
      int stack = o2::trd::HelperMethods::getStack(det);
      int layer = o2::trd::HelperMethods::getLayer(det);
      int sector = o2::trd::HelperMethods::getSector(det);
      hDigitsPerLayer[layer]->Fill(stack, sector * 2 + digit.getROB() % 2);
    }
    for (const auto& tracklet : *tracklets) {
      int det = tracklet.getHCID() / 2; // chamber
      int hcid = tracklet.getHCID();
      int stack = o2::trd::HelperMethods::getStack(det);
      int layer = o2::trd::HelperMethods::getLayer(det);
      int sector = o2::trd::HelperMethods::getSector(det);
      hTrackletsPerLayer[layer]->Fill(stack, sector * 2 + tracklet.getHCID() % 2);
    }
    LOG(info) << ndigitevreco << " digits entries found";
    LOG(info) << ntrackletevreco << " tracklet entries found";
    for (int iev = 0; iev < ntrackletevreco; ++iev) {
      digitTreereco->GetEvent(iev);
      trackletTreereco->GetEvent(iev);
      for (const auto& digit : *digitsreco) {
        int det = digit.getDetector(); // chamber
        int stack = o2::trd::HelperMethods::getStack(det);
        int layer = o2::trd::HelperMethods::getLayer(det);
        int sector = o2::trd::HelperMethods::getSector(det);
        hDigitsPerLayer_reco[layer]->Fill(stack, sector * 2 + digit.getROB() % 2);
      }
      for (const auto& tracklet : *trackletsreco) {
        int det = tracklet.getHCID() / 2; // chamber
        int hcid = tracklet.getHCID();
        int stack = o2::trd::HelperMethods::getStack(det);
        int layer = o2::trd::HelperMethods::getLayer(det);
        int sector = o2::trd::HelperMethods::getSector(det);
        hTrackletsPerLayer_reco[layer]->Fill(stack, sector * 2 + tracklet.getHCID() % 2);
      }
    }
  }
  //post simulation
  TCanvas* c = new TCanvas("c", "trd digits distribution", 800, 800);
  c->Divide(3, 2, 0.05, 0.05);
  for (int layer = 0; layer < 6; ++layer) {
    c->cd(layer + 1);
    hDigitsPerLayer[layer]->Draw("COLZ");
  }
  c->SaveAs("DigitsPerLayerAfterSim.pdf");
  TCanvas* c1 = new TCanvas("c1", "trd tracklet distribution", 800, 800);
  c1->Divide(3, 2, 0.05, 0.05);
  for (int layer = 0; layer < 6; ++layer) {
    c1->cd(layer + 1);
    hTrackletsPerLayer[layer]->Draw("COLZ");
  }
  c1->SaveAs("TrackletsPerLayerAfterSim.pdf");
  //post simulation
  TCanvas* c2 = new TCanvas("c2", "trd digits distribution reconstruction", 800, 800);
  c2->Divide(3, 2, 0.05, 0.05);
  for (int layer = 0; layer < 6; ++layer) {
    c2->cd(layer + 1);
    hDigitsPerLayer_reco[layer]->Draw("COLZ");
  }
  c2->SaveAs("DigitsPerLayerAfterReco.pdf");
  TCanvas* c3 = new TCanvas("c3", "trd tracklet distribution reconstruction", 800, 800);
  c3->Divide(3, 2, 0.05, 0.05);
  for (int layer = 0; layer < 6; ++layer) {
    c3->cd(layer + 1);
    hTrackletsPerLayer_reco[layer]->Draw("COLZ");
  }
  c3->SaveAs("TrackletsPerLayerAfterReco.pdf");
  // calculate spectra differences, and the spectra should be empty.
  for (int layer = 0; layer < 6; layer++) {
    hDigitsPerLayer_diff[layer]->Add(hDigitsPerLayer[layer], hDigitsPerLayer_reco[layer], 1, -1);
    std::cout << "Digit Difference layer: " << layer << " sim - reco : " << hDigitsPerLayer_diff[layer]->Integral() << std::endl;
    hTrackletsPerLayer_diff[layer]->Add(hTrackletsPerLayer[layer], hTrackletsPerLayer_reco[layer], 1, -1);
    std::cout << "Tracklet Difference layer: " << layer << " sim - reco : " << hTrackletsPerLayer_diff[layer]->Integral() << std::endl;
  }
  TCanvas* c4 = new TCanvas("c4", "trd digits distribution diff", 800, 800);
  c4->Divide(3, 2, 0.05, 0.05);
  for (int layer = 0; layer < 6; ++layer) {
    c4->cd(layer + 1);
    hDigitsPerLayer_diff[layer]->Draw("COLZ");
  }
  c4->SaveAs("DigitsPerLayerDiff.pdf");
  TCanvas* c5 = new TCanvas("c5", "trd tracklet distribution diff", 800, 800);
  c5->Divide(3, 2, 0.05, 0.05);
  for (int layer = 0; layer < 6; ++layer) {
    c5->cd(layer + 1);
    hTrackletsPerLayer_diff[layer]->Draw("COLZ");
  }
  c5->SaveAs("TrackletsPerLayerDiff.pdf");
  // the _diff spectra *should* be empty
}
