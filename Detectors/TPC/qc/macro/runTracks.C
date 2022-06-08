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

///
/// @file   runTracks.C
/// @author Stefan Heckel, sheckel@cern.ch
///

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "TPCQC/Tracks.h"
#include "TPCQC/Helpers.h"
#endif

using namespace o2::tpc;

void runTracks(std::string outputFileName = "tpcQcTracks", std::string_view inputFileName = "tpctracks.root", const size_t maxTracks = 0)
{
  // ===| track file and tree |=================================================
  auto file = TFile::Open(inputFileName.data());
  auto tree = (TTree*)file->Get("tpcrec");
  if (tree == nullptr) {
    std::cout << "Error getting tree\n";
    return;
  }

  // ===| branch setup |==========================================================
  std::vector<TrackTPC>* tpcTracks = nullptr;
  tree->SetBranchAddress("TPCTracks", &tpcTracks);

  // ===| create Tracks object |=====================================================
  qc::Tracks tracksQC;
  // set track cutss defaults are (eta = 1.0, nCluster = 60, dEdxTot  = 20)
  tracksQC.setTrackCuts(1., 60, 20.);
  tracksQC.initializeHistograms();
  gStyle->SetPalette(kCividis);
  qc::helpers::setStyleHistogramsInMap(tracksQC.getMapHist());

  // ===| event loop |============================================================
  for (int i = 0; i < tree->GetEntriesFast(); ++i) {
    tree->GetEntry(i);
    size_t nTracks = (maxTracks > 0) ? std::min(tpcTracks->size(), maxTracks) : tpcTracks->size();
    // ---| track loop |---
    for (size_t k = 0; k < nTracks; k++) {
      auto track = (*tpcTracks)[k];
      tracksQC.processTrack(track);
    }
  }

  // ===| get histograms |=======================================================
  auto& mMapOfHisto = tracksQC.getMapHist();

  // 1d hitograms
  auto& hNClustersBeforeCuts = mMapOfHisto["hNClustersBeforeCuts"];
  auto& hNClustersAfterCuts = mMapOfHisto["hNClustersAfterCuts"];
  auto& hEta = mMapOfHisto["hEta"];
  auto& hPhiAside = mMapOfHisto["hPhiAside"];
  auto& hPhiCside = mMapOfHisto["hPhiCside"];
  auto& hPt = mMapOfHisto["hPt"];
  auto& hSign = mMapOfHisto["hSign"];
  auto& hEtaNeg = mMapOfHisto["hEtaNeg"];
  auto& hEtaPos = mMapOfHisto["hEtaPos"];
  auto& hPhiAsideNeg = mMapOfHisto["hPhiAsideNeg"];
  auto& hPhiAsidePos = mMapOfHisto["hPhiAsidePos"];
  auto& hPhiCsideNeg = mMapOfHisto["hPhiCsideNeg"];
  auto& hPhiCsidePos = mMapOfHisto["hPhiCsidePos"];
  auto& hPtNeg = mMapOfHisto["hPtNeg"];
  auto& hPtPos = mMapOfHisto["hPtPos"];
  auto& hEtaBeforeCuts = mMapOfHisto["hEtaBeforeCuts"];
  auto& hPtBeforeCuts = mMapOfHisto["hPtBeforeCuts"];
  auto& hQOverPt = mMapOfHisto["hQOverPt"];
  auto& hPhiBothSides = mMapOfHisto["hPhiBothSides"];
  // 2d histograms
  auto& h2DNClustersEta = mMapOfHisto["h2DNClustersEta"];
  auto& h2DNClustersPhiAside = mMapOfHisto["h2DNClustersPhiAside"];
  auto& h2DNClustersPhiCside = mMapOfHisto["h2DNClustersPhiCside"];
  auto& h2DNClustersPt = mMapOfHisto["h2DNClustersPt"];
  auto& h2DEtaPhi = mMapOfHisto["h2DEtaPhi"];
  auto& h2DEtaPhiNeg = mMapOfHisto["h2DEtaPhiNeg"];
  auto& h2DEtaPhiPos = mMapOfHisto["h2DEtaPhiPos"];
  auto& h2DNClustersEtaBeforeCuts = mMapOfHisto["h2DNClustersEtaBeforeCuts"];
  auto& h2DNClustersPtBeforeCuts = mMapOfHisto["h2DNClustersPtBeforeCuts"];
  auto& h2DEtaPhiBeforeCuts = mMapOfHisto["h2DEtaPhiBeforeCuts"];
  auto& h2DQOverPtPhiAside = mMapOfHisto["h2DQOverPtPhiAside"];
  auto& h2DQOverPtPhiCside = mMapOfHisto["h2DQOverPtPhiCside"];

  // 1d histograms
  auto& hEtaRatio = mMapOfHisto["hEtaRatio"];
  auto& hPhiAsideRatio = mMapOfHisto["hPhiAsideRatio"];
  auto& hPhiCsideRatio = mMapOfHisto["hPhiCsideRatio"];
  auto& hPtRatio = mMapOfHisto["hPtRatio"];

  // ===| create canvases |======================================================
  auto* c1 = new TCanvas("c1", "eta_phi_pt", 1200, 600);
  c1->Divide(3, 2);
  c1->cd(1);
  h2DEtaPhiPos->Draw();
  c1->cd(2);
  hEta->Draw();
  c1->cd(3);
  hPhiCside->Draw();
  c1->cd(4);
  h2DEtaPhiNeg->Draw();
  c1->cd(5);
  hPt->Draw();
  gPad->SetLogy();
  c1->cd(6);
  hPhiAside->Draw();

  auto* c2 = new TCanvas("c2", "nClustersPerTrack_details", 1200, 300);
  c2->Divide(3, 1);
  c2->cd(1);
  h2DNClustersEta->Draw();
  c2->cd(2);
  h2DNClustersPhiCside->Draw();
  c2->cd(3);
  h2DNClustersPhiAside->Draw();

  auto* c3 = new TCanvas("c3", "nClustersPerTrack_cuts", 800, 300);
  c3->Divide(2, 1);
  c3->cd(1);
  hNClustersBeforeCuts->Draw();
  c3->cd(2);
  hNClustersAfterCuts->Draw();

  // ratio plots
  auto* c4 = new TCanvas("c4", "ratio", 800, 300);
  c4->Divide(2, 2);
  c4->cd(1);
  hEtaRatio->Draw();
  c4->cd(2);
  hPhiAsideRatio->Draw();
  c4->cd(3);
  hPhiCsideRatio->Draw();
  c4->cd(4);
  hPtRatio->Draw();

  if (outputFileName.find(".root") != std::string::npos) {
    outputFileName.resize(outputFileName.size() - 5);
  }

  //===| dump canvases to a file |=============================================
  std::string canvasFile = outputFileName + "_canvas.root";
  auto f = std::unique_ptr<TFile>(TFile::Open(canvasFile.c_str(), "recreate"));
  f->WriteObject(c1, "eta_phi_pt");
  f->WriteObject(c2, "nClustersPerTrack_details");
  f->WriteObject(c3, "nClustersPerTrack_cuts");
  f->WriteObject(c4, "RatioTrack_cuts");
  f->Close();
  delete c1;
  delete c2;
  delete c3;
  delete c4;

  //===| dump histograms to a file |=============================================
  std::string histFile = outputFileName + ".root";
  tracksQC.dumpToFile(histFile);

  tracksQC.resetHistograms();
}
