// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
//#include "TPCQC/TrackCuts.h"
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

  tracksQC.initializeHistograms();
  gStyle->SetPalette(kCividis);
  qc::helpers::setStyleHistogram1D(tracksQC.getHistograms1D());
  qc::helpers::setStyleHistogram2D(tracksQC.getHistograms2D());

  // ===| event loop |============================================================
  for (int i = 0; i < tree->GetEntriesFast(); ++i) {
    tree->GetEntry(i);
    size_t nTracks = (maxTracks > 0) ? std::min(tpcTracks->size(), maxTracks) : tpcTracks->size();
    // ---| track loop |---
    for (int k = 0; k < nTracks; k++) {
      auto track = (*tpcTracks)[k];
      tracksQC.processTrack(track);
    }
  }

  // ===| get histograms |=======================================================
  auto& histos1D = tracksQC.getHistograms1D();
  auto& histos2D = tracksQC.getHistograms2D();

  // ===| create canvases |======================================================
  auto* c1 = new TCanvas("c1", "eta_phi_pt", 1200, 600);
  c1->Divide(3, 2);
  c1->cd(1);
  histos2D[6].Draw();
  c1->cd(2);
  histos1D[2].Draw();
  c1->cd(3);
  histos1D[4].Draw();
  c1->cd(4);
  histos2D[5].Draw();
  c1->cd(5);
  histos1D[5].Draw();
  gPad->SetLogy();
  c1->cd(6);
  histos1D[3].Draw();

  auto* c2 = new TCanvas("c2", "nClustersPerTrack_details", 1200, 300);
  c2->Divide(3, 1);
  c2->cd(1);
  histos2D[0].Draw();
  c2->cd(2);
  histos2D[2].Draw();
  c2->cd(3);
  histos2D[1].Draw();

  auto* c3 = new TCanvas("c3", "nClustersPerTrack_cuts", 800, 300);
  c3->Divide(2, 1);
  c3->cd(1);
  histos1D[0].Draw();
  c3->cd(2);
  histos1D[1].Draw();

  if (outputFileName.find(".root") != std::string::npos) {
    outputFileName.resize(outputFileName.size() - 5);
  }

  //===| dump canvases to a file |=============================================
  std::string canvasFile = outputFileName + "_canvas.root";
  auto f = std::unique_ptr<TFile>(TFile::Open(canvasFile.c_str(), "recreate"));
  f->WriteObject(c1, "eta_phi_pt");
  f->WriteObject(c2, "nClustersPerTrack_details");
  f->WriteObject(c3, "nClustersPerTrack_cuts");
  f->Close();
  delete c1;
  delete c2;
  delete c3;

  //===| dump histograms to a file |=============================================
  std::string histFile = outputFileName + ".root";
  tracksQC.dumpToFile(histFile);

  tracksQC.resetHistograms();
}
