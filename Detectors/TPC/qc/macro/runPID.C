// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "TPCQC/PID.h"
#include "TPCQC/Helpers.h"
#include "TPCQC/TrackCuts.h"
#endif

using namespace o2::tpc;

void runPID(std::string outputFileName = "PID", std::string_view inputFileName = "tpctracks.root", const size_t maxTracks = 0)
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

  // ===| create PID object |=====================================================
  qc::PID pid;

  pid.initializeHistograms();
  gStyle->SetPalette(kCividis);
  qc::helpers::setStyleHistogram1D(pid.getHistograms1D());
  qc::helpers::setStyleHistogram2D(pid.getHistograms2D());

  // ===| event loop |============================================================
  for (int i = 0; i < tree->GetEntriesFast(); ++i) {
    tree->GetEntry(i);
    size_t nTracks = (maxTracks > 0) ? std::min(tpcTracks->size(), maxTracks) : tpcTracks->size();
    // ---| track loop |---
    for (int k = 0; k < nTracks; k++) {
      auto track = (*tpcTracks)[k];
      pid.processTrack(track);
    }
  }

  // ===| create canvas |========================================================
  auto* c1 = new TCanvas("c1", "PID", 1200, 600);
  c1->Divide(2, 2);
  auto histos2D = pid.getHistograms2D();
  c1->cd(1);
  histos2D[0].Draw();
  gPad->SetLogz();
  c1->cd(2);
  histos2D[1].Draw();
  gPad->SetLogz();
  c1->cd(3);
  histos2D[2].Draw();
  gPad->SetLogz();
  c1->cd(4);
  histos2D[3].Draw();
  gPad->SetLogz();
  gPad->SetLogx();

  if (outputFileName.find(".root") != std::string::npos) {
    outputFileName.resize(outputFileName.size() - 5);
  }

  std::string canvasFile = outputFileName + "_canvas.root";
  auto f = std::unique_ptr<TFile>(TFile::Open(canvasFile.c_str(), "recreate"));
  f->WriteObject(c1, "PID");
  f->Close();
  delete c1;

  //===| dump histograms to a file |=============================================
  std::string histFile = outputFileName + ".root";
  pid.dumpToFile(histFile);

  pid.resetHistograms();
}