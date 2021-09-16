#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <vector>
#include <iostream>
#include <string>
#include <TString.h>
#include <TH1I.h>
#include <TFile.h>
#include <TTree.h>
#include <TNtuple.h>
#include <TCanvas.h>
#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#endif

void CheckFakeClustersTracks(const std::string tracfile = "o2trac_its.root")
{
  TFile* file1 = TFile::Open(tracfile.data());
  TTree* recTree = (TTree*)file1->Get("o2sim");
  std::vector<o2::its::TrackITS>* recArr = nullptr;
  recTree->SetBranchAddress("ITSTrack", &recArr);
  std::vector<o2::MCCompLabel>* trkLabArr = nullptr;
  recTree->SetBranchAddress("ITSTrackMCTruth", &trkLabArr);
  std::vector<TH1I*> histLength;
  std::vector<TH1I*> histLength1Fake;
  histLength.resize(4);
  histLength1Fake.resize(4);
  for (int iH{4}; iH < 8; ++iH) {
    histLength[iH - 4] = new TH1I(Form("trk_len_%d", iH), Form("trk_len = %d", iH), 7, -.5, 6.5);
  }
  for (int iH{4}; iH < 8; ++iH) {
    histLength1Fake[iH - 4] = new TH1I(Form("trk_len_%d_1f", iH), Form("trk_len = %d, 1 Fake", iH), 7, -.5, 6.5);
  }
  auto nt = new TNtuple("TrackITSInfo", "Info about ITS tracks", "len:l0:l1:l2:l3:l4:l5:6");
  std::array<bool, 7> labels;
  for (unsigned int iEntry{0}; iEntry < recTree->GetEntriesFast(); ++iEntry) {
    recTree->GetEntry(iEntry);
    for (unsigned int iTrack{0}; iTrack < recArr->size(); ++iTrack) {
      auto& track = (*recArr)[iTrack];
      auto len = (*recArr)[iTrack].getNClusters();
      int count = 0;
      for (int iLabel{0}; iLabel < 7; ++iLabel) {
        if (track.hasHitOnLayer(iLabel)) {
          count++;
          if (track.isFakeOnLayer(iLabel)) {
            histLength[len - 4]->Fill(iLabel);
            if (track.getNFakeClusters() == 1) {
              histLength1Fake[len - 4]->Fill(iLabel);
            }
          }
        }
      }
      if (count != len) {
        std::cout << "mismatch!\n";
      }
    }
  }
  auto canvas = new TCanvas("c1", "c1", 2400, 1600);
  canvas->Divide(4, 2);
  for (int iH{0}; iH < 4; ++iH) {
    canvas->cd(iH + 1);
    histLength[iH]->Draw();
  }
  for (int iH{0}; iH < 4; ++iH) {
    canvas->cd(iH + 5);
    histLength1Fake[iH]->Draw();
  }
  canvas->SaveAs("fakeClusters.png", "recreate");
}