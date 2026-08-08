// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file postClusterSizeVsEta.C
/// \brief A post-processing macro to draw average cluster size vs eta

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TProfile.h>
#endif

using namespace std;

// ### required input file: CheckClusters.root, which is the output of CheckClusters.C macro
void postClusterSizeVsEta(const std::string& strFileInput = "CheckClusters.root")
{
  gStyle->SetOptStat(0);

  TFile* fileInput = new TFile(strFileInput.c_str());
  TTree* tree = (TTree*)fileInput->Get("ntc");
  std::cout << "Opened tree: " << tree->GetName() << ", entries = " << tree->GetEntries() << std::endl;

  // set branch addresses
  Float_t event;
  Float_t mcTrackID;
  Float_t hitLocX, hitLocZ;
  Float_t hitGlobX, hitGlobY, hitGlobZ;
  Float_t clusGlobX, clusGlobY, clusGlobZ;
  Float_t clusLocX, clusLocZ;
  Float_t rofFrame;
  Float_t clusSize;
  Float_t chipID;
  Float_t layer;
  Float_t disk;
  Float_t subdet;
  Float_t row, col;
  Float_t pt;

  // set branch addresses
  tree->SetBranchAddress("event", &event);
  tree->SetBranchAddress("mcTrackID", &mcTrackID);
  tree->SetBranchAddress("hitLocX", &hitLocX);
  tree->SetBranchAddress("hitLocZ", &hitLocZ);
  tree->SetBranchAddress("hitGlobX", &hitGlobX);
  tree->SetBranchAddress("hitGlobY", &hitGlobY);
  tree->SetBranchAddress("hitGlobZ", &hitGlobZ);
  tree->SetBranchAddress("clusGlobX", &clusGlobX);
  tree->SetBranchAddress("clusGlobY", &clusGlobY);
  tree->SetBranchAddress("clusGlobZ", &clusGlobZ);
  tree->SetBranchAddress("clusLocX", &clusLocX);
  tree->SetBranchAddress("clusLocZ", &clusLocZ);
  tree->SetBranchAddress("rofFrame", &rofFrame);
  tree->SetBranchAddress("clusSize", &clusSize);
  tree->SetBranchAddress("chipID", &chipID);
  tree->SetBranchAddress("layer", &layer);
  tree->SetBranchAddress("disk", &disk);
  tree->SetBranchAddress("subdet", &subdet);
  tree->SetBranchAddress("row", &row);
  tree->SetBranchAddress("col", &col);
  tree->SetBranchAddress("pt", &pt);

  // Some QA histograms
  TH1F* hPt = new TH1F("hPt", "p_{T};p_{T};Entries", 100, 0., 10.);
  TH1F* hClusSize = new TH1F("hClusSize", "Cluster size;clusSize;Entries", 20, 0., 20.);
  TH1F* hLayer = new TH1F("hLayer", "Layer;layer;Entries", 20, -0.5, 19.5);
  TH1F* hDxGlob = new TH1F("hDxGlob", "clusGlobX - hitGlobX;#DeltaX [global];Entries", 200, -1., 1.);
  TH1F* hDzGlob = new TH1F("hDzGlob", "clusGlobZ - hitGlobZ;#DeltaZ [global];Entries", 200, -1., 1.);
  TH2F* hHitXY = new TH2F("hHitXY", "Hit global XY;hitGlobX;hitGlobY", 200, -20., 20., 200, -20., 20.);
  TH2F* hClusVsHitX = new TH2F("hClusVsHitX", "clusGlobX vs hitGlobX;hitGlobX;clusGlobX", 200, -20., 20., 200, -20., 20.);

  // histograms for cluster size vs eta for each barrel layer:
  const int nLayers = 11;
  TH2F* hClustSizePerLayerVsEta[nLayers];
  for (int i = 0; i < nLayers; i++) {
    hClustSizePerLayerVsEta[i] = new TH2F(Form("hClustSizePerLayerVsEta_Lay%d", i), Form("Cluster size vs eta for layer %d;#eta;Cluster size", i), 200, -5, 5, 101, -0.5, 100.5);
  }

  // Loop over entries
  const Long64_t nEntries = tree->GetEntries();
  for (Long64_t i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);

    // Fill QA histograms
    float dXGlob = clusGlobX - hitGlobX;
    float dZGlob = clusGlobZ - hitGlobZ;
    hPt->Fill(pt);
    hClusSize->Fill(clusSize);
    hLayer->Fill(layer);
    hDxGlob->Fill(dXGlob);
    hDzGlob->Fill(dZGlob);
    hHitXY->Fill(hitGlobX, hitGlobY);
    hClusVsHitX->Fill(hitGlobX, clusGlobX);

    // cls size vs eta:
    float clustR = sqrt(clusGlobX * clusGlobX + clusGlobY * clusGlobY);
    float clustPhi = atan2(clusGlobY, clusGlobX);
    float clustTheta = atan2(clustR, clusGlobZ);
    float clustEta = -log(tan(clustTheta / 2));

    // !!! important: to avoid VD layers (numeration for ML starts from 0, while VD layers are also numbered as 0,1,2)
    if (clustR > 5) // cm
      hClustSizePerLayerVsEta[(int)layer + 3]->Fill(clustEta, clusSize);
    else if (layer < 3) // VD layers
      hClustSizePerLayerVsEta[(int)layer]->Fill(clustEta, clusSize);

    // progress print
    if ((i + 1) % 200000 == 0) {
      std::cout << "Processed " << (i + 1) << " / " << nEntries << " entries" << std::endl;
    }
  }

  // Save histograms to file
  TFile* fout = TFile::Open("clusterSizes_vs_eta.root", "RECREATE");
  hPt->Write();
  hClusSize->Write();
  hLayer->Write();
  hDxGlob->Write();
  hDzGlob->Write();
  hHitXY->Write();
  hClusVsHitX->Write();

  // draw some QA histograms
  TCanvas* c1 = new TCanvas("canv_clusters_QA", "Clusters QA", 1200, 800);
  c1->Divide(2, 2);
  c1->cd(1);
  hPt->Draw();
  c1->cd(2);
  hClusSize->Draw();
  c1->cd(3);
  hDxGlob->Draw();
  c1->cd(4);
  hHitXY->Draw("COLZ");

  int colors[] = {kRed, kBlue + 1, kMagenta + 1,
                  kRed, kBlue + 1, kMagenta + 1,
                  kCyan + 1, kGray + 2, kRed, kBlue, kMagenta + 1, kCyan, kAzure + 1, kOrange - 9, kRed + 2, kBlue + 2, kMagenta + 2};

  TCanvas* canv_clsSize_vs_eta[nLayers];
  TProfile* profPerLayerVsEta[nLayers];
  for (int i = 0; i < nLayers; i++) {
    canv_clsSize_vs_eta[i] = new TCanvas(Form("canv_clsSize_vs_eta_Lay%d", i), Form("Cluster size vs eta for layer %d", i), 800, 600);
    hClustSizePerLayerVsEta[i]->Draw("COLZ");
    gPad->SetLogz();
    profPerLayerVsEta[i] = hClustSizePerLayerVsEta[i]->ProfileX();
    profPerLayerVsEta[i]->SetLineColor(colors[i]);
    profPerLayerVsEta[i]->SetMarkerColor(colors[i]);
    profPerLayerVsEta[i]->SetMarkerStyle(i < 8 ? 20 : 24);
    profPerLayerVsEta[i]->SetTitle(";#eta;#LTcluster size#GT");
    profPerLayerVsEta[i]->DrawCopy("same");

    hClustSizePerLayerVsEta[i]->Write();
    profPerLayerVsEta[i]->Write();
  }

  // ### canvas with profiles for 3 VD layers
  TCanvas* canv_av_clsSize_vs_eta_VD_layers = new TCanvas("canv_clsSize_vs_eta_VD_layers", "Cluster size vs eta for VD layers", 800, 600);
  TLegend* legLayersVD = new TLegend(0.3, 0.72, 0.65, 0.89);
  for (int i = 0; i < 3; i++) {
    profPerLayerVsEta[i]->GetYaxis()->SetRangeUser(0., 60.);
    profPerLayerVsEta[i]->DrawCopy(i == 0 ? "P" : "P same");
    legLayersVD->AddEntry(profPerLayerVsEta[i], Form("VD layer %d", i), "P");
  }
  legLayersVD->Draw();
  gPad->SetGrid();
  canv_av_clsSize_vs_eta_VD_layers->SaveAs("clsSize_vs_eta_VD_layers.png");
  canv_av_clsSize_vs_eta_VD_layers->Write();

  // ### canvas with profiles for MLOT layers
  TCanvas* canv_av_clsSize_vs_eta_MLOT_layers = new TCanvas("canv_clsSize_vs_eta_MLOT_layers", "Cluster size vs eta for MLOT layers", 800, 600);
  TLegend* legLayersMLOT = new TLegend(0.3, 0.52, 0.65, 0.89);
  for (int i = 3; i < nLayers; i++) {
    profPerLayerVsEta[i]->GetYaxis()->SetRangeUser(0., 12.5);
    profPerLayerVsEta[i]->GetXaxis()->SetRangeUser(-3.5, 3.5);
    profPerLayerVsEta[i]->DrawCopy(i == 3 ? "P" : "P same");
    legLayersMLOT->AddEntry(profPerLayerVsEta[i], Form("MLOT layer %d", i), "P");
  }
  legLayersMLOT->Draw();
  gPad->SetGrid();
  canv_av_clsSize_vs_eta_MLOT_layers->SaveAs("clsSize_vs_eta_MLOT_layers.png");
  canv_av_clsSize_vs_eta_MLOT_layers->Write();
}