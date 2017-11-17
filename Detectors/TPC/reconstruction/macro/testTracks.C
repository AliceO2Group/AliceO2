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
#include "TCanvas.h"
#include "TStyle.h"
#include "TROOT.h"
#include "TH1F.h"
#include "TGraph.h"
#include "TLine.h"
#include "TMath.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCReconstruction/TrackTPC.h"
#include "DetectorsBase/Track.h"
#include "TPCReconstruction/Cluster.h"
#include "TPCBase/Mapper.h"
#endif

using namespace o2::TPC;

void drawSectorBoundaries();
void testTracks(int checkEvent = 0,
                std::string trackFile  ="~/AliSoftware/sw/BUILD/O2-latest-O2dir/O2/tracks.root",
                std::string clusterFile="~/AliSoftware/sw/BUILD/O2-latest-O2dir/O2/AliceO2_TGeant3.tpc.clusters_100_event.root",
                std::array<float,3> bField = {{0,0,0}})
{
  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(0.5);
  gStyle->SetTitleSize(24);

  Mapper &mapper = Mapper::instance();

  // Clusters
  TFile *clusFile = TFile::Open(clusterFile.data());
  TTree *clusterTree = (TTree *)gDirectory->Get("cbmsim");

  std::vector<o2::TPC::Cluster> *clusters=nullptr;
  clusterTree->SetBranchAddress("TPCClusterHW",&clusters);

  TGraph *grClusters   = new TGraph();
  grClusters->SetTitle("x - y plane (global); x [cm]; y [cm]");
  grClusters->SetMarkerColor(kOrange+2);
  grClusters->SetMarkerSize(1);
  TGraph *grClustersXZ = new TGraph();
  grClustersXZ->SetTitle("x - z plane (global); x [cm]; z [cm]");
  grClustersXZ->SetMarkerColor(kOrange+2);
  grClustersXZ->SetMarkerSize(1);
  TGraph2D *grClusters3D = new TGraph2D();
  grClusters3D->SetTitle("; x [cm]; y [cm]; z [cm]");
  grClusters3D->SetMarkerColor(kOrange+2);
  grClusters3D->SetMarkerSize(1);

  TGraph *grClustersLoc   = new TGraph();
  grClustersLoc->SetTitle("x - y plane (local); x [cm]; y [cm]");
  grClustersLoc->SetMarkerColor(kOrange+2);
  grClustersLoc->SetMarkerSize(1);
  TGraph *grClustersLocXZ = new TGraph();
  grClustersLocXZ->SetTitle("x - z plane (local); x [cm]; z [cm]");
  grClustersLocXZ->SetMarkerColor(kOrange+2);
  grClustersLocXZ->SetMarkerSize(1);
  TGraph2D *grClustersLoc3D = new TGraph2D();
  grClustersLoc3D->SetTitle("; x [cm]; y [cm]; z [cm]");
  grClustersLoc3D->SetMarkerColor(kOrange+2);
  grClustersLoc3D->SetMarkerSize(1);


  int clusCounter = 0;
  clusterTree->GetEntry(checkEvent);
  for(auto& clusterObject : *clusters) {
    Cluster* inputcluster = &clusterObject;
    const CRU cru(inputcluster->getCRU());

    const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
    const int rowInSector       = inputcluster->getRow() + region.getGlobalRowOffset();
    const GlobalPadNumber pad   = mapper.globalPadNumber(PadPos(rowInSector, inputcluster->getPadMean()));
    const PadCentre& padCentre  = mapper.padCentre(pad);
    const float localYfactor    = (cru.side()==Side::A)?-1.f:1.f;
          float zPosition       = Digitizer::getZfromTimeBin(inputcluster->getTimeMean(), cru.side());

    LocalPosition3D posLoc(padCentre.X(), localYfactor*padCentre.Y(), zPosition);
    GlobalPosition3D posGlob = Mapper::LocalToGlobal(posLoc, cru.sector());

    const float clusterX = posGlob.X();
    const float clusterY = posGlob.Y();
    const float clusterZ = zPosition;

    grClusters->SetPoint(clusCounter, clusterX, clusterY);
    grClustersXZ->SetPoint(clusCounter, clusterX, clusterZ);
    grClusters3D->SetPoint(clusCounter, clusterX, clusterY, clusterZ);
    grClustersLoc->SetPoint(clusCounter, posLoc.X(), posLoc.Y());
    grClustersLocXZ->SetPoint(clusCounter, posLoc.X(), posLoc.Z());
    grClustersLoc3D->SetPoint(clusCounter++, posLoc.X(), posLoc.Y(), posLoc.Z());


  }

  // Tracks
  TFile *tracks = TFile::Open(trackFile.data());
  TTree *trackTree = (TTree *)gDirectory->Get("events");

  TH1F *hResY = new TH1F("hResY", "Pad residuals; Residual y [cm]; Entries", 101, -2, 2);
  TH1F *hResZ = new TH1F("hResZ", "Time residuals; Residual z [cm]; Entries", 101, -2, 2);

  TGraph *grTracks     = new TGraph();
  TGraph *grTracksXZ   = new TGraph();
  TGraph2D *grTracks3D = new TGraph2D();
  TGraph *grTracksLoc  = new TGraph();
  TGraph *grTracksLocXZ= new TGraph();
  TGraph2D *grTracksLoc3D= new TGraph2D();


  std::vector<TrackTPC> *arrTracks = 0;
  trackTree->SetBranchAddress("Tracks", &arrTracks);

  int counter = 0;

  for(int iEv=0; iEv < trackTree->GetEntries(); ++iEv) {
    trackTree->GetEntry(iEv);
    for (auto trackObject : *arrTracks) {
      std::vector<Cluster> clCont;
      trackObject.getClusterVector(clCont);
      for(auto& clusterObject : clCont) {
        const CRU cru(clusterObject.getCRU());

        const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
        const int rowInSector       = clusterObject.getRow() + region.getGlobalRowOffset();
        const GlobalPadNumber pad   = mapper.globalPadNumber(PadPos(rowInSector, clusterObject.getPadMean()));
        const PadCentre& padCentre  = mapper.padCentre(pad);
        const float localYfactor    = (cru.side()==Side::A)?-1.f:1.f;
              float zPosition       = Digitizer::getZfromTimeBin(clusterObject.getTimeMean(), cru.side());

        LocalPosition3D clusLoc(padCentre.X(), localYfactor*padCentre.Y(), zPosition);
        GlobalPosition3D clusGlob = Mapper::LocalToGlobal(clusLoc, cru.sector());

        // Track parameters are in local coordinate system - propagate to pad row of the cluster
        trackObject.propagateTo(clusLoc.X(), bField);

        LocalPosition3D trackLoc(trackObject.getX(), trackObject.getY(), trackObject.getZ());
        /// \todo sector hardcoded for the time  being
        GlobalPosition3D trackGlob = Mapper::LocalToGlobal(trackLoc, 0.1632);

        const float resY = trackLoc.Y() - clusLoc.Y();
        const float resZ = trackLoc.Y() - clusLoc.Z();

        hResY->Fill(resY);
        hResZ->Fill(resZ);

        if(iEv == checkEvent) {
          grTracks->SetPoint(counter, trackGlob.X(), trackGlob.Y());
          grTracksXZ->SetPoint(counter, trackGlob.X(), trackGlob.Z());
          grTracks3D->SetPoint(counter, trackGlob.X(), trackGlob.Y(), trackGlob.Z());
          grTracksLoc->SetPoint(counter, trackLoc.X(), trackLoc.Y());
          grTracksLocXZ->SetPoint(counter, trackLoc.X(), trackLoc.Z());
          grTracksLoc3D->SetPoint(counter++, trackLoc.X(), trackLoc.Y(), trackLoc.Z());

        }
      }
    }
  }

  TCanvas *CTracks = new TCanvas("CTracks", "CTracks", 1200, 600);
  CTracks->Divide(2,1);
  CTracks->cd(1);
  grClusters->Draw("ap");
  grClusters->GetXaxis()->SetLimits(-250, 250);
  grClusters->SetMinimum(-250);
  grClusters->SetMaximum(250);
  grTracks->Draw("p");
  drawSectorBoundaries();
  CTracks->cd(2);
  grClustersXZ->Draw("ap");
  grClustersXZ->GetXaxis()->SetLimits(-250, 250);
  grClustersXZ->SetMinimum(-250);
  grClustersXZ->SetMaximum(250);
  grTracksXZ->Draw("p");

  TCanvas *CTracksLoc = new TCanvas("CTracksLoc", "CTracksLoc", 1200, 600);
  CTracksLoc->Divide(2,1);
  CTracksLoc->cd(1);
  grClustersLoc->Draw("ap");
  grClustersLoc->GetXaxis()->SetLimits(80, 150);
  grClustersLoc->SetMinimum(-30);
  grClustersLoc->SetMaximum(30);
  grTracksLoc->Draw("p");
  CTracksLoc->cd(2);
  grClustersLocXZ->Draw("ap");
  grClustersLocXZ->GetXaxis()->SetLimits(80, 150);
  grClustersLocXZ->SetMinimum(200);
  grClustersLocXZ->SetMaximum(250);
  grTracksLocXZ->Draw("p");


  TCanvas *c3D = new TCanvas();
  grClusters3D->Draw("ap");

  grClusters3D->GetXaxis()->SetLimits(-250, 250);
  grClusters3D->GetYaxis()->SetLimits(-250, 250);
  grClusters3D->GetZaxis()->SetLimits(-250, 250);
  grTracks3D->Draw("p same");

  TCanvas *cRes = new TCanvas();
  cRes->Divide(2,1);
  cRes->cd(1);
  hResY->Draw();
  cRes->cd(2);
  hResZ->Draw();
}

void drawSectorBoundaries()
{
  TLine *sectorBoundary[18];
  for(float i = 0; i<18; ++i) {
    const float angle = i*20.f*TMath::DegToRad();
    sectorBoundary[int(i)] = new TLine(80.f*std::cos(angle), 80.f*std::sin(angle), 250.f*std::cos(angle), 250.f*std::sin(angle));
    sectorBoundary[int(i)]->SetLineStyle(2);
    sectorBoundary[int(i)]->SetLineColor(kGray);
    sectorBoundary[int(i)]->Draw("same");
  }
}

