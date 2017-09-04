// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <vector>
#include <fstream>
#include <iostream>

#include "TROOT.h"
#include "TLine.h"
#include "TMath.h"
#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TH1F.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/DigitMC.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCReconstruction/TrackTPC.h"
#include "DetectorsBase/Track.h"
#include "TPCSimulation/Cluster.h"
#include "TPCBase/Mapper.h"

using namespace o2::TPC;

void drawSectorBoundaries();

void testHitsDigitsClusters(int iEv=0,
               std::string simFile="~/AliSoftware/sw/BUILD/O2-latest-O2dir/O2/AliceO2_TGeant3.tpc.mc_100_event.root",
               std::string digiFile="~/AliSoftware/sw/BUILD/O2-latest-O2dir/O2/AliceO2_TGeant3.tpc.digi_100_event.root",
               std::string clusFile="~/AliSoftware/sw/BUILD/O2-latest-O2dir/O2/AliceO2_TGeant3.tpc.clusters_100_event.root",
               std::string trackFile="~/AliSoftware/sw/BUILD/O2-latest-O2dir/O2/tracks.root")
{
  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(0.5);
  gStyle->SetTitleSize(24);

  // process the hits
  TFile *hitFile   = TFile::Open(simFile.data());
  TTree *hitTree = (TTree *)gDirectory->Get("cbmsim");

  TClonesArray pointArr("Point");
  TClonesArray *points(&pointArr);
  hitTree->SetBranchAddress("TPCPoint",&points);

  TGraph *grHitsA = new TGraph();
  grHitsA->SetTitle("A side ; x [cm]; y [cm]");
  grHitsA->SetMarkerColor(kBlue+2);
  TGraph *grHitsC = new TGraph();
  grHitsC->SetTitle("C side ; x [cm]; y [cm]");
  grHitsC->SetMarkerColor(kBlue+2);
  TGraph *grHitsAxz = new TGraph();
  grHitsAxz->SetMarkerColor(kBlue+2);
  grHitsAxz->SetTitle("; x [cm]; z [cm]");
  TGraph *grHitsCxz = new TGraph();
  grHitsCxz->SetMarkerColor(kBlue+2);
  grHitsCxz->SetTitle("; x [cm]; z [cm]");


  int hitCounterA = 0;
  int hitCounterC = 0;
  hitTree->GetEntry(iEv);
  for(auto pointObject : *points) {
    Point *inputpoint = static_cast<Point *>(pointObject);
    // A side
    if(inputpoint->GetZ() > 0 ) {
      grHitsA->SetPoint(hitCounterA, inputpoint->GetX(), inputpoint->GetY());
      grHitsAxz->SetPoint(hitCounterA++, inputpoint->GetX(), inputpoint->GetZ());
    }
    // C side
    if(inputpoint->GetZ() < 0 ) {
      grHitsC->SetPoint(hitCounterC, inputpoint->GetX(), inputpoint->GetY());
      grHitsCxz->SetPoint(hitCounterC++, inputpoint->GetX(), inputpoint->GetZ());
    }
  }

  // process the digits
  TFile *digitFile = TFile::Open(digiFile.data());
  TTree *digitTree = (TTree *)gDirectory->Get("cbmsim");

  TClonesArray digitArr("DigitMC");
  TClonesArray *digits(&digitArr);
  digitTree->SetBranchAddress("TPCDigitMC",&digits);

  const Mapper& mapper = Mapper::instance();

  TGraph *grDigitsA = new TGraph();
  grDigitsA->SetMarkerColor(kGreen+2);
  TGraph *grDigitsC = new TGraph();
  grDigitsC->SetMarkerColor(kGreen+2);
  TGraph *grDigitsAxz = new TGraph();
  grDigitsAxz->SetMarkerColor(kGreen+2);
  TGraph *grDigitsCxz = new TGraph();
  grDigitsCxz->SetMarkerColor(kGreen+2);

  int digiCounterA = 0;
  int digiCounterC = 0;
  digitTree->GetEntry(iEv);
  for(auto digitObject : *digits) {
    DigitMC *inputdigit = static_cast<DigitMC *>(digitObject);

    const CRU cru(inputdigit->getCRU());

    const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
    const int rowInSector       = inputdigit->getRow() + region.getGlobalRowOffset();
    const GlobalPadNumber pad   = mapper.globalPadNumber(PadPos(rowInSector, inputdigit->getPad()));
    const PadCentre& padCentre  = mapper.padCentre(pad);
    const float localYfactor    = (cru.side()==Side::A)?-1.f:1.f;
          float zPosition       = Digitizer::getZfromTimeBin(inputdigit->getTimeStamp(), cru.side());

    LocalPosition3D posLoc(padCentre.X(), localYfactor*padCentre.Y(), zPosition);
    GlobalPosition3D posGlob = Mapper::LocalToGlobal(posLoc, cru.sector());

    const float digiX = posGlob.X();
    const float digiY = posGlob.Y();
    const float digiZ = zPosition;

    if(cru.side() == Side::A) {
      grDigitsA->SetPoint(digiCounterA, digiX, digiY);
      grDigitsAxz->SetPoint(digiCounterA++, digiX, digiZ);
    }
    if(cru.side() == Side::C) {
      grDigitsC->SetPoint(digiCounterC, digiX, digiY);
      grDigitsCxz->SetPoint(digiCounterC++, digiX, digiZ);
    }
  }

  // process the clusters
  TFile *clusterFile = TFile::Open(clusFile.data());
  TTree *clusterTree = (TTree *)gDirectory->Get("cbmsim");

  TClonesArray clusterArr("Cluster");
  TClonesArray *clusters(&clusterArr);
  clusterTree->SetBranchAddress("TPCClusterHW",&clusters);

  TGraph *grClustersA = new TGraph();
  grClustersA->SetMarkerColor(kRed+2);
  grClustersA->SetMarkerSize(1);
  TGraph *grClustersC = new TGraph();
  grClustersC->SetMarkerColor(kRed+2);
  grClustersC->SetMarkerSize(1);
  TGraph *grClustersAxz = new TGraph();
  grClustersAxz->SetMarkerColor(kRed+2);
  grClustersAxz->SetMarkerSize(1);
  TGraph *grClustersCxz = new TGraph();
  grClustersCxz->SetMarkerColor(kRed+2);
  grClustersCxz->SetMarkerSize(1);

  int clusCounterA = 0;
  int clusCounterC = 0;
  clusterTree->GetEntry(iEv);
  for(auto clusterObject : *clusters) {
    Cluster *inputcluster = static_cast<Cluster *>(clusterObject);
    const CRU cru(inputcluster->getCRU());

    const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
    const int rowInSector       = inputcluster->getRow() + region.getGlobalRowOffset();
    const GlobalPadNumber pad   = mapper.globalPadNumber(PadPos(rowInSector, inputcluster->getPadMean()));
    const PadCentre& padCentre  = mapper.padCentre(pad);
    const float localYfactor    = (cru.side()==Side::A)?-1.f:1.f;
          float zPosition       = Digitizer::getZfromTimeBin(inputcluster->getTimeMean(), cru.side());

    LocalPosition3D posLoc(padCentre.X(), localYfactor*padCentre.Y(), zPosition);
    GlobalPosition3D posGlob = Mapper::LocalToGlobal(posLoc, cru.sector());

    const float digiX = posGlob.X();
    const float digiY = posGlob.Y();
    const float digiZ = zPosition;

    if(cru.side() == Side::A) {
      grClustersA->SetPoint(clusCounterA, digiX, digiY);
      grClustersAxz->SetPoint(clusCounterA++, digiX, digiZ);
    }
    if(cru.side() == Side::C) {
      grClustersC->SetPoint(clusCounterC, digiX, digiY);
      grClustersCxz->SetPoint(clusCounterC++, digiX, digiZ);
    }
  }

  // Tracks
  TFile *tracks = TFile::Open(trackFile.data());
  TTree *trackTree = (TTree *)gDirectory->Get("events");

  TGraph *grClustersTrackA = new TGraph();
  grClustersTrackA->SetMarkerColor(kOrange+2);
  grClustersTrackA->SetMarkerSize(1);
  TGraph *grClustersTrackC = new TGraph();
  grClustersTrackC->SetMarkerColor(kOrange+2);
  grClustersTrackC->SetMarkerSize(1);
  TGraph *grClustersTrackAxz = new TGraph();
  grClustersTrackAxz->SetMarkerColor(kOrange+2);
  grClustersTrackAxz->SetMarkerSize(1);
  TGraph *grClustersTrackCxz = new TGraph();
  grClustersTrackCxz->SetMarkerColor(kOrange+2);
  grClustersTrackCxz->SetMarkerSize(1);

  TGraph *grTrackA = new TGraph();
  TGraph *grTrackC = new TGraph();
  TGraph *grTrackxz = new TGraph();

  TH1F *hResY = new TH1F("hResY", "; Residual y [cm]; Entries", 101, -2, 2);
  TH1F *hResZ = new TH1F("hResZ", "; Residual z [cm]; Entries", 101, -2, 2);

  std::vector<TrackTPC> *arrTracks = 0;
  trackTree->SetBranchAddress("Tracks", &arrTracks);

  int clusCounterTrackA = 0;
  int clusCounterTrackC = 0;

  int trackCounterA = 0;
  int trackCounterC = 0;
  int trackCounterXZ = 0;

  trackTree->GetEntry(iEv);
  for (auto trackObject : *arrTracks) {
    std::vector<Cluster> clCont;
    trackObject.getClusterVector(clCont);
    for(auto clusterObject : clCont) {
      const CRU cru(clusterObject.getCRU());

      const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
      const int rowInSector       = clusterObject.getRow() + region.getGlobalRowOffset();
      const GlobalPadNumber pad   = mapper.globalPadNumber(PadPos(rowInSector, clusterObject.getPadMean()));
      const PadCentre& padCentre  = mapper.padCentre(pad);
      const float localYfactor    = (cru.side()==Side::A)?-1.f:1.f;
            float zPosition       = Digitizer::getZfromTimeBin(clusterObject.getTimeMean(), cru.side());

      LocalPosition3D posLoc(padCentre.X(), localYfactor*padCentre.Y(), zPosition);
      GlobalPosition3D posGlob = Mapper::LocalToGlobal(posLoc, cru.sector());

      const float digiX = posGlob.X();
      const float digiY = posGlob.Y();
      const float digiZ = zPosition;

      const std::array<float,3> bField = {{0,0,-5}};

      // Track parameters are in local coordinate system - propagate to pad row of the cluster
      trackObject.propagateParamTo(posLoc.X(), bField);

      LocalPosition3D trackLoc(trackObject.getX(), trackObject.getY(), trackObject.getZ());
      GlobalPosition3D trackGlob = Mapper::LocalToGlobal(trackLoc, cru.sector());

      const float resY = trackLoc.Y() - posLoc.Y();
      const float resZ = trackLoc.Y() - posLoc.Z();

      hResY->Fill(resY);
      hResZ->Fill(resZ);

      if(cru.side() == Side::A) {
        grClustersTrackA->SetPoint(clusCounterTrackA, digiX, digiY);
        grClustersTrackAxz->SetPoint(clusCounterTrackA++, digiX, digiZ);
        grTrackA->SetPoint(trackCounterA++, trackGlob.X(), trackGlob.Y());
      }
      if(cru.side() == Side::C) {
        grClustersTrackC->SetPoint(clusCounterTrackC, digiX, digiY);
        grClustersTrackCxz->SetPoint(clusCounterTrackC++, digiX, digiZ);
        grTrackC->SetPoint(trackCounterC++, trackGlob.X(), trackGlob.Y());
      }
      grTrackxz->SetPoint(trackCounterXZ++, trackGlob.X(), trackGlob.Z());
    }
  }

  // Drawing
  TCanvas *CDigits = new TCanvas("CDigits", "Compare Digits - Hits on A & C side", 1200, 600);
  CDigits->Divide(2,1);
  CDigits->cd(1);
  grHitsA->Draw("ap");
  grHitsA->GetXaxis()->SetLimits(-250, 250);
  grHitsA->SetMinimum(-250);
  grHitsA->SetMaximum(250);
  grDigitsA->Draw("p");
  grClustersA->Draw("p");
  grClustersTrackA->Draw("p");
  CDigits->cd(2);
  grHitsC->Draw("ap");
  grHitsC->GetXaxis()->SetLimits(-250, 250);
  grHitsC->SetMinimum(-250);
  grHitsC->SetMaximum(250);
  grDigitsC->Draw("p");
  grClustersC->Draw("p");
  grClustersTrackC->Draw("p");

  TCanvas *CDigitsXZ = new TCanvas("CDigitsXZ", "Compare Digits - Hits on A & C side", 600, 600);
  grHitsAxz->Draw("ap");
  grHitsAxz->GetXaxis()->SetLimits(-250, 250);
  grHitsAxz->SetMinimum(-250);
  grHitsAxz->SetMaximum(250);
  grDigitsAxz->Draw("p");
  grClustersAxz->Draw("p");
  grClustersTrackAxz->Draw("p");
  grHitsCxz->Draw("p");
  grHitsCxz->GetXaxis()->SetLimits(-250, 250);
  grHitsCxz->SetMinimum(-250);
  grHitsCxz->SetMaximum(250);
  grDigitsCxz->Draw("p");
  grClustersCxz->Draw("p");
  grClustersTrackCxz->Draw("p");

  TCanvas *CTracks = new TCanvas("CTracks", "Compare Tracks - Cluster on A & C side", 1200, 600);
  CTracks->Divide(2,1);
  CTracks->cd(1);
  grClustersTrackA->Draw("ap");
  grClustersTrackA->GetXaxis()->SetLimits(-250, 250);
  grClustersTrackA->SetMinimum(-250);
  grClustersTrackA->SetMaximum(250);
  grTrackA->Draw("p");
  drawSectorBoundaries();
  CTracks->cd(2);
  grClustersTrackC->Draw("ap");
  grClustersTrackC->GetXaxis()->SetLimits(-250, 250);
  grClustersTrackC->SetMinimum(-250);
  grClustersTrackC->SetMaximum(250);
  grTrackC->Draw("p");
  drawSectorBoundaries();

  TCanvas *CTracksXZ = new TCanvas("CTracksXZ", "Compare Tracks - Clusters on A & C side", 600, 600);
  grClustersAxz->Draw("ap");
  grClustersAxz->GetXaxis()->SetLimits(-250, 250);
  grClustersAxz->SetMinimum(-250);
  grClustersAxz->SetMaximum(250);
  grClustersCxz->Draw("p");
  grTrackxz->Draw("p");
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
