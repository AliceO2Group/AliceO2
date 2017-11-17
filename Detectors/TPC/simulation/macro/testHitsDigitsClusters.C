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
#include "TPCBase/Digit.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCReconstruction/TrackTPC.h"
#include "DetectorsBase/Track.h"
#include "TPCReconstruction/Cluster.h"
#include "TPCBase/Mapper.h"
#endif

using namespace o2::TPC;

void drawSectorBoundaries();

void testHitsDigitsClusters(int iEv=0,
               std::string simFile="",
               std::string digiFile="",
               std::string clusFile="",
               std::string trackFile="")
{
  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(0.5);
  gStyle->SetTitleSize(24);

  // ===| process the hits |====================================================
  TFile *hitFile   = TFile::Open(simFile.data());
  TTree *hitTree = (TTree *)gDirectory->Get("o2sim");

  std::vector<o2::TPC::HitGroup> *sectorHitsArray[Sector::MAXSECTOR];
  for (int s=0;s<Sector::MAXSECTOR;++s){
    sectorHitsArray[s] = nullptr;
    std::stringstream sectornamestr;
    sectornamestr << "TPCHitsSector" << s;
    hitTree->SetBranchAddress(sectornamestr.str().c_str(), &sectorHitsArray[s]);
  }

  TGraph *grHitsA = new TGraph();
  grHitsA->SetTitle(Form("Hits - Cluster comparison A-Side Event %d;x (cm);y (cm)", iEv));
  grHitsA->SetMarkerColor(kBlue+2);

  TGraph *grHitsC = new TGraph();
  grHitsC->SetTitle(Form("Hits - Cluster comparison C-Side Event %d;x (cm);y (cm)", iEv));
  grHitsC->SetMarkerColor(kBlue+2);

  TGraph *grHitsAzr = new TGraph();
  grHitsAzr->SetTitle(Form("Hits - Cluster comparison A-Side Event %d;z (cm);r (cm)", iEv));
  grHitsAzr->SetMarkerColor(kBlue+2);

  TGraph *grHitsCzr = new TGraph();
  grHitsCzr->SetTitle(Form("Hits - Cluster comparison C-Side Event %d;z (cm);r (cm)", iEv));
  grHitsCzr->SetMarkerColor(kBlue+2);


  int hitCounterA = 0;
  int hitCounterC = 0;
  hitTree->GetEntry(iEv);
  for (auto hits : sectorHitsArray) { // loop over sectors
    for(auto& inputgroup : *hits) {
      const int MCTrackID = inputgroup.GetTrackID();
      for(size_t hitindex = 0; hitindex < inputgroup.getSize(); ++hitindex){
        const auto& eh = inputgroup.getHit(hitindex);

        // A side
        if(eh.GetZ() > 0 ) {
          grHitsA->SetPoint(hitCounterA, eh.GetX(), eh.GetY());
          grHitsAzr->SetPoint(hitCounterA++, eh.GetZ(), TMath::Sqrt(eh.GetX()*eh.GetX() + eh.GetY()*eh.GetY()));
        }
        // C side
        if(eh.GetZ() < 0 ) {
          grHitsC->SetPoint(hitCounterC, eh.GetX(), eh.GetY());
          grHitsCzr->SetPoint(hitCounterC++, eh.GetZ(), TMath::Sqrt(eh.GetX()*eh.GetX() + eh.GetY()*eh.GetY()));
        }
      }
    }
  }

  // ===| process the digits |==================================================
  TFile *digitFile = TFile::Open(digiFile.data());
  TTree *digitTree = (TTree *)gDirectory->Get("o2sim");

  std::vector<o2::TPC::Digit> *digitsArray = nullptr;
  digitTree->SetBranchAddress("TPCDigit", &digitsArray);

  const Mapper& mapper = Mapper::instance();

  TGraph *grDigitsA = new TGraph();
  grDigitsA->SetMarkerColor(kGreen+2);

  TGraph *grDigitsC = new TGraph();
  grDigitsC->SetMarkerColor(kGreen+2);

  TGraph *grDigitsAzr = new TGraph();
  grDigitsAzr->SetMarkerColor(kGreen+2);

  TGraph *grDigitsCzr = new TGraph();
  grDigitsCzr->SetMarkerColor(kGreen+2);

  int digiCounterA = 0;
  int digiCounterC = 0;
  digitTree->GetEntry(iEv);
  for(auto& digit : *digitsArray) {
    const CRU cru(digit.getCRU());

    const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
    const int rowInSector       = digit.getRow() + region.getGlobalRowOffset();
    const GlobalPadNumber pad   = mapper.globalPadNumber(PadPos(rowInSector, digit.getPad()));
    const PadCentre& padCentre  = mapper.padCentre(pad);
    const float localYfactor    = (cru.side()==Side::A)?-1.f:1.f;
          float zPosition       = Digitizer::getZfromTimeBin(digit.getTimeStamp(), cru.side());

    LocalPosition3D posLoc(padCentre.X(), localYfactor*padCentre.Y(), zPosition);
    GlobalPosition3D posGlob = Mapper::LocalToGlobal(posLoc, cru.sector());

    const float digiX = posGlob.X();
    const float digiY = posGlob.Y();
    const float digiZ = zPosition;

    if(cru.side() == Side::A) {
      grDigitsA->SetPoint(digiCounterA, digiX, digiY);
      grDigitsAzr->SetPoint(digiCounterA++, digiZ, TMath::Sqrt(digiX*digiX + digiY*digiY));
    }
    if(cru.side() == Side::C) {
      grDigitsC->SetPoint(digiCounterC, digiX, digiY);
      grDigitsCzr->SetPoint(digiCounterC++, digiZ, TMath::Sqrt(digiX*digiX + digiY*digiY));
      grDigitsCzr->SetPoint(digiCounterC++, digiX, digiZ);
    }
  }

  // ===| process the clusters |================================================
  TFile *clusterFile = TFile::Open(clusFile.data());
  TTree *clusterTree = (TTree *)gDirectory->Get("o2sim");

  std::vector<o2::TPC::Cluster>  *clustersArray = nullptr;
  clusterTree->SetBranchAddress("TPCClusterHW", &clustersArray);

  TGraph *grClustersA = new TGraph();
  grClustersA->SetMarkerColor(kRed+2);
  grClustersA->SetMarkerSize(1);

  TGraph *grClustersC = new TGraph();
  grClustersC->SetMarkerColor(kRed+2);
  grClustersC->SetMarkerSize(1);

  TGraph *grClustersAzr = new TGraph();
  grClustersAzr->SetMarkerColor(kRed+2);
  grClustersAzr->SetMarkerSize(1);

  TGraph *grClustersCzr = new TGraph();
  grClustersCzr->SetMarkerColor(kRed+2);
  grClustersCzr->SetMarkerSize(1);

  int clusCounterA = 0;
  int clusCounterC = 0;
  clusterTree->GetEntry(iEv);
  for(auto& cluster: *clustersArray) {
    const CRU cru(cluster.getCRU());

    const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
    const int rowInSector       = cluster.getRow() + region.getGlobalRowOffset();
    const GlobalPadNumber pad   = mapper.globalPadNumber(PadPos(rowInSector, cluster.getPadMean()));
    const PadCentre& padCentre  = mapper.padCentre(pad);
    const float localYfactor    = (cru.side()==Side::A)?-1.f:1.f;
          float zPosition       = Digitizer::getZfromTimeBin(cluster.getTimeMean(), cru.side());

    LocalPosition3D posLoc(padCentre.X(), localYfactor*padCentre.Y(), zPosition);
    GlobalPosition3D posGlob = Mapper::LocalToGlobal(posLoc, cru.sector());

    const float clusterX = posGlob.X();
    const float clusterY = posGlob.Y();
    const float clusterZ = zPosition;

    if(cru.side() == Side::A) {
      grClustersA->SetPoint(clusCounterA, clusterX, clusterY);
      grClustersAzr->SetPoint(clusCounterA++, clusterZ, TMath::Sqrt(clusterX*clusterX + clusterY*clusterY));
    }
    if(cru.side() == Side::C) {
      grClustersC->SetPoint(clusCounterC, clusterX, clusterY);
      grClustersCzr->SetPoint(clusCounterC++, clusterZ, TMath::Sqrt(clusterX*clusterX + clusterY*clusterY));
    }
  }

  // ===| process tracks |======================================================
  auto tracks = TFile::Open(trackFile.data());
  auto trackTree = (TTree *)gDirectory->Get("events");

  // cluster graphs
  TGraph *grClustersTrackA = new TGraph();
  grClustersTrackA->SetMarkerColor(kOrange+2);
  grClustersTrackA->SetMarkerSize(1);

  TGraph *grClustersTrackC = new TGraph();
  grClustersTrackC->SetMarkerColor(kOrange+2);
  grClustersTrackC->SetMarkerSize(1);

  TGraph *grClustersTrackAzr = new TGraph();
  grClustersTrackAzr->SetMarkerColor(kOrange+2);
  grClustersTrackAzr->SetMarkerSize(1);

  TGraph *grClustersTrackCzr = new TGraph();
  grClustersTrackCzr->SetMarkerColor(kOrange+2);
  grClustersTrackCzr->SetMarkerSize(1);

  // Track graphs
  TGraph *grTrackA = new TGraph();
  grTrackA->SetTitle(Form("Track - Cluster comparison A-Side Event %d;x (cm);y (cm)", iEv));

  TGraph *grTrackC = new TGraph();
  grTrackC->SetTitle(Form("Track - Cluster comparison C-Side Event %d;x (cm);y (cm)", iEv));

  TGraph *grTrackzr = new TGraph();
  grTrackzr->SetTitle(Form("Track - Cluster comparison Event %d;z (cm);x (cm)", iEv));

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
        grClustersTrackAzr->SetPoint(clusCounterTrackA++, digiZ, TMath::Sqrt(digiX*digiX + digiY*digiY));
        grTrackA->SetPoint(trackCounterA++, trackGlob.X(), trackGlob.Y());
      }
      if(cru.side() == Side::C) {
        grClustersTrackC->SetPoint(clusCounterTrackC, digiX, digiY);
        grClustersTrackCzr->SetPoint(clusCounterTrackC++, digiZ, TMath::Sqrt(digiX*digiX + digiY*digiY));
        grTrackC->SetPoint(trackCounterC++, trackGlob.X(), trackGlob.Y());
      }
      grTrackzr->SetPoint(trackCounterXZ++, trackGlob.Z(), TMath::Sqrt(trackGlob.X()*trackGlob.X() + trackGlob.Y()*trackGlob.Y()));
    }
  }

  // ===| Drawing |=============================================================
  auto CDigits = new TCanvas("CDigits", "Compare Digits - Hits on A & C side", 1200, 600);
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

  auto CDigitsXZ = new TCanvas("CDigitsXZ", "Compare Digits - Hits on A & C side", 600, 600);
  grHitsAzr->Draw("ap");
  grHitsAzr->GetXaxis()->SetLimits(-250, 250);
  grHitsAzr->SetMinimum(-250);
  grHitsAzr->SetMaximum(250);
  grDigitsAzr->Draw("p");
  grClustersAzr->Draw("p");
  grClustersTrackAzr->Draw("p");
  grHitsCzr->Draw("p");
  grHitsCzr->GetXaxis()->SetLimits(-250, 250);
  grHitsCzr->SetMinimum(-250);
  grHitsCzr->SetMaximum(250);
  grDigitsCzr->Draw("p");
  grClustersCzr->Draw("p");
  grClustersTrackCzr->Draw("p");

  auto CTracks = new TCanvas("CTracks", "Compare Tracks - Cluster on A & C side", 1200, 600);
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

  auto CTracksXZ = new TCanvas("CTracksXZ", "Compare Tracks - Clusters on A & C side", 600, 600);
  grClustersAzr->Draw("ap");
  grClustersAzr->GetXaxis()->SetLimits(-250, 250);
  grClustersAzr->SetMinimum(-250);
  grClustersAzr->SetMaximum(250);
  grClustersCzr->Draw("p");
  grTrackzr->Draw("p");
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
