#include <vector>
#include <fstream>
#include <iostream>

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/DigitMC.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCReconstruction/TrackTPC.h"
#include "DetectorsBase/Track.h"
#include "TPCSimulation/Cluster.h"
#include "TPCBase/Mapper.h"

using namespace o2::TPC;

void testHitsDigitsClusters(int iEv=0,
               std::string simFile="~/AliSoftware/sw/BUILD/O2-latest-O2dir/O2/bin/AliceO2_TGeant3.tpc.mc_10_event.root",
               std::string digiFile="~/AliSoftware/sw/BUILD/O2-latest-O2dir/O2/bin/AliceO2_TGeant3.tpc.digi_10_event.root",
               std::string clusFile="~/AliSoftware/sw/BUILD/O2-latest-O2dir/O2/bin/AliceO2_TGeant3.tpc.clusters_10_event.root",
               std::string trackFile="~/AliSoftware/sw/BUILD/O2-latest-O2dir/O2/bin/tracks.root")
{
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
  TGraph *grHitsC = new TGraph();
  grHitsC->SetTitle("C side ; x [cm]; y [cm]");
  TGraph *grHitsAxz = new TGraph();
  grHitsAxz->SetTitle("; x [cm]; z [cm]");
  TGraph *grHitsCxz = new TGraph();
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

    LocalPosition3D posLoc(padCentre.getX(), localYfactor*padCentre.getY(), zPosition);
    GlobalPosition3D posGlob = Mapper::LocalToGlobal(posLoc, cru.sector());

    const float digiX = posGlob.getX();
    const float digiY = posGlob.getY();
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

    LocalPosition3D posLoc(padCentre.getX(), localYfactor*padCentre.getY(), zPosition);
    GlobalPosition3D posGlob = Mapper::LocalToGlobal(posLoc, cru.sector());

    const float digiX = posGlob.getX();
    const float digiY = posGlob.getY();
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

  // clusters from tracks
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


  std::vector<TrackTPC> *arrTracks = 0;
  trackTree->SetBranchAddress("Tracks", &arrTracks);

  int clusCounterTrackA = 0;
  int clusCounterTrackC = 0;

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

      LocalPosition3D posLoc(padCentre.getX(), localYfactor*padCentre.getY(), zPosition);
      GlobalPosition3D posGlob = Mapper::LocalToGlobal(posLoc, cru.sector());

      const float digiX = posGlob.getX();
      const float digiY = posGlob.getY();
      const float digiZ = zPosition;

      if(cru.side() == Side::A) {
        grClustersTrackA->SetPoint(clusCounterTrackA, digiX, digiY);
        grClustersTrackAxz->SetPoint(clusCounterTrackA++, digiX, digiZ);
      }
      if(cru.side() == Side::C) {
        grClustersTrackC->SetPoint(clusCounterTrackC, digiX, digiY);
        grClustersTrackCxz->SetPoint(clusCounterTrackC++, digiX, digiZ);
      }
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
}
