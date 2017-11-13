/// \file CheckDigits.C
/// \brief Simple macro to check MFT clusters

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TFile.h>
#include <TTree.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TCanvas.h>
#include <TString.h>

#include "ITSMFTSimulation/Hit.h"
#include "DetectorsBase/Utils.h"
#include "MathUtils/Cartesian3D.h"
#include "MFTBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#endif

using namespace o2::Base;
using o2::ITSMFT::Cluster;

void CheckClusters_mft(Int_t nEvents = 1, Int_t nMuons = 10, TString mcEngine = "TGeant3") 
{

  using namespace o2::Base;
  using namespace o2::MFT;

  using o2::ITSMFT::Hit;

  TH1F *hTrackID = new TH1F("hTrackID","hTrackID",1.1*nMuons+1,-0.5,(nMuons+0.1*nMuons)+0.5);
  TH2F *hDifLocXrZc = new TH2F("hDifLocXrZc","hDifLocXrZc",100,-50.,+50.,100,-50.,+50.);

  TFile *f = TFile::Open("CheckClusters.root","recreate");
  TNtuple *nt = new TNtuple("ntc","cluster ntuple","x:y:z:dx:dz:lab:rof:ev:hlx:hlz:clx:clz");

  Char_t filename[100];

  // Geometry
  sprintf(filename, "AliceO2_%s.params_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TFile *file = TFile::Open(filename);
  gFile->Get("FairGeoParSet");
  
  auto gman = o2::MFT::GeometryTGeo::Instance();
  gman->fillMatrixCache( Utils::bit2Mask(TransformType::T2L, TransformType::T2G, TransformType::L2G) ); // request cached transforms
  
  // Hits
  sprintf(filename, "AliceO2_%s.mc_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TFile *file0 = TFile::Open(filename);
  TTree *hitTree = (TTree*)gFile->Get("o2sim");
  std::vector<Hit> *hitArray = nullptr;
  hitTree->SetBranchAddress("MFTHit",&hitArray);

  // Clusters
  sprintf(filename, "AliceO2_%s.clus_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TFile *file1 = TFile::Open(filename);
  TTree *clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<Cluster> *clusArray = nullptr;
  //clusTree->SetBranchAddress("MFTCluster",&clusArray);
  auto *branch = clusTree->GetBranch("MFTCluster");
  if (!branch) {
    std::cout << "No clusters !" << std::endl;
    return;
  }
  branch->SetAddress(&clusArray);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> *clusLabArray = nullptr;
  clusTree->SetBranchAddress("MFTClusterMCTruth",&clusLabArray);
 
  Int_t nevCl = clusTree->GetEntries(); // clusters in cont. readout may be grouped as few events per entry
  Int_t nevH = hitTree->GetEntries(); // hits are stored as one event per entry
  Int_t ievC = 0, ievH = 0;
  Int_t lastReadHitEv = -1;

  Int_t nNoise = 0;

  for (ievC = 0; ievC < nevCl; ievC++) {
    clusTree->GetEvent(ievC);
    Int_t nc = clusArray->size();

    while (nc--) {

      // cluster is in tracking coordinates always
      Cluster &c=(*clusArray)[nc];

      Int_t chipID = c.getSensorID();      
      const auto locC = c.getXYZLoc(*gman); // convert from tracking to local frame
      const auto gloC = c.getXYZGlo(*gman); // convert from tracking to global frame
      auto lab = (clusLabArray->getLabels(nc))[0];

      Float_t dx = 0, dz = 0;
      Int_t trID = lab.getTrackID();
      Int_t ievH = lab.getEventID();
      
      Point3D<Float_t> locH,locHsta;

      if (trID >= 0) { // is this cluster from hit or noise ?  

	Hit* p = nullptr;
	if (lastReadHitEv != ievH) {
	  hitTree->GetEvent(ievH);
	  lastReadHitEv = ievH;
	}

        for (auto& ptmp : *hitArray) {
	  if (ptmp.GetDetectorID() != (Short_t)chipID) continue; 
	  if (ptmp.GetTrackID() != (Int_t)trID) continue;
	  hTrackID->Fill((Float_t)ptmp.GetTrackID());
	  p = &ptmp;
	  break;
	} // hits

	if (!p) {
	  printf("did not find hit (scanned HitEvs %d %d) for cluster of tr%d on chip %d\n",ievH,nevH,trID,chipID);
	  locH.SetXYZ(0.f,0.f,0.f);
	} else {
	  // mean local position of the hit
	  locH    = gman->getMatrixL2G(chipID)^( p->GetPos() );  // inverse conversion from global to local
	  locHsta = gman->getMatrixL2G(chipID)^( p->GetPosStart() );
	  locH.SetXYZ( 0.5*(locH.X()+locHsta.X()),0.5*(locH.Y()+locHsta.Y()),0.5*(locH.Z()+locHsta.Z()) );
	  //std::cout << "chip "<< p->qGetDetectorID() << "  PposGlo " << p->GetPos() << std::endl;
	  //std::cout << "chip "<< c.getSensorID() << "  PposLoc " << locH << std::endl;
	  dx = locH.X()-locC.X();
	  dz = locH.Z()-locC.Z();
	  hDifLocXrZc->Fill(1.e4*dx,1.e4*dz);
	}

      } else {
	nNoise++;
      } // not noise

      nt->Fill(gloC.X(),gloC.Y(),gloC.Z(), dx, dz, trID, c.getROFrame(), ievC,
	       locH.X(),locH.Z(), locC.X(),locC.Z());

    } //clusters

  } // events

  printf("nt has %lld entries\n",nt->GetEntriesFast());

  TCanvas *c1 = new TCanvas("c1","hTrackID",50,50,600,600);
  hTrackID->Scale(1./(Float_t)nEvents);
  hTrackID->SetMinimum(0.);
  hTrackID->DrawCopy();

  TCanvas *c2 = new TCanvas("c2","hDifLocXrZc",50,50,600,600);
  hDifLocXrZc->DrawCopy("COL2");

  new TCanvas; nt->Draw("y:x");
  new TCanvas; nt->Draw("dx:dz");
  f->cd();
  nt->Write();
  hTrackID->Write();
  f->Close();

  printf("noise clusters %d \n",nNoise);

}
