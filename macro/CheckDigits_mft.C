/// \file CheckDigits.C
/// \brief Simple macro to check ITSU digits

#if !defined(__CINT__) || defined(__MAKECINT__)

#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TCanvas.h>
#include <TString.h>

#include "DetectorsBase/Utils.h"
#include "ITSMFTBase/SegmentationPixel.h"
#include "ITSMFTBase/Digit.h"
#include "ITSMFTSimulation/Hit.h"
#include "MFTBase/GeometryTGeo.h"

#endif

using namespace o2::Base;

void CheckDigits(Int_t nEvents = 1, Int_t nMuons = 10, TString mcEngine = "TGeant3") 
{

  using o2::ITSMFT::SegmentationPixel;
  using o2::ITSMFT::Digit;
  using o2::ITSMFT::Hit;
  using namespace o2::MFT;
  
  TH1F *hTrackID = new TH1F("hTrackID","hTrackID",1.1*nMuons+1,-0.5,(nMuons+0.1*nMuons)+0.5);

  TFile *f = TFile::Open("CheckDigits.root","recreate");
  TNtuple *nt = new TNtuple("ntd","digit ntuple","x:y:z:dx:dz");

  Char_t filename[100];

  // Geometry
  sprintf(filename, "AliceO2_%s.params_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TFile *file = TFile::Open(filename);
  gFile->Get("FairGeoParSet");
  
  auto *gman = o2::MFT::GeometryTGeo::Instance();
  gman->fillMatrixCache( Utils::bit2Mask(TransformType::L2G) );
  
  SegmentationPixel *seg = (SegmentationPixel*)gman->getSegmentationById(0);

  // Hits
  sprintf(filename, "AliceO2_%s.mc_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TFile *file0 = TFile::Open(filename);
  std::cout << " Open hits file " << filename << std::endl;
  TTree *hitTree = (TTree*)gFile->Get("o2sim");
  TClonesArray hitArr("o2::ITSMFT::Hit"), *phitArr(&hitArr);
  hitTree->SetBranchAddress("MFTHits",&phitArr);
  
  // Digits
  sprintf(filename, "AliceO2_%s.digi_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TFile *file1 = TFile::Open(filename);
  std::cout << " Open digits file " << filename << std::endl;
  TTree *digTree = (TTree*)gFile->Get("o2sim");
  TClonesArray digArr("o2::ITSMFT::Digit"), *pdigArr(&digArr);
  digTree->SetBranchAddress("MFTDigits",&pdigArr);
  
  Int_t nevD = digTree->GetEntries(); // digits in cont. readout may be grouped as few events per entry
  Int_t nevH = hitTree->GetEntries(); // hits are stored as one event per entry
  Int_t lastReadHitEv = -1;
  
  std::cout << "Found " << nevH << " events with hits " << std::endl;
  std::cout << "Found " << nevD << " events with digits " << std::endl;

  Int_t nNoise = 0;

  for (Int_t iev = 0; iev < nevD; iev++) {

    digTree->GetEvent(iev);
    Int_t nd = digArr.GetEntriesFast();

    while (nd--) {

      Digit *d = (Digit *)digArr.UncheckedAt(nd);
      Int_t ix = d->getRow(), iz=d->getColumn();
      Float_t x,z; 
      seg->detectorToLocal(ix,iz,x,z);
      const Point3D<Float_t> locD(x,0.,z);
      
      Int_t chipID = d->getChipIndex();
      o2::MCCompLabel lab = d->getLabel(0);
      Int_t trID = lab.getTrackID();
      Int_t ievH = lab.getEventID();

      if (trID >= 0) { // not a noise

	const auto gloD = gman->getMatrixL2G(chipID)(locD); // convert to global
	float dx=0., dz=0.;
	
	if (lastReadHitEv != ievH) {
	  hitTree->GetEvent(ievH);
	  lastReadHitEv = ievH;
	}

	Int_t nh = hitArr.GetEntriesFast();
	
	for (Int_t i = 0; i < nh; i++) {

	  Hit *p = (Hit *)hitArr.UncheckedAt(i);
	  if (p->GetDetectorID() != (Short_t)chipID) continue; 
	  if (p->GetTrackID() != (Int_t)lab) continue;
	  auto locH    = gman->getMatrixL2G(chipID)^( p->GetPos() );  // inverse conversion from global to local
	  auto locHsta = gman->getMatrixL2G(chipID)^( p->GetPosStart() ); // ITS specific only
	  locH.SetXYZ( 0.5*(locH.X()+locHsta.X()),0.5*(locH.Y()+locHsta.Y()),0.5*(locH.Z()+locHsta.Z()) );
	  //
	  nt->Fill(gloD.X(),gloD.Y(),gloD.Z(),locH.X()-locD.X(),locH.Z()-locD.Z());
	  hTrackID->Fill((Float_t)p->GetTrackID());
	  break;

	} // hits

      } else {
	nNoise++;
      } // not noise

    } // digits

  } // events

  printf("nt has %lld entries\n",nt->GetEntriesFast());

  TCanvas *c1 = new TCanvas("c1","hTrackID",50,50,600,600);
  hTrackID->Scale(1./(Float_t)nEvents);
  hTrackID->SetMinimum(0.);
  hTrackID->DrawCopy();

  new TCanvas; nt->Draw("y:x");
  new TCanvas; nt->Draw("dx:dz");
  f->Write();
  f->Close();

  printf("noise digits %d \n",nNoise);

}
