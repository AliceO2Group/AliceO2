/// \file CheckDigits.C
/// \brief Simple macro to check ITSU digits

#if (!defined(__CINT__) && !defined(__CLING__)) || defined(__MAKECINT__)
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
  #include "ITSBase/GeometryTGeo.h"
#endif

using namespace o2::Base;

void CheckDigits(Int_t nEvents = 10, TString mcEngine = "TGeant3") {
  using o2::ITSMFT::SegmentationPixel;
  using o2::ITSMFT::Digit;
  using o2::ITSMFT::Hit;
  using namespace o2::ITS;

  TFile *f=TFile::Open("CheckDigits.root","recreate");
  TNtuple *nt=new TNtuple("ntd","digit ntuple","x:y:z:dx:dz");

  char filename[100];

  // Geometry
  sprintf(filename, "AliceO2_%s.params_%i.root", mcEngine.Data(), nEvents);
  TFile *file = TFile::Open(filename);
  gFile->Get("FairGeoParSet");
  
  auto *gman = o2::ITS::GeometryTGeo::Instance();
  gman->fillMatrixCache( Utils::bit2Mask(TransformType::L2G) );
  
  SegmentationPixel *seg = (SegmentationPixel*)gman->getSegmentationById(0);

  // Hits
  sprintf(filename, "AliceO2_%s.mc_%i_event.root", mcEngine.Data(), nEvents);
  TFile *file0 = TFile::Open(filename);
  TTree *hitTree=(TTree*)gFile->Get("o2sim");
  TClonesArray hitArr("o2::ITSMFT::Hit"), *phitArr(&hitArr);
  hitTree->SetBranchAddress("ITSHit",&phitArr);

  // Digits
  sprintf(filename, "AliceO2_%s.digi_%i_event.root", mcEngine.Data(), nEvents);
  TFile *file1 = TFile::Open(filename);
  TTree *digTree=(TTree*)gFile->Get("o2sim");
  TClonesArray digArr("o2::ITSMFT::Digit"), *pdigArr(&digArr);
  digTree->SetBranchAddress("ITSDigit",&pdigArr);
  
  int nevD = digTree->GetEntries(); // digits in cont. readout may be grouped as few events per entry
  int nevH = hitTree->GetEntries(); // hits are stored as one event per entry
  int lastReadHitEv = -1;
  
  for (int iev = 0;iev<nevD; iev++) {

    digTree->GetEvent(iev);
    Int_t nd=digArr.GetEntriesFast();

    while(nd--) {
      Digit *d=(Digit *)digArr.UncheckedAt(nd);
      Int_t ix=d->getRow(), iz=d->getColumn();
      Float_t x,z; 
      seg->detectorToLocal(ix,iz,x,z);
      const Point3D<float> locD(x,0.,z);
      
      Int_t chipID=d->getChipIndex();
      o2::MCCompLabel lab = d->getLabel(0);
      int trID = lab.getTrackID();
      int ievH = lab.getEventID();

      if (trID>=0) { // not a noise
	const auto gloD = gman->getMatrixL2G(chipID)(locD); // convert to global
	float dx=0., dz=0.;
	
	if (lastReadHitEv!=ievH) {
	  hitTree->GetEvent(ievH);
	  lastReadHitEv = ievH;
	}
	Int_t nh=hitArr.GetEntriesFast();

	for (Int_t i=0; i<nh; i++) {
	  Hit *p=(Hit *)hitArr.UncheckedAt(i);
	  if (p->GetDetectorID() != chipID) continue; 
	  if (p->GetTrackID() != lab) continue;
	  auto locH    = gman->getMatrixL2G(chipID)^( p->GetPos() );  // inverse conversion from global to local
	  auto locHsta = gman->getMatrixL2G(chipID)^( p->GetPosStart() );
	  locH.SetXYZ( 0.5*(locH.X()+locHsta.X()),0.5*(locH.Y()+locHsta.Y()),0.5*(locH.Z()+locHsta.Z()) );
	  //
	  nt->Fill(gloD.X(),gloD.Y(),gloD.Z(),locH.X()-locD.X(),locH.Z()-locD.Z());
	  break;
	}      
      }
    }
  }
  new TCanvas; nt->Draw("y:x");
  new TCanvas; nt->Draw("dx:dz");
  f->Write();
  f->Close();
}
