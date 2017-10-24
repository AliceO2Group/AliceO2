/// \file CheckDigits.C
/// \brief Simple macro to check ITSU digits

#if !defined(__CLING__) || defined(__ROOTCLING__)
  #include <TFile.h>
  #include <TTree.h>
  #include <TH2F.h>
  #include <TNtuple.h>
  #include <TCanvas.h>
  #include <TString.h>

  #include "DetectorsBase/Utils.h"
  #include "ITSMFTBase/SegmentationAlpide.h"
  #include "ITSMFTBase/Digit.h"
  #include "ITSMFTSimulation/Hit.h"
  #include "ITSBase/GeometryTGeo.h"
  #include <vector>
#endif

using namespace o2::Base;

void CheckDigits(Int_t nEvents = 10, TString mcEngine = "TGeant3") {
  using o2::ITSMFT::SegmentationAlpide;
  using o2::ITSMFT::Digit;
  using o2::ITSMFT::Hit;
  using namespace o2::ITS;

  TFile *f=TFile::Open("CheckDigits.root","recreate");
  TNtuple *nt=new TNtuple("ntd","digit ntuple","id:x:y:z:rowD:colD:rowH:colH:xlH:zlH:xlcH:zlcH:dx:dz");

  char filename[100];

  // Geometry
  sprintf(filename, "AliceO2_%s.params_%i.root", mcEngine.Data(), nEvents);
  TFile *file = TFile::Open(filename);
  gFile->Get("FairGeoParSet");
  
  auto *gman = o2::ITS::GeometryTGeo::Instance();
  gman->fillMatrixCache( Utils::bit2Mask(TransformType::L2G) );
  
  SegmentationAlpide seg;

  // Hits
  sprintf(filename, "AliceO2_%s.mc_%i_event.root", mcEngine.Data(), nEvents);
  TFile *file0 = TFile::Open(filename);
  TTree *hitTree=(TTree*)gFile->Get("o2sim");
  std::vector<o2::ITSMFT::Hit> *hitArray = nullptr;
  hitTree->SetBranchAddress("ITSHit", &hitArray);

  // Digits
  sprintf(filename, "AliceO2_%s.digi_%i_event.root", mcEngine.Data(), nEvents);
  TFile *file1 = TFile::Open(filename);
  TTree *digTree=(TTree*)gFile->Get("o2sim");
  std::vector<o2::ITSMFT::Digit> *digArr = nullptr;
  digTree->SetBranchAddress("ITSDigit",&digArr);
  
  int nevD = digTree->GetEntries(); // digits in cont. readout may be grouped as few events per entry
  int nevH = hitTree->GetEntries(); // hits are stored as one event per entry
  int lastReadHitEv = -1;

  int ndr=0,ndf=0;
  
  for (int iev = 0;iev<nevD; iev++) {

    digTree->GetEvent(iev);

    int nd=-1;
    for (const auto &d : *digArr) {
      nd++;
      Int_t ix=d.getRow(), iz=d.getColumn();
      Float_t x=0.f,z=0.f; 
      seg.detectorToLocal(ix,iz,x,z);
      const Point3D<float> locD(x,0.,z);
      
      Int_t chipID=d.getChipIndex();
      o2::MCCompLabel lab = d.getLabel(0);
      int trID = lab.getTrackID();
      int ievH = lab.getEventID();

      if (trID>=0) { // not a noise
	ndr++;
	const auto gloD = gman->getMatrixL2G(chipID)(locD); // convert to global
	float dx=0., dz=0.;
	
	if (lastReadHitEv!=ievH) {
	  hitTree->GetEvent(ievH);
	  lastReadHitEv = ievH;
	}
	bool ok = false;
	for (auto& p : *hitArray) {
	  if (p.GetDetectorID() != chipID) continue; 
	  if (p.GetTrackID() != trID) continue;
	  auto locH    = gman->getMatrixL2G(chipID)^( p.GetPos() );  // inverse conversion from global to local
	  auto locHsta = gman->getMatrixL2G(chipID)^( p.GetPosStart() );
	  locH.SetXYZ( 0.5*(locH.X()+locHsta.X()),0.5*(locH.Y()+locHsta.Y()),0.5*(locH.Z()+locHsta.Z()) );
	  int row,col;
	  float xlc,zlc;
	  seg.localToDetector(locH.X(),locH.Z(), row, col);
	  seg.detectorToLocal(row,col,xlc,zlc);
	  //
	  nt->Fill(chipID,gloD.X(),gloD.Y(),gloD.Z(),ix,iz,row,col,
		   locH.X(),locH.Z(), xlc,zlc,  locH.X()-locD.X(),locH.Z()-locD.Z());
	  ok = true;
	  ndf++;
	  break;
	}
	if (!ok) {
	  printf("did not find hit for digit %d in ev %d: MCEv:%d MCTrack %d\n",nd,iev,ievH,trID);
	}
      }
    }
  }
  new TCanvas; nt->Draw("y:x");
  new TCanvas; nt->Draw("dx:dz");
  f->Write();
  f->Close();
  printf("read %d filled %d\n",ndr,ndf);
}
