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

  #include "ITSMFTBase/SegmentationPixel.h"
  #include "ITSMFTBase/Digit.h"
  #include "ITSMFTSimulation/Point.h"
  #include "ITSBase/GeometryTGeo.h"
#endif

void CheckDigits(Int_t nEvents = 10, TString mcEngine = "TGeant3") {
  using o2::ITSMFT::SegmentationPixel;
  using o2::ITSMFT::Digit;
  using o2::ITSMFT::Point;
  using namespace o2::ITS;

  TFile *f=TFile::Open("CheckDigits.root","recreate");
  TNtuple *nt=new TNtuple("ntd","digit ntuple","x:y:z:dx:dz");

  char filename[100];

  // Geometry
  sprintf(filename, "AliceO2_%s.params_%i.root", mcEngine.Data(), nEvents);
  TFile *file = TFile::Open(filename);
  gFile->Get("FairGeoParSet");
  GeometryTGeo *gman = new GeometryTGeo(kTRUE);
  SegmentationPixel *seg =
    (SegmentationPixel*)gman->getSegmentationById(0);

  // Hits
  sprintf(filename, "AliceO2_%s.mc_%i_event.root", mcEngine.Data(), nEvents);
  TFile *file0 = TFile::Open(filename);
  TTree *hitTree=(TTree*)gFile->Get("cbmsim");
  TClonesArray hitArr("o2::ITSMFT::Point"), *phitArr(&hitArr);
  hitTree->SetBranchAddress("ITSPoint",&phitArr);

  // Digits
  sprintf(filename, "AliceO2_%s.digi_%i_event.root", mcEngine.Data(), nEvents);
  TFile *file1 = TFile::Open(filename);
  TTree *digTree=(TTree*)gFile->Get("cbmsim");
  TClonesArray digArr("o2::ITSMFT::Digit"), *pdigArr(&digArr);
  digTree->SetBranchAddress("ITSDigit",&pdigArr);
  
  Int_t nev=hitTree->GetEntries();
  while (nev--) {
    hitTree->GetEvent(nev);
    Int_t nh=hitArr.GetEntriesFast();
    digTree->GetEvent(nev);
    Int_t nd=digArr.GetEntriesFast();
    while(nd--) {
      Digit *d=(Digit *)digArr.UncheckedAt(nd);
      Int_t ix=d->getRow(), iz=d->getColumn();
      Float_t x,z; 
      seg->detectorToLocal(ix,iz,x,z);
      const Double_t loc[3]={x,0.,z};
      
      Int_t chipID=d->getChipIndex();
      Int_t lab=d->getLabel(0);

      Double_t glo[3]={0., 0., 0.}, dx=0., dz=0.;
      gman->localToGlobal(chipID,loc,glo);

      for (Int_t i=0; i<nh; i++) {
        Point *p=(Point *)hitArr.UncheckedAt(i);
	if (p->GetDetectorID() != chipID) continue; 
	if (p->GetTrackID() != lab) continue;
        Double_t x=0.5*(p->GetX() + p->GetStartX());
        Double_t y=0.5*(p->GetY() + p->GetStartY());
        Double_t z=0.5*(p->GetZ() + p->GetStartZ());
        Double_t g[3]={x, y, z}, l[3];
	gman->globalToLocal(chipID,g,l);
        dx=l[0]-loc[0]; dz=l[2]-loc[2];
	break;
      }
      
      nt->Fill(glo[0],glo[1],glo[2],dx,dz);

    }
  }
  new TCanvas; nt->Draw("y:x");
  new TCanvas; nt->Draw("dx:dz");
  f->Write();
  f->Close();
}
