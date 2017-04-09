/// \file CheckDigits.C
/// \brief Simple macro to check ITSU clusters

#if !defined(__CINT__) || defined(__MAKECINT__)
  #include <TFile.h>
  #include <TTree.h>
  #include <TClonesArray.h>
  #include <TH2F.h>
  #include <TNtuple.h>
  #include <TCanvas.h>
  #include <TString.h>

  #include "ITSMFTSimulation/Point.h"
  #include "ITSBase/GeometryTGeo.h"
  #include "ITSReconstruction/Cluster.h"
#endif


void CheckClusters(Int_t nEvents = 10, TString mcEngine = "TGeant3") {
  using o2::ITSMFT::Point;
  using namespace o2::ITS;

  TFile *f=TFile::Open("CheckClusters.root","recreate");
  TNtuple *nt=new TNtuple("ntc","cluster ntuple","x:y:z:dx:dz");

  char filename[100];

  // Geometry
  sprintf(filename, "AliceO2_%s.params_%i.root", mcEngine.Data(), nEvents);
  TFile *file = TFile::Open(filename);
  gFile->Get("FairGeoParSet");
  GeometryTGeo *gman = new GeometryTGeo(kTRUE);

  Cluster::setGeom(gman);
  
  // Hits
  sprintf(filename, "AliceO2_%s.mc_%i_event.root", mcEngine.Data(), nEvents);
  TFile *file0 = TFile::Open(filename);
  TTree *hitTree=(TTree*)gFile->Get("cbmsim");
  TClonesArray hitArr("o2::ITSMFT::Point"), *phitArr(&hitArr);
  hitTree->SetBranchAddress("ITSPoint",&phitArr);

  // Clusters
  sprintf(filename, "AliceO2_%s.clus_%i_event.root", mcEngine.Data(), nEvents);
  TFile *file1 = TFile::Open(filename);
  TTree *clusTree=(TTree*)gFile->Get("cbmsim");
  TClonesArray clusArr("o2::ITS::Cluster"), *pclusArr(&clusArr);
  clusTree->SetBranchAddress("ITSCluster",&pclusArr);
  
  Int_t nev=hitTree->GetEntries();
  while (nev--) {
    hitTree->GetEvent(nev);
    Int_t nh=hitArr.GetEntriesFast();
    clusTree->GetEvent(nev);
    Int_t nc=clusArr.GetEntriesFast();
    while(nc--) {
      Cluster *c=static_cast<Cluster *>(clusArr.UncheckedAt(nc));
      c->goToFrameLoc();
      const Double_t loc[3]={c->getX(), 0., c->getZ()};
      
      Int_t chipID=c->getVolumeId();
      Int_t lab=c->getLabel(0);

      Double_t glo[3]={0., 0., 0.}, dx=0., dz=0.;
      gman->localToGlobal(chipID,loc,glo);

      for (Int_t i=0; i<nh; i++) {
        Point *p=static_cast<Point *>(hitArr.UncheckedAt(i));
	if (p->GetDetectorID() != chipID) continue; 
	if (p->GetTrackID() != lab) continue;
        Double_t x=0.5*(p->GetX() + p->GetStartX());
        Double_t y=0.5*(p->GetY() + p->GetStartY());
        Double_t z=0.5*(p->GetZ() + p->GetStartZ());
        Double_t g[3]={x, y, z}, l[3];
	gman->globalToLocal(chipID,g,l);
        dx=l[0]-loc[0]; dz=l[2]-loc[2];
        dx=loc[0]; dz=loc[2];
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
