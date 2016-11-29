/// \file CheckDigits.C
/// \brief Simple macro to check ITSU clusters

#if !defined(__CINT__) || defined(__MAKECINT__)
  #include <TFile.h>
  #include <TTree.h>
  #include <TClonesArray.h>
  #include <TH2F.h>
  #include <TTree.h>
  #include <TNtuple.h>
  #include <TGeoManager.h>
  #include <TCanvas.h>

  #include "ITSBase/GeometryTGeo.h"
  #include "ITSSimulation/Point.h"
  #include "ITSReconstruction/Cluster.h"
#endif


void CheckClusters() {
  using namespace AliceO2::ITS;

  TFile *f=TFile::Open("cluster_xyz.root","recreate");
  TNtuple *nt=new TNtuple("nt","my ntuple","x:y:z:dx:dz");

  TGeoManager::Import("geofile_full.root");
  GeometryTGeo *gman = new GeometryTGeo(kTRUE);

  Cluster::SetGeom(gman);
  
  // Hits
  TFile *file0 = TFile::Open("AliceO2_TGeant3.mc_10_event.root");
  TTree *hitTree=(TTree*)gFile->Get("cbmsim");
  TClonesArray hitArr("AliceO2::ITS::Point"), *phitArr(&hitArr);
  hitTree->SetBranchAddress("ITSPoint",&phitArr);

  // Clusters
  TFile *file1 = TFile::Open("AliceO2_TGeant3.clus_10_event.root");
  TTree *clusTree=(TTree*)gFile->Get("cbmsim");
  TClonesArray clusArr("AliceO2::ITS::Cluster"), *pclusArr(&clusArr);
  clusTree->SetBranchAddress("ITSCluster",&pclusArr);
  
  Int_t nev=hitTree->GetEntries();
  while (nev--) {
    hitTree->GetEvent(nev);
    Int_t nh=hitArr.GetEntriesFast();
    clusTree->GetEvent(nev);
    Int_t nc=clusArr.GetEntriesFast();
    while(nc--) {
      Cluster *c=static_cast<Cluster *>(clusArr.UncheckedAt(nc));
      c->GoToFrameLoc();
      const Double_t loc[3]={c->GetX(), 0., c->GetZ()};
      
      Int_t chipID=c->GetVolumeId();
      Int_t lab=c->GetLabel(0);

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
}
