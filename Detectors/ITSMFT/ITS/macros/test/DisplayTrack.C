/// \file DisplayTrack.C
/// \brief Simple macro to display ITSU tracks

#if !defined(__CINT__) || defined(__MAKECINT__)
  #include <string>

  #include <TFile.h>
  #include <TTree.h>
  #include <TGeoManager.h>
  #include <TGLViewer.h>
  #include <TEveManager.h>
  #include <TEveGeoShapeExtract.h>
  #include <TEveGeoShape.h>
  #include <TEveGeoNode.h>
  #include <TEvePointSet.h>
  #include <TClonesArray.h>
  #include <TMath.h>
  #include <TString.h>

  #include "ITSMFTSimulation/Point.h"
  #include "ITSReconstruction/Cluster.h"
  #include "ITSReconstruction/CookedTrack.h"
#endif

extern TGeoManager *gGeoManager;

void DisplayTrack(Int_t nEvents = 10, TString mcEngine = "TGeant3", Int_t event=0, Int_t track=0) {
  using o2::ITSMFT::Point;
  using namespace o2::ITS;

  char filename[100];

  TEveManager::Create();

  // Full geometry
  sprintf(filename, "AliceO2_%s.params_%i.root", mcEngine.Data(), nEvents);
  TFile *f = TFile::Open(filename);
  f->Get("FairGeoParSet");
  f->Close();
  
  gGeoManager->GetVolume("obSuppCyl")->SetInvisible();
  gGeoManager->GetVolume("ibSuppCyl")->SetInvisible();
  gGeoManager->GetVolume("ITSUStave0_StaveStruct")->SetInvisible();
  gGeoManager->GetVolume("ITSUStave1_StaveStruct")->SetInvisible();
  gGeoManager->GetVolume("ITSUStave2_StaveStruct")->SetInvisible();

  gGeoManager->GetVolume("ITSUHalfStave0")->SetTransparency(50);
  gGeoManager->GetVolume("ITSUHalfStave1")->SetTransparency(50);
  gGeoManager->GetVolume("ITSUHalfStave2")->SetTransparency(50);
  gGeoManager->GetVolume("ITSUHalfStave3")->SetTransparency(50);
  gGeoManager->GetVolume("ITSUHalfStave4")->SetTransparency(50);
  gGeoManager->GetVolume("ITSUHalfStave5")->SetTransparency(50);
  gGeoManager->GetVolume("ITSUHalfStave6")->SetTransparency(50);
  
  TGeoNode* tnode = gGeoManager->GetTopVolume()->FindNode("ITSV_2");
  TEveGeoTopNode *evenode = new TEveGeoTopNode(gGeoManager, tnode); 
  evenode->SetVisLevel(4);
  gEve->AddGlobalElement(evenode);
  
  TGLViewer *view=gEve->GetDefaultGLViewer();
  Double_t center[3]{0};
  view->CurrentCamera().Reset();
  view->CurrentCamera().Configure(3., 1200., center, 0., 89*3.14/180);
  
  gEve->Redraw3D();
  
  /*
  // Simplified geometry
  f = TFile::Open("simple_geom_ITS.root");
  TEveGeoShapeExtract* gse = (TEveGeoShapeExtract*) f->Get("ITS");
  TEveGeoShape* gsre = TEveGeoShape::ImportShapeExtract(gse);
  gEve->AddElement(gsre,0);
  f->Close();
  */
  
  // Hits
  sprintf(filename, "AliceO2_%s.mc_%i_event.root", mcEngine.Data(), nEvents);
  f = TFile::Open(filename);
  TTree *tree = (TTree *)gDirectory->Get("cbmsim");

  string s{"hits"};
  s+=std::to_string(track);
  TEvePointSet* points = new TEvePointSet(s.data());
  points->SetMarkerColor(kBlue);

  TClonesArray pntArr("o2::ITSMFT::Point"), *ppntArr(&pntArr);
  tree->SetBranchAddress("ITSPoint",&ppntArr);

  tree->GetEvent(event);

  Int_t nc=pntArr.GetEntriesFast(), n=0;
  while(nc--) {
      Point *c=static_cast<Point *>(pntArr.UncheckedAt(nc));
      if (c->GetTrackID() == track) {
         points->SetNextPoint(c->GetX(),c->GetY(),c->GetZ());
         n++;
      }      
  } 
  cout<<"Number of points: "<<n<<endl;

  gEve->AddElement(points,0);
  f->Close();


  // Clusters
  sprintf(filename, "AliceO2_%s.clus_%i_event.root", mcEngine.Data(), nEvents);
  f = TFile::Open(filename);
  tree = (TTree *)gDirectory->Get("cbmsim");

  s="clusters";
  s+=std::to_string(track);
  points = new TEvePointSet(s.data());
  points->SetMarkerColor(kMagenta);
  
  TClonesArray clusArr("o2::ITS::Cluster"), *pclusArr(&clusArr);
  tree->SetBranchAddress("ITSCluster",&pclusArr);

  tree->GetEvent(event);

  GeometryTGeo *gman = new GeometryTGeo(kTRUE);
  Cluster::setGeom(gman);  

  nc=clusArr.GetEntriesFast(); n=0;
  while(nc--) {
      Cluster *c=static_cast<Cluster *>(clusArr.UncheckedAt(nc));
      c->goToFrameGlo();
      if (c->getLabel(0) == track) {
         points->SetNextPoint(c->getX(),c->getY(),c->getZ());
         n++;
      }      
  } 
  cout<<"Number of clusters: "<<n<<endl;

  gEve->AddElement(points,0);
  f->Close();


  
  // Track
  sprintf(filename, "AliceO2_%s.trac_%i_event.root", mcEngine.Data(), nEvents);
  f = TFile::Open(filename);
  tree = (TTree *)gDirectory->Get("cbmsim");

  s="track";
  s+=std::to_string(track);
  points = new TEvePointSet(s.data());
  points->SetMarkerColor(kGreen);
  
  TClonesArray trkArr("o2::ITS::CookedTrack"), *ptrkArr(&trkArr);
  tree->SetBranchAddress("ITSTrack",&ptrkArr);

  tree->GetEvent(event);

  Int_t nt=trkArr.GetEntriesFast(); n=0;
  while(nt--) {
      CookedTrack *t=static_cast<CookedTrack *>(trkArr.UncheckedAt(nt));
      if (TMath::Abs(t->getLabel()) != track) continue;
      Int_t nc=t->getNumberOfClusters();
      while (n<nc) {
	Int_t idx=t->getClusterIndex(n);
        Cluster *c=static_cast<Cluster *>(clusArr.UncheckedAt(idx));
        c->goToFrameGlo();
        points->SetNextPoint(c->getX(),c->getY(),c->getZ());
        n++;
      }      
      break;
  } 
  cout<<"Number of attached clusters: "<<n<<endl;

  gEve->AddElement(points,0);
  f->Close();
  
}

