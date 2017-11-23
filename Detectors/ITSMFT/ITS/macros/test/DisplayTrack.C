/// \file DisplayTrack.C
/// \brief Simple macro to display ITSU tracks

#if !defined(__CLING__) || defined(__ROOTCLING__)
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
  #include <TMath.h>
  #include <TString.h>

  #include "SimulationDataFormat/MCCompLabel.h"
  #include "SimulationDataFormat/MCTruthContainer.h"
  #include "ITSMFTSimulation/Hit.h"
  #include "DetectorsBase/Utils.h"
  #include "MathUtils/Cartesian3D.h"
  #include "ITSBase/GeometryTGeo.h"
  #include "ITSMFTReconstruction/Cluster.h"
  #include "ITSReconstruction/CookedTrack.h"
#endif

void DisplayTrack(Int_t nEvents = 10, TString mcEngine = "TGeant3", Int_t event=0, Int_t track=0) {
  using namespace o2::Base;
  using namespace o2::ITS;
  
  using o2::ITSMFT::Hit;
  using o2::ITSMFT::Cluster;

  char filename[100];
  TFile *f=nullptr;

  if (gEve == nullptr) {
     TEveManager::Create();
  }

  // Load geometry
  if (gGeoManager == nullptr) {
     sprintf(filename, "AliceO2_%s.params_%i.root", mcEngine.Data(), nEvents);
     f = TFile::Open(filename);
     f->Get("FairGeoParSet");
     f->Close();
  }
  
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
  TTree *tree = (TTree *)gDirectory->Get("o2sim");

  string s{"event"};
  s+=std::to_string(event);
  s+="_hits";
  s+=std::to_string(track);
  TEvePointSet* points = new TEvePointSet(s.data());
  points->SetMarkerColor(kBlue);

  std::vector<Hit>* hitArr = nullptr;
  tree->SetBranchAddress("ITSHit", &hitArr);

  tree->GetEvent(event);

  Int_t nc=hitArr->size(), n=0;
  while(nc--) {
    Hit& c=(*hitArr)[nc];
    if (c.GetTrackID() == track) {
      points->SetNextPoint(c.GetX(),c.GetY(),c.GetZ());
      n++;
    }
  }
  cout<<"Number of hits: "<<n<<endl;

  gEve->AddElement(points,0);
  f->Close();


  // Clusters
  sprintf(filename, "AliceO2_%s.clus_%i_event.root", mcEngine.Data(), nEvents);
  f = TFile::Open(filename);
  tree = (TTree *)gDirectory->Get("o2sim");

  s="event";
  s+=std::to_string(event);
  s+="_clusters";
  s+=std::to_string(track);
  points = new TEvePointSet(s.data());
  points->SetMarkerColor(kMagenta);
  
  std::vector<Cluster> *clusArr=nullptr;
  //tree->SetBranchAddress("ITSCluster",&clusArr); // Why this does not work ???
  auto *branch = tree->GetBranch("ITSCluster");
  if (!branch) {
    std::cout<<"No clusters !"<<std::endl;
    return;
  }
  branch->SetAddress(&clusArr);
  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> *clsLabArr=nullptr;
  tree->SetBranchAddress("ITSClusterMCTruth",&clsLabArr);

  int tf=0;
  int lastTF=tree->GetEntries();
  for (; tf<lastTF; ++tf) {
     tree->GetEvent(tf);
     int nc=clusArr->size();
     for (int i=0; i<nc; i++) { // Find the TF containing this MC event
	 auto mclab = (clsLabArr->getLabels(i))[0];
	 auto id = mclab.getEventID();
	 if (id == event) goto found;
     }
  }
  std::cout<<"Time Frame containing the MC event "<<event<<" was not found"<<std::endl;

found:
  std::cout<<"MC event "<<event<<" found in the Time Frame #"<<tf<<std::endl;
  o2::ITS::GeometryTGeo *gman = GeometryTGeo::Instance();
  gman->fillMatrixCache( Utils::bit2Mask(TransformType::T2GRot) ); // request cached transforms

  nc=clusArr->size(); n=0;
  while(nc--) {
      Cluster &c=(*clusArr)[nc];
      auto lab=(clsLabArr->getLabels(nc))[0];
      auto gloC = c.getXYZGloRot(*gman); // convert from tracking to global frame
      if (lab.getEventID() != event) continue;
      if (lab.getTrackID() == track) {
         points->SetNextPoint(gloC.X(),gloC.Y(),gloC.Z());
         n++;
      }      
  } 
  cout<<"Number of clusters: "<<n<<endl;

  gEve->AddElement(points,0);
  f->Close();
  
  // Track
  sprintf(filename, "AliceO2_%s.trac_%i_event.root", mcEngine.Data(), nEvents);
  f = TFile::Open(filename);
  tree = (TTree *)gDirectory->Get("o2sim");

  s="event";
  s+=std::to_string(event);
  s+="_track";
  s+=std::to_string(track);
  points = new TEvePointSet(s.data());
  points->SetMarkerColor(kGreen);
  
  std::vector<CookedTrack> *trkArr=nullptr;
  tree->SetBranchAddress("ITSTrack",&trkArr);
  // Track MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> *trkLabArr=nullptr;
  tree->SetBranchAddress("ITSTrackMCTruth",&trkLabArr);

  tree->GetEvent(tf);

  Int_t nt=trkArr->size(); n=0;
  while(nt--) {
      const CookedTrack &t=(*trkArr)[nt];
      auto lab=(trkLabArr->getLabels(nt))[0];
      if (TMath::Abs(lab.getEventID()) != event) continue;
      if (TMath::Abs(lab.getTrackID()) != track) continue;
      Int_t nc=t.getNumberOfClusters();
      while (n<nc) {
	Int_t idx=t.getClusterIndex(n);
        Cluster &c=(*clusArr)[idx];
	auto gloC = c.getXYZGloRot(*gman); // convert from tracking to global frame
        points->SetNextPoint(gloC.X(),gloC.Y(),gloC.Z());
        n++;
      }
      break;
  } 
  cout<<"Number of attached clusters: "<<n<<endl;

  gEve->AddElement(points,0);
  f->Close();

}

