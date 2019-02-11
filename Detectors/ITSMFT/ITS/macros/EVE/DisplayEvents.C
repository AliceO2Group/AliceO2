/// \file DisplayEvents.C
/// \brief Simple macro to display ITS clusters and tracks

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include <array>

#include <TFile.h>
#include <TTree.h>
#include <TEveManager.h>
#include <TEvePointSet.h>
#include <TEveTrackPropagator.h>
#include <TEveTrack.h>

#include "EventVisualisationView/MultiView.h"

#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITS/TrackITS.h"
#endif

extern TEveManager* gEve;

static TTree* gClusTree = nullptr;
static TTree* gTracTree = nullptr;

static TEveElementList* gEvent = nullptr;

void displayEvent(int entry = 0)
{
  if ((entry < 0) || (entry >= gClusTree->GetEntries())) {
    std::cerr << "Out of event range ! " << entry << '\n';
    return;
  }
  std::cout << "\n*** Event #" << entry << " ***\n";

  // Clusters
  std::vector<o2::ITSMFT::Cluster> clus, *clusArr = &clus;
  gClusTree->SetBranchAddress("ITSCluster", &clusArr);
  gClusTree->GetEvent(entry);
  std::cout << "Number of clusters: " << clus.size() << '\n';

  auto gman = o2::ITS::GeometryTGeo::Instance();
  TEvePointSet* points = new TEvePointSet("clusters");
  points->SetMarkerColor(kBlue);
  for (const auto& c : clus) {
    const auto& gloC = c.getXYZGloRot(*gman);
    points->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
  }

  // Tracks
  std::vector<o2::ITS::TrackITS> trks, *tracArr = &trks;
  gTracTree->SetBranchAddress("ITSTrack", &tracArr);
  gTracTree->GetEvent(entry);
  std::cout << "Number of tracks: " << trks.size() << "\n\n";

  TEveTrackList* tracks = new TEveTrackList("tracks");
  auto prop = tracks->GetPropagator();
  prop->SetMagField(0.5);
  prop->SetMaxR(50.);
  for (const auto& rec : trks) {
    std::array<float, 3> p;
    rec.getPxPyPzGlo(p);
    TEveRecTrackD t;
    t.fP = { p[0], p[1], p[2] };
    t.fSign = (rec.getSign() < 0) ? -1 : 1;
    TEveTrack* track = new TEveTrack(&t, prop);
    track->SetLineColor(kMagenta);

    TEvePointSet* tpoints = new TEvePointSet("tclusters");
    tpoints->SetMarkerColor(kGreen);
    int nc = rec.getNumberOfClusters();
    while (nc--) {
      Int_t idx = rec.getClusterIndex(nc);
      o2::ITSMFT::Cluster& c = (*clusArr)[idx];
      const auto& gloC = c.getXYZGloRot(*gman);
      tpoints->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
    }
    track->AddElement(tpoints);

    tracks->AddElement(track);
  }
  tracks->MakeTracks();

  // Event
  std::string ename("Event");
  ename += std::to_string(entry);
  delete gEvent;
  gEvent = new TEveElementList(ename.c_str());
  gEvent->AddElement(points);
  gEvent->AddElement(tracks);

  gEve->AddElement(gEvent);
  auto multi = o2::EventVisualisation::MultiView::getInstance();
  multi->registerEvent(gEvent);

  gEve->Redraw3D();
}

Int_t nev = -1;

void load(int i = 0)
{
  nev = i;
  displayEvent(nev);
}

void init(int i = 0,
          std::string clusfile = "o2clus_its.root",
          std::string tracfile = "o2trac_its.root",
          std::string inputGeom = "O2geometry.root")
{
  TEveManager::Create();

  auto multi = o2::EventVisualisation::MultiView::getInstance();
  multi->drawGeometryForDetector("ITS");

  o2::Base::GeometryManager::loadGeometry(inputGeom, "FAIRGeom");
  auto gman = o2::ITS::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G));

  TFile::Open(clusfile.data());
  gClusTree = (TTree*)gFile->Get("o2sim");
  TFile::Open(tracfile.data());
  gTracTree = (TTree*)gFile->Get("o2sim");

  load(i);
}

void next()
{
  nev++;
  displayEvent(nev);
}

void prev()
{
  nev--;
  displayEvent(nev);
}

void DisplayEvents()
{
  // A dummy function with the same name as this macro
}
