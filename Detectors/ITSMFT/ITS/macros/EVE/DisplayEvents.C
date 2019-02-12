/// \file DisplayEvents.C
/// \brief Simple macro to display ITS digits, clusters and tracks

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include <array>
#include <algorithm>

#include <TFile.h>
#include <TTree.h>
#include <TEveManager.h>
#include <TEveBrowser.h>
#include <TGTab.h>
#include <TGLCameraOverlay.h>
#include <TEveFrameBox.h>
#include <TEveQuadSet.h>
#include <TEveTrans.h>
#include <TEvePointSet.h>
#include <TEveTrackPropagator.h>
#include <TEveTrack.h>

#include "EventVisualisationView/MultiView.h"

#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTBase/Digit.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITS/TrackITS.h"
#endif

using namespace o2::ITSMFT;

extern TEveManager* gEve;

static TTree* gDigiTree = nullptr;
static TTree* gClusTree = nullptr;
static TTree* gTracTree = nullptr;
static Int_t gEntry = -1, gChipID = -1;

static TEveElementList* gEvent = nullptr;
static TEveElementList* gDigits = nullptr;

void displayEvent(int entry = 0, int chip = 0)
{
  if ((entry < 0) || (entry >= gDigiTree->GetEntries())) {
    std::cerr << "Out of event range ! " << entry << '\n';
    return;
  }
  std::cout << "\n*** Event #" << entry << " ***\n";

  // Digits
  std::vector<Digit> digi, *digiArr = &digi;
  gDigiTree->SetBranchAddress("ITSDigit", &digiArr);
  gDigiTree->GetEvent(entry);
  std::cout << "Number of digits: " << digi.size() << '\n';

  auto gman = o2::ITS::GeometryTGeo::Instance();
  std::vector<int> occup(gman->getNumberOfChips());
  
  const float sizey = SegmentationAlpide::ActiveMatrixSizeRows;
  const float sizex = SegmentationAlpide::ActiveMatrixSizeCols;
  const float dy = SegmentationAlpide::PitchRow;
  const float dx = SegmentationAlpide::PitchCol;

  static TEveFrameBox *box = new TEveFrameBox();
  box->SetAAQuadXY(0, 0, 0, sizex, sizey);
  box->SetFrameColor(kGray);

  std::string cname("ALPIDE chip #");
  cname += std::to_string(chip);
  TEveQuadSet* q = new TEveQuadSet(cname.c_str());
  q->SetOwnIds(kTRUE);
  q->SetFrame(box);
  q->Reset(TEveQuadSet::kQT_RectangleXY, kFALSE, 32);

  for (const auto &d : digi)
  {
    auto id=d.getChipIndex();
    occup[id]++;
    if (id != chip) continue;

    int row = d.getRow();
    int col = d.getColumn();
    int charge=d.getCharge();
    q->AddQuad(col*dx,row*dy,0.,dx,dy);
    q->QuadValue(charge);
  }   
  q->RefitPlex();

  TEveTrans& t = q->RefMainTrans();
  t.RotateLF(1, 3, 0.5*TMath::Pi());
  t.SetPos(0, 0, 0);

  auto most = std::distance(occup.begin(), std::max_element(occup.begin(), occup.end()));
  std::cout<<"Most occupied chip: " << most << " ("<<occup[most]<<" digits)\n";
  

  // Clusters
  std::vector<Cluster> clus, *clusArr = &clus;
  gClusTree->SetBranchAddress("ITSCluster", &clusArr);
  gClusTree->GetEvent(entry);
  std::cout << "Number of clusters: " << clus.size() << '\n';

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
      Cluster& c = (*clusArr)[idx];
      const auto& gloC = c.getXYZGloRot(*gman);
      tpoints->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
    }
    track->AddElement(tpoints);

    tracks->AddElement(track);
  }
  tracks->MakeTracks();

  // Event
  std::string ename("Event #");
  ename += std::to_string(entry);
  delete gEvent;
  gEvent = new TEveElementList(ename.c_str());
  gEvent->AddElement(points);
  gEvent->AddElement(tracks);
  auto multi = o2::EventVisualisation::MultiView::getInstance();
  multi->registerEvent(gEvent);

  delete gDigits;
  gDigits = new TEveElementList(ename.c_str());
  gDigits->AddElement(q);
  gEve->AddElement(gDigits);

  gEve->Redraw3D(kFALSE);
}

void load(int i = 0, int c = 0)
{
  gEntry = i;
  gChipID = c;
  displayEvent(gEntry, gChipID);
}

void init(int entry = 0, int chip = 13, 
          std::string digifile = "itsdigits.root",
          std::string clusfile = "o2clus_its.root",
          std::string tracfile = "o2trac_its.root",
          std::string inputGeom = "O2geometry.root")
{
  TEveManager::Create(kTRUE,"V");
  TEveBrowser *browser = gEve->GetBrowser();

  // Chip View
  browser->GetTabRight()->SetText("Chip View");
  TGLViewer* v = gEve->GetDefaultGLViewer();
  v->SetCurrentCamera(TGLViewer::kCameraOrthoZOY);
  TGLCameraOverlay* co = v->GetCameraOverlay();
  co->SetShowOrthographic(kTRUE);
  co->SetOrthographicMode(TGLCameraOverlay::kGridFront);

  // Event View
  auto multi = o2::EventVisualisation::MultiView::getInstance();
  multi->drawGeometryForDetector("ITS");

  o2::Base::GeometryManager::loadGeometry(inputGeom, "FAIRGeom");
  auto gman = o2::ITS::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G));

  TFile::Open(digifile.data());
  gDigiTree = (TTree*)gFile->Get("o2sim");
  TFile::Open(clusfile.data());
  gClusTree = (TTree*)gFile->Get("o2sim");
  TFile::Open(tracfile.data());
  gTracTree = (TTree*)gFile->Get("o2sim");

  std::cout<<"\n **** Navigation over events and chips ****\n";
  std::cout<<" load(event, chip) \t jump to the specified event and chip\n";
  std::cout<<" next() \t\t load next event \n";
  std::cout<<" prev() \t\t load previous event \n";
  std::cout<<" loadChip(chip) \t jump to the specified chip within the current event \n";
  std::cout<<" nextChip() \t\t load the next chip within the current event \n";
  std::cout<<" prevChip() \t\t load the previous chip within the current event \n";
  
  load(entry, chip);
  gEve->Redraw3D(kTRUE);
}

void next()
{
  gEntry++;
  displayEvent(gEntry, gChipID);
}

void prev()
{
  gEntry--;
  displayEvent(gEntry, gChipID);
}

void loadChip(int chip) {
  gChipID = chip;
  load(gEntry, gChipID);
}

void nextChip()
{
  gChipID++;
  load(gEntry, gChipID);
}

void prevChip()
{
  gChipID--;
  load(gEntry, gChipID);
}

void DisplayEvents()
{
  // A dummy function with the same name as this macro
}
