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

static Int_t gEntry = -1, gChipID = -1;

class Data {
public:
  void loadData(int entry);
  void displayData(int entry, int chip);
  static void setDigiTree(TTree *t) {gDigiTree=t;}
  static void setClusTree(TTree *t) {gClusTree=t;}
  static void setTracTree(TTree *t) {gTracTree=t;}
  
private:
  // Data loading members 
  static TTree* gDigiTree;
  static TTree* gClusTree;
  static TTree* gTracTree;
  std::vector<Digit> mDigits;
  std::vector<Cluster> mClusters;
  std::vector<o2::ITS::TrackITS> mTracks;
  template<typename T>
  static void load(TTree *tree, const char *name, int entry, std::vector<T> *arr);

  // TEve-related members
  static TEveElementList* gEvent;
  static TEveElementList* gDigits;
  static TEveElement *getEveDigits(int entry, int chip, const std::vector<Digit> &digi);
  static TEveElement *getEveClusters(int entry, const std::vector<Cluster> &clus);
  static TEveElement *getEveTracks(int entry, const std::vector<Cluster> &clus, const std::vector<o2::ITS::TrackITS> &trks);
};

TTree* Data::gDigiTree = nullptr;
TTree* Data::gClusTree = nullptr;
TTree* Data::gTracTree = nullptr;
TEveElementList* Data::gEvent = nullptr;
TEveElementList* Data::gDigits = nullptr;

template<typename T>
void Data::load(TTree *tree, const char *name, int entry, std::vector<T> *arr) {
  if (tree == nullptr) {
    std::cerr << "No tree for " << name << '\n';
    return;
  }
  if ((entry < 0) || (entry >= tree->GetEntries())) {
    std::cerr << name <<": Out of event range ! " << entry << '\n';
    return;
  }
  tree->SetBranchAddress(name, &arr);
  tree->GetEvent(entry);
  std::cout << "Number of "<< name <<"s: " << arr->size() << '\n';
}

void Data::loadData(int entry) {
  load<Digit>(gDigiTree, "ITSDigit", entry, &mDigits);
  load<Cluster>(gClusTree,"ITSCluster", entry, &mClusters);
  load<o2::ITS::TrackITS>(gTracTree, "ITSTrack", entry, &mTracks);
}


// Dealing with graphics
TEveElement *Data::getEveDigits(int entry, int chip, const std::vector<Digit> &digi) {
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
  
  return q;
}

TEveElement *Data::getEveClusters(int entry, const std::vector<Cluster> &clus) {
  auto gman = o2::ITS::GeometryTGeo::Instance();
  TEvePointSet* clusters = new TEvePointSet("clusters");
  clusters->SetMarkerColor(kBlue);
  for (const auto& c : clus) {
    const auto& gloC = c.getXYZGloRot(*gman);
    clusters->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
  }
  return clusters;
}

TEveElement *Data::getEveTracks(int entry, const std::vector<Cluster> &clus, const std::vector<o2::ITS::TrackITS> &trks) {
  auto gman = o2::ITS::GeometryTGeo::Instance();
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
    tracks->AddElement(track);

    if (clus.empty()) continue;
    TEvePointSet* tpoints = new TEvePointSet("tclusters");
    tpoints->SetMarkerColor(kGreen);
    int nc = rec.getNumberOfClusters();
    while (nc--) {
      Int_t idx = rec.getClusterIndex(nc);
      const Cluster& c = clus[idx];
      const auto& gloC = c.getXYZGloRot(*gman);
      tpoints->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
    }
    track->AddElement(tpoints);
  }
  tracks->MakeTracks();

  return tracks;
}

void Data::displayData(int entry, int chip) {
  std::string ename("Event #");
  ename += std::to_string(entry);

  auto digits = getEveDigits(entry, chip, mDigits);
  delete gDigits;
  gDigits = new TEveElementList(ename.c_str());
  gDigits->AddElement(digits);
  gEve->AddElement(gDigits);
  
  auto clusters = getEveClusters(entry, mClusters);
  auto tracks = getEveTracks(entry, mClusters, mTracks);
  delete gEvent;
  gEvent = new TEveElementList(ename.c_str());
  gEvent->AddElement(clusters);
  gEvent->AddElement(tracks);
  auto multi = o2::EventVisualisation::MultiView::getInstance();
  multi->registerEvent(gEvent);

  gEve->Redraw3D(kFALSE);
}

void load(int entry=0, int chip=13)
{
  gEntry = entry;
  gChipID = chip;

  Data data;
  std::cout << "\n*** Event #" << entry << " ***\n";

  data.loadData(entry);
  data.displayData(entry, chip);
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

  auto file = TFile::Open(digifile.data());
  if (file && gFile->IsOpen())
    Data::setDigiTree((TTree*)gFile->Get("o2sim"));
  else
    std::cerr << "Cannot open file: " << digifile << '\n';
  
  file = TFile::Open(clusfile.data());
  if (file && gFile->IsOpen())
    Data::setClusTree((TTree*)gFile->Get("o2sim"));
  else
    std::cerr << "Cannot open file: " << clusfile << '\n';
  
  file = TFile::Open(tracfile.data());
  if (file && gFile->IsOpen())
    Data::setTracTree((TTree*)gFile->Get("o2sim"));
  else
    std::cerr << "Cannot open file: " << tracfile << '\n';

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
  load(gEntry, gChipID);
}

void prev()
{
  gEntry--;
  load(gEntry, gChipID);
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
