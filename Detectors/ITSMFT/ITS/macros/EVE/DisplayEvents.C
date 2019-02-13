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
  static TEveElementList* gChip;
  static TEveElement *getEveChipDigits(int chip, const std::vector<Digit> &digi);
  static TEveElement *getEveChipClusters(int chip, const std::vector<Cluster> &clus);
  static TEveElement *getEveClusters(const std::vector<Cluster> &clus);
  static TEveElement *getEveTracks(const std::vector<Cluster> &clus, const std::vector<o2::ITS::TrackITS> &trks);
};

TTree* Data::gDigiTree = nullptr;
TTree* Data::gClusTree = nullptr;
TTree* Data::gTracTree = nullptr;
TEveElementList* Data::gEvent = nullptr;
TEveElementList* Data::gChip = nullptr;

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

constexpr float sizey = SegmentationAlpide::ActiveMatrixSizeRows;
constexpr float sizex = SegmentationAlpide::ActiveMatrixSizeCols;
constexpr float dy = SegmentationAlpide::PitchRow;
constexpr float dx = SegmentationAlpide::PitchCol;

TEveElement *Data::getEveChipDigits(int chip, const std::vector<Digit> &digi) {
  static TEveFrameBox *box = new TEveFrameBox();
  box->SetAAQuadXY(0, 0, 0, sizex, sizey);
  box->SetFrameColor(kGray);

  // Digits
  TEveQuadSet* qdigi = new TEveQuadSet("digits");
  qdigi->SetOwnIds(kTRUE);
  qdigi->SetFrame(box);
  qdigi->Reset(TEveQuadSet::kQT_RectangleXY, kFALSE, 32);
  auto gman = o2::ITS::GeometryTGeo::Instance();
  std::vector<int> occup(gman->getNumberOfChips());
  for (const auto &d : digi)
  {
    auto id=d.getChipIndex();
    occup[id]++;
    if (id != chip) continue;

    int row = d.getRow();
    int col = d.getColumn();
    int charge=d.getCharge();
    qdigi->AddQuad(col*dx,row*dy,0.,dx,dy);
    qdigi->QuadValue(charge);
  }   
  qdigi->RefitPlex();
  TEveTrans& t = qdigi->RefMainTrans();
  t.RotateLF(1, 3, 0.5*TMath::Pi());
  t.SetPos(0, 0, 0);

  auto most = std::distance(occup.begin(), std::max_element(occup.begin(), occup.end()));
  std::cout<<"Most occupied chip: " << most << " ("<<occup[most]<<" digits)\n";
  
  return qdigi;
}

TEveElement *Data::getEveChipClusters(int chip, const std::vector<Cluster> &clus) {
  static TEveFrameBox *box = new TEveFrameBox();
  box->SetAAQuadXY(0, 0, 0, sizex, sizey);
  box->SetFrameColor(kGray);

  // Clusters
  TEveQuadSet* qclus = new TEveQuadSet("clusters");
  qclus->SetOwnIds(kTRUE);
  qclus->SetFrame(box);
  qclus->Reset(TEveQuadSet::kQT_LineXYFixedZ, kFALSE, 32);
  for (const auto &c : clus)
  {
    auto id=c.getSensorID();
    if (id != chip) continue;

    int row = c.getPatternRowMin();
    int col = c.getPatternColMin();
    int len = c.getPatternColSpan();
    int wid = c.getPatternRowSpan();
    qclus->AddLine(col*dx,row*dy,len*dx,0.);
    qclus->AddLine(col*dx,row*dy,0., wid*dy);
    qclus->AddLine((col+len)*dx,row*dy,0.,wid*dy);
    qclus->AddLine(col*dx,(row+wid)*dy,len*dx, 0.);
  }   
  qclus->RefitPlex();
  TEveTrans& ct = qclus->RefMainTrans();
  ct.RotateLF(1, 3, 0.5*TMath::Pi());
  ct.SetPos(0, 0, 0);

  return qclus;
}

TEveElement *Data::getEveClusters(const std::vector<Cluster> &clus) {
  auto gman = o2::ITS::GeometryTGeo::Instance();
  TEvePointSet* clusters = new TEvePointSet("clusters");
  clusters->SetMarkerColor(kBlue);
  for (const auto& c : clus) {
    const auto& gloC = c.getXYZGloRot(*gman);
    clusters->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
  }
  return clusters;
}

TEveElement *Data::getEveTracks(const std::vector<Cluster> &clus, const std::vector<o2::ITS::TrackITS> &trks) {
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

  // Chip display
  auto chipDigits = getEveChipDigits(chip, mDigits);
  auto chipClusters = getEveChipClusters(chip, mClusters);
  delete gChip;
  std::string cname(ename+"  ALPIDE chip #");
  cname += std::to_string(chip);  
  gChip = new TEveElementList(cname.c_str());
  gChip->AddElement(chipDigits);
  gChip->AddElement(chipClusters);
  gEve->AddElement(gChip);

  // Event display
  auto clusters = getEveClusters(mClusters);
  auto tracks = getEveTracks(mClusters, mTracks);
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
