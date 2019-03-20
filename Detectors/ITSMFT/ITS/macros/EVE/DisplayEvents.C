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
#include <TGButton.h>
#include <TGNumberEntry.h>
#include <TGFrame.h>
#include <TGTab.h>
#include <TGLCameraOverlay.h>
#include <TEveFrameBox.h>
#include <TEveQuadSet.h>
#include <TEveTrans.h>
#include <TEvePointSet.h>
#include <TEveTrackPropagator.h>
#include <TEveTrack.h>

#include "EventVisualisationView/MultiView.h"

#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "ITSMFTReconstruction/RawPixelReader.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTBase/Digit.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITS/TrackITS.h"
#endif

using namespace o2::ITSMFT;

extern TEveManager* gEve;

static TGNumberEntry* gEntry;
static TGNumberEntry* gChipID;

class Data
{
 public:
  void loadData(int entry);
  void displayData(int entry, int chip);
  void setRawPixelReader(std::string input) {
    auto reader = new RawPixelReader<ChipMappingITS>();
    reader->openInput(input);
    gPixelReader=reader;
    gPixelReader->getNextChipData(mChipData);
  }
  void setDigitPixelReader(std::string input) {
    auto reader = new DigitPixelReader();
    reader->openInput(input, o2::detectors::DetID("ITS"));
    reader->init();
    reader->readNextEntry();
    gPixelReader=reader;
    gPixelReader->getNextChipData(mChipData);
  }
  void setDigiTree(TTree* t) { mDigiTree = t; }
  void setClusTree(TTree* t) { mClusTree = t; }
  void setTracTree(TTree* t) { mTracTree = t; }

 private:
  // Data loading members
  PixelReader* gPixelReader = nullptr;
  ChipPixelData mChipData;  
  std::vector<Digit> mDigits;
  std::vector<Cluster> mClusters;
  std::vector<o2::ITS::TrackITS> mTracks;
  void loadDigits(int entry);

  TTree* mDigiTree = nullptr;
  TTree* mClusTree = nullptr;
  TTree* mTracTree = nullptr;
  template <typename T>
  void load(TTree* tree, const char* name, int entry, std::vector<T>* arr);

  // TEve-related members
  TEveElementList* mEvent = nullptr;
  TEveElementList* mChip = nullptr;
  TEveElement* getEveChipDigits(int chip);
  TEveElement* getEveChipClusters(int chip);
  TEveElement* getEveClusters();
  TEveElement* getEveTracks();
} evdata;


void Data::loadDigits(int entry) {
  static auto ir_old = mChipData.getInteractionRecord();
  auto ir=mChipData.getInteractionRecord();
  std::cout<<"orbit/crossing: "<<' '<<ir.orbit<<'/'<<ir.bc<<'\n';
  
  mDigits.clear();

  do {
     auto chipID=mChipData.getChipID();
     auto pixels = mChipData.getData();
     for (auto &pixel : pixels) {
       auto col=pixel.getCol();
       auto row=pixel.getRow();
       mDigits.emplace_back(chipID,0,row,col);
     }
     mChipData.clear();
     gPixelReader->getNextChipData(mChipData);
     ir=mChipData.getInteractionRecord();
  } while (ir_old==ir);
  ir_old=ir;

  std::cout<<"Number of ITSDigits: "<<mDigits.size()<<'\n';
}

template <typename T>
void Data::load(TTree* tree, const char* name, int entry, std::vector<T>* arr)
{
  if (tree == nullptr) {
    std::cerr << "No tree for " << name << '\n';
    return;
  }
  if ((entry < 0) || (entry >= tree->GetEntries())) {
    std::cerr << name << ": Out of event range ! " << entry << '\n';
    return;
  }
  tree->SetBranchAddress(name, &arr);
  tree->GetEvent(entry);
  std::cout << "Number of " << name << "s: " << arr->size() << '\n';
}

void Data::loadData(int entry)
{
  loadDigits(entry);
  //load<Digit>(mDigiTree, "ITSDigit", entry, &mDigits);
  load<Cluster>(mClusTree, "ITSCluster", entry, &mClusters);
  load<o2::ITS::TrackITS>(mTracTree, "ITSTrack", entry, &mTracks);
}

constexpr float sizey = SegmentationAlpide::ActiveMatrixSizeRows;
constexpr float sizex = SegmentationAlpide::ActiveMatrixSizeCols;
constexpr float dy = SegmentationAlpide::PitchRow;
constexpr float dx = SegmentationAlpide::PitchCol;
constexpr float gap = 1e-4; // For a better visualization of pixels

TEveElement* Data::getEveChipDigits(int chip)
{
  static TEveFrameBox* box = new TEveFrameBox();
  box->SetAAQuadXY(0, 0, 0, sizex, sizey);
  box->SetFrameColor(kGray);

  // Digits
  TEveQuadSet* qdigi = new TEveQuadSet("digits");
  qdigi->SetOwnIds(kTRUE);
  qdigi->SetFrame(box);
  qdigi->Reset(TEveQuadSet::kQT_RectangleXY, kFALSE, 32);
  auto gman = o2::ITS::GeometryTGeo::Instance();
  std::vector<int> occup(gman->getNumberOfChips());
  for (const auto& d : mDigits) {
    auto id = d.getChipIndex();
    occup[id]++;
    if (id != chip)
      continue;

    int row = d.getRow();
    int col = d.getColumn();
    int charge = d.getCharge();
    qdigi->AddQuad(col * dx + gap, row * dy + gap, 0., dx - 2 * gap, dy - 2 * gap);
    qdigi->QuadValue(charge);
  }
  qdigi->RefitPlex();
  TEveTrans& t = qdigi->RefMainTrans();
  t.RotateLF(1, 3, 0.5 * TMath::Pi());
  t.SetPos(0, 0, 0);

  auto most = std::distance(occup.begin(), std::max_element(occup.begin(), occup.end()));
  std::cout << "Most occupied chip: " << most << " (" << occup[most] << " digits)\n";
  std::cout << "Chip "<<chip<<" number of digits "<< occup[chip]<<'\n';

  return qdigi;
}

TEveElement* Data::getEveChipClusters(int chip)
{
  static TEveFrameBox* box = new TEveFrameBox();
  box->SetAAQuadXY(0, 0, 0, sizex, sizey);
  box->SetFrameColor(kGray);

  // Clusters
  TEveQuadSet* qclus = new TEveQuadSet("clusters");
  qclus->SetOwnIds(kTRUE);
  qclus->SetFrame(box);
  int ncl=0;
  qclus->Reset(TEveQuadSet::kQT_LineXYFixedZ, kFALSE, 32);
  for (const auto& c : mClusters) {
    auto id = c.getSensorID();
    if (id != chip)
      continue;
    ncl++;
    
    int row = c.getPatternRowMin();
    int col = c.getPatternColMin();
    int len = c.getPatternColSpan();
    int wid = c.getPatternRowSpan();
    qclus->AddLine(col * dx, row * dy, len * dx, 0.);
    qclus->AddLine(col * dx, row * dy, 0., wid * dy);
    qclus->AddLine((col + len) * dx, row * dy, 0., wid * dy);
    qclus->AddLine(col * dx, (row + wid) * dy, len * dx, 0.);
  }
  qclus->RefitPlex();
  TEveTrans& ct = qclus->RefMainTrans();
  ct.RotateLF(1, 3, 0.5 * TMath::Pi());
  ct.SetPos(0, 0, 0);

  std::cout << "Chip "<<chip<<" number of clusters "<< ncl<<'\n';

  return qclus;
}

TEveElement* Data::getEveClusters()
{
  auto gman = o2::ITS::GeometryTGeo::Instance();
  TEvePointSet* clusters = new TEvePointSet("clusters");
  clusters->SetMarkerColor(kBlue);
  for (const auto& c : mClusters) {
    const auto& gloC = c.getXYZGloRot(*gman);
    clusters->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
  }
  return clusters;
}

TEveElement* Data::getEveTracks()
{
  auto gman = o2::ITS::GeometryTGeo::Instance();
  TEveTrackList* tracks = new TEveTrackList("tracks");
  auto prop = tracks->GetPropagator();
  prop->SetMagField(0.5);
  prop->SetMaxR(50.);
  for (const auto& rec : mTracks) {
    std::array<float, 3> p;
    rec.getPxPyPzGlo(p);
    TEveRecTrackD t;
    t.fP = { p[0], p[1], p[2] };
    t.fSign = (rec.getSign() < 0) ? -1 : 1;
    TEveTrack* track = new TEveTrack(&t, prop);
    track->SetLineColor(kMagenta);
    tracks->AddElement(track);

    if (mClusters.empty())
      continue;
    TEvePointSet* tpoints = new TEvePointSet("tclusters");
    tpoints->SetMarkerColor(kGreen);
    int nc = rec.getNumberOfClusters();
    while (nc--) {
      Int_t idx = rec.getClusterIndex(nc);
      const Cluster& c = mClusters[idx];
      const auto& gloC = c.getXYZGloRot(*gman);
      tpoints->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
    }
    track->AddElement(tpoints);
  }
  tracks->MakeTracks();

  return tracks;
}

void Data::displayData(int entry, int chip)
{
  std::string ename("Event #");
  ename += std::to_string(entry);

  // Chip display
  auto chipDigits = getEveChipDigits(chip);
  auto chipClusters = getEveChipClusters(chip);
  delete mChip;
  std::string cname(ename + "  ALPIDE chip #");
  cname += std::to_string(chip);
  mChip = new TEveElementList(cname.c_str());
  mChip->AddElement(chipDigits);
  mChip->AddElement(chipClusters);
  gEve->AddElement(mChip);

  // Event display
  auto clusters = getEveClusters();
  auto tracks = getEveTracks();
  delete mEvent;
  mEvent = new TEveElementList(ename.c_str());
  mEvent->AddElement(clusters);
  mEvent->AddElement(tracks);
  auto multi = o2::EventVisualisation::MultiView::getInstance();
  multi->registerEvent(mEvent);

  gEve->Redraw3D(kFALSE);
}

void load(int entry = 0, int chip = 13)
{
  std::cout << "\n*** Event #" << entry << " ***\n";

  gEntry->SetIntNumber(entry);
  gChipID->SetIntNumber(chip);

  evdata.loadData(entry);
  evdata.displayData(entry, chip);
}

void init(int entry = 0, int chip = 13,
          std::string digifile = "itsdigits.root",
	  bool rawdata = false,
          std::string clusfile = "o2clus_its.root",
          std::string tracfile = "o2trac_its.root",
          std::string inputGeom = "O2geometry.root")
{
  TEveManager::Create(kTRUE, "V");
  TEveBrowser* browser = gEve->GetBrowser();

  // Geometry
  o2::Base::GeometryManager::loadGeometry(inputGeom, "FAIRGeom");
  auto gman = o2::ITS::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G));

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

  // Event navigation
  browser->StartEmbedding(TRootBrowser::kBottom);
  auto frame = new TGMainFrame(gClient->GetRoot(), 1000, 600, kVerticalFrame);

  auto h = new TGHorizontalFrame(frame);
  auto b = new TGTextButton(h, "PrevEvnt", "prev()");
  h->AddFrame(b);
  gEntry = new TGNumberEntry(h, 0, 5, -1, TGNumberFormat::kNESInteger, TGNumberFormat::kNEANonNegative, TGNumberFormat::kNELLimitMinMax, 0, 10000);
  gEntry->Connect("ValueSet(Long_t)", 0, 0, "navigate()");
  h->AddFrame(gEntry);
  b = new TGTextButton(h, "NextEvnt", "next()");
  h->AddFrame(b);
  frame->AddFrame(h);

  // Chip navigation
  h = new TGHorizontalFrame(frame);
  b = new TGTextButton(h, "PrevChip", "prevChip()");
  h->AddFrame(b);
  gChipID = new TGNumberEntry(h, 0, 5, -1, TGNumberFormat::kNESInteger, TGNumberFormat::kNEANonNegative, TGNumberFormat::kNELLimitMinMax, 0, gman->getNumberOfChips());
  gChipID->Connect("ValueSet(Long_t)", 0, 0, "loadChip()");
  h->AddFrame(gChipID);
  b = new TGTextButton(h, "NextChip", "nextChip()");
  h->AddFrame(b);
  frame->AddFrame(h);

  frame->MapSubwindows();
  frame->MapWindow();
  browser->StopEmbedding("Navigator");

  TFile *file;

  // Data sources
  if (rawdata) {
    std::cout<<"Running with raw digits...\n";
    evdata.setRawPixelReader(digifile.data());
  } else {
    std::cout<<"Running with MC digits...\n";
    evdata.setDigitPixelReader(digifile.data());
    /*
    file = TFile::Open(digifile.data());
    if (file && gFile->IsOpen())
      evdata.setDigiTree((TTree*)gFile->Get("o2sim"));
    else
      std::cerr << "Cannot open file: " << digifile << '\n';
    */
  }
  
  file = TFile::Open(clusfile.data());
  if (file && gFile->IsOpen())
    evdata.setClusTree((TTree*)gFile->Get("o2sim"));
  else
    std::cerr << "Cannot open file: " << clusfile << '\n';

  file = TFile::Open(tracfile.data());
  if (file && gFile->IsOpen())
    evdata.setTracTree((TTree*)gFile->Get("o2sim"));
  else
    std::cerr << "Cannot open file: " << tracfile << '\n';

  std::cout << "\n **** Navigation over events and chips ****\n";
  std::cout << " load(event, chip) \t jump to the specified event and chip\n";
  std::cout << " next() \t\t load next event \n";
  std::cout << " prev() \t\t load previous event \n";
  std::cout << " loadChip(chip) \t jump to the specified chip within the current event \n";
  std::cout << " nextChip() \t\t load the next chip within the current event \n";
  std::cout << " prevChip() \t\t load the previous chip within the current event \n";

  load(entry, chip);
  gEve->Redraw3D(kTRUE);
}

void navigate()
{
  auto event = gEntry->GetNumberEntry()->GetIntNumber();
  auto chip = gChipID->GetNumberEntry()->GetIntNumber();
  load(event, chip);
}

void next()
{
  auto event = gEntry->GetNumberEntry()->GetIntNumber();
  event++;
  gEntry->SetIntNumber(event);
  auto chip = gChipID->GetNumberEntry()->GetIntNumber();
  load(event, chip);
}

void prev()
{
  auto event = gEntry->GetNumberEntry()->GetIntNumber();
  event--;
  gEntry->SetIntNumber(event);
  auto chip = gChipID->GetNumberEntry()->GetIntNumber();
  load(event, chip);
}

void loadChip()
{
  auto event = gEntry->GetNumberEntry()->GetIntNumber();
  auto chip = gChipID->GetNumberEntry()->GetIntNumber();
  evdata.displayData(event, chip);
}

void loadChip(int chip)
{
  gChipID->SetIntNumber(chip);
  loadChip();
}

void nextChip()
{
  auto chip = gChipID->GetNumberEntry()->GetIntNumber();
  chip++;
  gChipID->SetIntNumber(chip);
  loadChip();
}

void prevChip()
{
  auto chip = gChipID->GetNumberEntry()->GetIntNumber();
  chip--;
  gChipID->SetIntNumber(chip);
  loadChip();
}

void DisplayEvents()
{
  // A dummy function with the same name as this macro
}
