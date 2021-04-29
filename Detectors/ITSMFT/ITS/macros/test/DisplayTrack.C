/// \file DisplayTrack.C
/// \brief Simple macro to display ITSU tracks

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <string>

#include <TEveGeoNode.h>
#include <TEveGeoShape.h>
#include <TEveGeoShapeExtract.h>
#include <TEveManager.h>
#include <TEvePointSet.h>
#include <TFile.h>
#include <TGLViewer.h>
#include <TGeoManager.h>
#include <TMath.h>
#include <TString.h>
#include <TTree.h>

#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSMFTSimulation/Hit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITS/TrackITS.h"
#include "MathUtils/Cartesian.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#endif

using namespace std;

void DisplayTrack(Int_t event = 0, Int_t track = 0, std::string tracfile = "o2trac_its.root", std::string clusfile = "o2clus_its.root", std::string hitfile = "o2sim_HitsITS.root", std::string inputGeom = "", std::string dictfile = "")
{
  using namespace o2::base;
  using namespace o2::its;

  using o2::itsmft::Cluster;
  using o2::itsmft::Hit;

  TFile* f = nullptr;

  if (gEve == nullptr) {
    TEveManager::Create();
  }

  // Load geometry
  if (gGeoManager == nullptr) {
    o2::base::GeometryManager::loadGeometry(inputGeom);
  }

  gGeoManager->GetVolume("IBCYSSCylinderFoam")->SetInvisible();

  gGeoManager->GetVolume("ITSUHalfStave0")->SetTransparency(50);
  gGeoManager->GetVolume("ITSUHalfStave0")->SetLineColor(2);
  gGeoManager->GetVolume("ITSUHalfStave1")->SetTransparency(50);
  gGeoManager->GetVolume("ITSUHalfStave1")->SetLineColor(2);
  gGeoManager->GetVolume("ITSUHalfStave2")->SetTransparency(50);
  gGeoManager->GetVolume("ITSUHalfStave2")->SetLineColor(2);
  gGeoManager->GetVolume("ITSUHalfStave3")->SetTransparency(50);
  gGeoManager->GetVolume("ITSUHalfStave3")->SetLineColor(2);
  gGeoManager->GetVolume("ITSUHalfStave4")->SetTransparency(50);
  gGeoManager->GetVolume("ITSUHalfStave4")->SetLineColor(2);
  gGeoManager->GetVolume("ITSUHalfStave5")->SetTransparency(50);
  gGeoManager->GetVolume("ITSUHalfStave5")->SetLineColor(2);
  gGeoManager->GetVolume("ITSUHalfStave6")->SetTransparency(50);
  gGeoManager->GetVolume("ITSUHalfStave6")->SetLineColor(2);

  TGeoNode* tnode = gGeoManager->GetVolume("barrel")->FindNode("ITSV_2");
  TEveGeoTopNode* evenode = new TEveGeoTopNode(gGeoManager, tnode);
  evenode->SetVisLevel(4);
  gEve->AddGlobalElement(evenode);

  TGLViewer* view = gEve->GetDefaultGLViewer();
  Double_t center[3]{0};
  view->CurrentCamera().Reset();
  view->CurrentCamera().Configure(3., 1200., center, 0., 89 * 3.14 / 180);

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
  f = TFile::Open(hitfile.data());
  TTree* tree = (TTree*)gDirectory->Get("o2sim");

  string s{"event"};
  s += std::to_string(event);
  s += "_hits";
  s += std::to_string(track);
  TEvePointSet* points = new TEvePointSet(s.data());
  points->SetMarkerColor(kBlue);

  std::vector<Hit>* hitArr = nullptr;
  tree->SetBranchAddress("ITSHit", &hitArr);

  tree->GetEvent(event);

  Int_t n = 0;
  for (const auto& h : *hitArr) {
    if (h.GetTrackID() == track) {
      points->SetNextPoint(h.GetX(), h.GetY(), h.GetZ());
      n++;
    }
  }
  std::cout << "Number of hits: " << n << std::endl;

  gEve->AddElement(points, 0);
  f->Close();

  // Clusters
  f = TFile::Open(clusfile.data());
  tree = (TTree*)gDirectory->Get("o2sim");

  s = "event";
  s += std::to_string(event);
  s += "_clusters";
  s += std::to_string(track);
  points = new TEvePointSet(s.data());
  points->SetMarkerColor(kMagenta);

  std::vector<o2::itsmft::CompClusterExt>* clusArr = nullptr;
  tree->SetBranchAddress("ITSClusterComp", &clusArr);
  if (dictfile.empty()) {
    dictfile = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "", ".bin");
  }
  o2::itsmft::TopologyDictionary dict;
  std::ifstream file(dictfile.c_str());
  if (file.good()) {
    LOG(INFO) << "Running with dictionary: " << dictfile.c_str();
    dict.readBinaryFile(dictfile);
  } else {
    LOG(INFO) << "Cannot run without the dictionary !";
    return;
  }

  // ROFrecords
  std::vector<o2::itsmft::ROFRecord>* rofRecVecP = nullptr;
  tree->SetBranchAddress("ITSClustersROF", &rofRecVecP);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clsLabArr = nullptr;
  tree->SetBranchAddress("ITSClusterMCTruth", &clsLabArr);

  tree->GetEvent(0);
  int nc = clusArr->size();
  int offset = 0;
  for (auto& rof : *rofRecVecP) {
    offset = rof.getFirstEntry();
    for (int i = 0; i < rof.getNEntries(); i++) { // Find the TF containing this MC event
      auto mclab = (clsLabArr->getLabels(offset + i))[0];
      auto id = mclab.getEventID();
      if (id == event)
        goto found;
    }
  }
  std::cout << "RO frame containing the MC event " << event << " was not found" << std::endl;

found:
  o2::its::GeometryTGeo* gman = GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G)); // request cached transforms

  nc = clusArr->size();
  n = 0;
  for (int i = 0; i < nc; i++) {
    const auto& c = (*clusArr)[i];
    auto lab = (clsLabArr->getLabels(i))[0];
    if (lab.getEventID() != event)
      continue;
    if (lab.getTrackID() == track) {
      auto pattID = c.getPatternID();
      auto locC = dict.getClusterCoordinates(c);
      auto chipID = c.getSensorID();
      auto gloC = gman->getMatrixL2G(chipID) * locC;
      points->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
      n++;
    }
  }
  std::cout << "Number of clusters: " << n << std::endl;

  gEve->AddElement(points, 0);
  f->Close();

  // Track
  f = TFile::Open(tracfile.data());
  tree = (TTree*)gDirectory->Get("o2sim");

  s = "event";
  s += std::to_string(event);
  s += "_track";
  s += std::to_string(track);
  points = new TEvePointSet(s.data());
  points->SetMarkerColor(kGreen);

  std::vector<TrackITS>* trkArr = nullptr;
  std::vector<int>* clIdx = nullptr;
  tree->SetBranchAddress("ITSTrack", &trkArr);
  tree->SetBranchAddress("ITSTrackClusIdx", &clIdx);
  // Track MC labels
  std::vector<o2::MCCompLabel>* trkLabArr = nullptr;
  tree->SetBranchAddress("ITSTrackMCTruth", &trkLabArr);

  tree->GetEvent(0);

  Int_t nt = trkArr->size();
  n = 0;
  while (nt--) {
    const TrackITS& t = (*trkArr)[nt];
    auto lab = (*trkLabArr)[nt];
    if (TMath::Abs(lab.getEventID()) != event)
      continue;
    if (TMath::Abs(lab.getTrackID()) != track)
      continue;
    Int_t nc = t.getNumberOfClusters();
    int idxRef = t.getFirstClusterEntry();
    while (n < nc) {
      Int_t idx = (*clIdx)[idxRef + n];
      auto& c = (*clusArr)[offset + idx];
      auto locC = dict.getClusterCoordinates(c);
      auto chipID = c.getSensorID();
      auto gloC = gman->getMatrixL2G(chipID) * locC;
      points->SetNextPoint(gloC.X(), gloC.Y(), gloC.Z());
      n++;
    }
    break;
  }
  std::cout << "Number of attached clusters: " << n << std::endl;

  gEve->AddElement(points, 0);
  f->Close();
}
