// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CheckClustersIOTOF.C
/// \brief Simple macro to create clusters from TF3 digits

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <algorithm>

#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2D.h>
#include <TNtuple.h>
#include <TTree.h>
#include <TStyle.h>

#include "IOTOFBase/IOTOFBaseParam.h"
#include "IOTOFBase/GeometryTGeo.h"
#include "IOTOFBase/Segmentation.h"
#include "IOTOFSimulation/Chip.h"
#include "IOTOFReconstruction/TopologyClassifier.h"
#include "ITSMFTSimulation/Hit.h"
#include "DataFormatsIOTOF/Digit.h"
#include "DataFormatsIOTOF/Cluster.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "DetectorsBase/GeometryManager.h"

#include "DataFormatsITSMFT/ROFRecord.h"

#endif

using namespace o2::base;
using namespace o2::iotof;
using o2::iotof::Digit;
using o2::iotof::Cluster;

struct ClusterProperties {
  int clsIdx = -1;
  int eventID = -1;
  int trackID = -1;
  int chipID = -1;
  int layer = -1;
  uint16_t pattern = 0;
  int rowStart = 0;
  uint8_t rowSpan = 0;
  int colStart = 0;
  uint8_t colSpan = 0;
  int size = 0;
  bool isPrimary = false;
  bool isFake = false;
  bool isFakeDiffHits = false;
  bool isFakeDiffTrks = false;
  bool isFakeDiffEvts = false;
  int hitIdx = -1;
  Topologies topology = kOther;
  uint32_t topoKey = 0;
};

struct HitData {
  int hitIdx = -1;      // In the hitsPerEvent[iEvt] array
  std::vector<int> assocClsIdxs{};
  std::vector<int> assocDigitIdxs{};
};

struct TrackData {
  std::unordered_map<int, std::vector<HitData>> hitsByDetector;
};

void GetHitAvgPositionGlobal(const o2::itsmft::Hit& hit, o2::math_utils::Point3D<float>& avgPos) {

  o2::math_utils::Point3D<float> startPos = hit.GetPosStart();
  o2::math_utils::Point3D<float> endPos = hit.GetPos();

  avgPos = o2::math_utils::Point3D<float>((startPos.X() + endPos.X()) / 2, (startPos.Y() + endPos.Y()) / 2, (startPos.Z() + endPos.Z()) / 2);
}



void GetHitAvgPositionLocal(const o2::itsmft::Hit& hit, o2::iotof::GeometryTGeo* geom, o2::math_utils::Point3D<float>& avgPos) {

  const int chipID = hit.GetDetectorID();

  o2::math_utils::Point3D<float> startPos = hit.GetPosStart();
  auto startPosLocal = geom->getMatrixL2G(chipID) ^ (startPos);
  o2::math_utils::Point3D<float> endPos = hit.GetPos();
  auto endPosLocal = geom->getMatrixL2G(chipID) ^ (endPos);

  avgPos = o2::math_utils::Point3D<float>((startPosLocal.X() + endPosLocal.X()) / 2, (startPosLocal.Y() + endPosLocal.Y()) / 2, (startPosLocal.Z() + endPosLocal.Z()) / 2);
}


void GetDigitGlobalPos(const Digit& digit,
                       o2::math_utils::Point3D<float>& globalPos,
                       o2::iotof::GeometryTGeo* geom,
                       o2::iotof::Segmentation* segm) {
    const int chipID = digit.getChipIndex();
    const int layer = geom->getIOTOFLayer(chipID);

    float x = 0.f;
    float z = 0.f;
    if (layer >= 0)
        segm->detectorToLocal(digit.getRow(), digit.getColumn(), x, z, layer);

    globalPos = geom->getMatrixL2G(chipID)(o2::math_utils::Point3D<float>{x, 0.f, z});
}


void PrintMcTrack(bool verbose, const o2::MCTrack& mcTrack) {
  if (!verbose) {
    return;
  }
  std::cout << "MCTrack: pdgCode = " << mcTrack.GetPdgCode() << ", isPrimary = " << mcTrack.isPrimary() << ", process: " << mcTrack.getProcess() << ", pt = " << mcTrack.GetPt() << ", eta = " << mcTrack.GetEta() << ", phi = " << mcTrack.GetPhi() << std::endl;
}


void PrintHit(bool verbose, o2::itsmft::Hit hit, o2::iotof::GeometryTGeo* iotofGeom) {
  if (!verbose) {
    return;
  }
  int layer{-1}, stave{-1}, subStave{-1}, module{-1}, chip{-1};
  iotofGeom->getIOTOFChipId(hit.GetDetectorID(), layer, stave, subStave, module, chip);
  std::cout << "Hit: detectorID = " << hit.GetDetectorID() << ", layer = " << layer << ", stave = " << stave << ", subStave = " << subStave << ", module = " << module << ", chip = " << chip << ", trackID = " << hit.GetTrackID() << ", X = " << hit.GetX() << ", Y = " << hit.GetY() << ", Z = " << hit.GetZ() << ", time = " << hit.GetTime() << std::endl;
}


void PrintDigit(bool verbose, const o2::iotof::Digit& digit, auto& labels, o2::iotof::GeometryTGeo* iotofGeom, o2::iotof::Segmentation* segmInfo) {
  if (!verbose) {
    return;
  }

  if (labels.empty()) {
    std::cout << "Digit: no MCCompLabel associated, chipID = " << digit.getChipIndex() << ", row = " << digit.getRow() << ", col = " << digit.getColumn() << ", charge = " << digit.getCharge() << ", time = " << digit.getTime() << std::endl;
    return;
  }
  const auto& evtTrackLabel = labels[0];
  if (!evtTrackLabel.isValid()) {
    std::cout << "Digit: invalid MCCompLabel, chipID = " << digit.getChipIndex() << ", row = " << digit.getRow() << ", col = " << digit.getColumn() << ", charge = " << digit.getCharge() << ", time = " << digit.getTime() << std::endl;
    return;
  }

  const int eventID = evtTrackLabel.getEventID();
  const int trackID = evtTrackLabel.getTrackID();

  int layer{-1}, stave{-1}, subStave{-1}, module{-1}, chip{-1};
  iotofGeom->getIOTOFChipId(digit.getChipIndex(), layer, stave, subStave, module, chip);
  o2::math_utils::Point3D<float> digitPos;
  GetDigitGlobalPos(digit, digitPos, iotofGeom, segmInfo);
  std::cout << "Digit: trackID = " << trackID << ", eventID = " << eventID << ", chipID = "
            << digit.getChipIndex() << ", layer = " << layer << ", stave = " << stave
            << ", subStave = " << subStave << ", module = " << module << ", chip = " << chip
            << ", row = " << digit.getRow() << ", col = " << digit.getColumn() << ", charge = " << digit.getCharge()
            << ", time = " << digit.getTime()  << ", global position = (" << digitPos.X() << ", " << digitPos.Y()
            << ", " << digitPos.Z() << ")" << std::endl;
}


void PrintCluster(bool verbose,
                  const o2::iotof::Cluster& cluster,
                  auto clsLabel,
                  o2::iotof::GeometryTGeo* iotofGeom,
                  o2::iotof::Segmentation* segmInfo) {
  if (!verbose) {
    return;
  }

  if (clsLabel.empty())
    return;

  std::cout << "Cluster: " << clsLabel.size() << " MCCompLabels, chipID=" << cluster.getChipID() << ", row=" << cluster.getRow() << ", col=" << cluster.getCol() << ", rowSpan=" << cluster.getRowSpan() << ", colSpan=" << cluster.getColSpan() << ", topology=" << cluster.getTopology() << std::endl;
  for (int iLabel = 0; iLabel < clsLabel.size(); ++iLabel) {
    const auto& evtTrackLabel = clsLabel[iLabel];
    if (!evtTrackLabel.isValid())
      continue;

    const int eventID = evtTrackLabel.getEventID();
    const int trackID = evtTrackLabel.getTrackID();
    std::cout << "    Label " << iLabel << ", eventID=" << eventID << ", trackID=" << trackID << std::endl;
  }
}


template <typename... Args>
void Print(bool verbose, Args&&... args) {
  if (!verbose) {
    return;
  }

  (std::cout << ... << std::forward<Args>(args)) << std::endl;
}


void GetClusterGlobalPos(const o2::iotof::Cluster& cluster,
                         TopologyInfo topoInfo,
                         o2::math_utils::Point3D<float>& globalPos,
                         o2::iotof::GeometryTGeo* iotofGeom,
                         o2::iotof::Segmentation* segmInfo){

  float x = 0.f;
  float y = 0.f;
  float z = 0.f;
  int rowCOG = cluster.getRow() + topoInfo.mOffsetXToCOG;
  int colCOG = cluster.getCol() + topoInfo.mOffsetZToCOG;
  segmInfo->detectorToLocal(rowCOG, colCOG, x, z, cluster.getChipID());
  globalPos = iotofGeom->getMatrixL2G(cluster.getChipID())(o2::math_utils::Point3D<float>{x, 0.f, z});
}


int FindBestMatchingHit(const o2::iotof::Cluster& cluster,
                        TopologyInfo topoInfo, 
                        std::vector<HitData>& chipHitsIdxs,
                        std::vector<o2::itsmft::Hit>* evtChipHits,
                        const std::vector<o2::iotof::Digit>* digitsArray,
                        o2::iotof::GeometryTGeo* iotofGeom,
                        o2::iotof::Segmentation* segmInfo){
    int bestHitIdx = -1;
    float minDistanceSq = std::numeric_limits<float>::max();
    o2::math_utils::Point3D<float> clsPos;
    GetClusterGlobalPos(cluster, topoInfo, clsPos, iotofGeom, segmInfo);

    for (int i = 0; i < chipHitsIdxs.size(); ++i) {
        const auto& hit = (*evtChipHits)[chipHitsIdxs[i].hitIdx];

        float dx = clsPos.X() - hit.GetX();
        float dy = clsPos.Y() - hit.GetY();
        float dz = clsPos.Z() - hit.GetZ();
        float distSq = dx*dx + dy*dy + dz*dz;

        if (distSq < minDistanceSq) {
            minDistanceSq = distSq;
            bestHitIdx = i;
        }
    }

    return bestHitIdx; // Returns -1 if no hit is within maxToleranceCm (true fake cluster)
}


void CheckClustersIOTOF(std::string kinefile = "o2sim_Kine.root",
                        std::string hitfile = "o2sim_HitsTF3.root",
                        std::string digiFilePath = "tf3digits.root",
                        std::string clsFilePath = "tf3clusters.root",
                        std::string clsFileTopoPath = "TF3ClustersTopologies.root",
                        std::string inputGeomPath = "o2sim_geometry.root",
                        bool verbose = false)
{
  Print(verbose, "CheckClustersTopologiesIOTOF: kinefile = ", kinefile, ", hitfile = ", hitfile, ", digiFilePath = ", digiFilePath, ", clsFilePath = ", clsFilePath, ", inputGeomPath = ", inputGeomPath);
  gStyle->SetPalette(55);

  o2::conf::ConfigurableParam::updateFromString("IOTOFBase.segmentedInnerTOF=true;IOTOFBase.segmentedOuterTOF=true;IOTOFBase.enableForwardTOF=false;IOTOFBase.enableBackwardTOF=false");

  auto segmInfo = o2::iotof::Segmentation::Instance();

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeomPath);
  auto* iotofGeom = o2::iotof::GeometryTGeo::Instance();
  iotofGeom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  // Cluster topologies dictionary
  TFile* clsTopoFile = TFile::Open(clsFileTopoPath.data(), "READ");
  auto* clsTopoMapPtr = clsTopoFile->Get<std::unordered_map<uint32_t, o2::iotof::TopologyInfo>>("TF3ClusterTopologies");
  if (clsTopoMapPtr) {
    std::cout << "Loaded " << clsTopoMapPtr->size() << " entries from TF3ClusterTopologies.root" << std::endl;
  } else {
    std::cerr << "Failed to load TF3ClusterTopologies from file!" << std::endl;
  }
  clsTopoFile->Close();

  std::cout << std::endl;
  std::cout << "Topologies summary: " << std::endl;
  TopologyClassifier topoClassifier(*clsTopoMapPtr);
  topoClassifier.print();
  std::cout << std::endl;

  // Generated MC tracks and TrackRefs information
  TFile* kineFile = TFile::Open(kinefile.data());
  TTree* kineTree = (TTree*)kineFile->Get("o2sim");
  const int nEvts = kineTree->GetEntries();
  std::vector<std::vector<o2::MCTrack>*> mcTracksPerEvent(nEvts, nullptr);
  std::vector<std::vector<o2::TrackReference>*> mcTracksRefsPerEvent(nEvts, nullptr);

  // Hits information
  TFile* hitFile = TFile::Open(hitfile.data());
  TTree* hitTree = (TTree*)hitFile->Get("o2sim");
  std::vector<std::vector<o2::itsmft::Hit>*> hitsPerEvent(nEvts, nullptr);

  // Digits information
  TFile* digFile = TFile::Open(digiFilePath.data());
  TTree* digitsTree = (TTree*)digFile->Get("o2sim");
  std::vector<o2::iotof::Digit>* digitsArray = nullptr;
  o2::dataformats::IOMCTruthContainerView* digitsLabelsArr = nullptr;

  digitsTree->SetBranchAddress("TF3Digit", &digitsArray);
  digitsTree->SetBranchAddress("TF3DigitMCTruth", &digitsLabelsArr);

  // Clusters information
  TFile* clsFile = TFile::Open(clsFilePath.data());
  TTree* clustersTree = (TTree*)clsFile->Get("o2sim");
  std::vector<o2::iotof::Cluster>* clustersArray = nullptr;
  std::vector<unsigned char>* clustersPatternsArray = nullptr;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clustersLabelsArr = nullptr;

  clustersTree->SetBranchAddress("TF3Cluster", &clustersArray);
  clustersTree->SetBranchAddress("TF3ClusterPatt", &clustersPatternsArray);
  clustersTree->SetBranchAddress("TF3ClusterMCTruth", &clustersLabelsArr);

  // Load hits and MC track refs, stored per-event
  hitTree->SetBranchAddress("TF3Hit", &hitsPerEvent[0]);
  kineTree->SetBranchAddress("MCTrack", &mcTracksPerEvent[0]);
  kineTree->SetBranchAddress("TrackRefs", &mcTracksRefsPerEvent[0]);
  for (int iEvt = 0; iEvt < nEvts; ++iEvt) {
    hitTree->SetBranchAddress("TF3Hit", &hitsPerEvent[iEvt]);
    hitTree->GetEntry(iEvt);
    kineTree->SetBranchAddress("MCTrack", &mcTracksPerEvent[iEvt]);
    kineTree->SetBranchAddress("TrackRefs", &mcTracksRefsPerEvent[iEvt]);
    kineTree->GetEntry(iEvt);
    Print(verbose, "Loaded hit event ", iEvt, " with ", hitsPerEvent[iEvt]->size(), " hits");
  }

  // Digits: TTree entries are not separated per-event, but all digits are stored in a single entry
  digitsTree->GetEntry(0);
  o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> digitsLabels;
  digitsLabelsArr->copyandflatten(digitsLabels);

  // Clusters: TTree entries are not separated per-event, but all clusters are stored in a single entry
  clustersTree->GetEntry(0);
  o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> clustersLabels;

  // Store hit, digit and cluster properties for all tracks in all events
  std::vector<std::unordered_map<int, TrackData>> allEvtsTrackData(nEvts);
  TH2F* hEtaPhiHitsPrmTrkLayer0 = new TH2F("hEtaPhiHitsPrmTrkLayer0", "hEtaPhiHitsPrmTrkLayer0;#phi;#eta", 300, 0, 6.28319, 40, -2, 2);
  TH2F* hEtaPhiHitsSecTrkLayer0 = new TH2F("hEtaPhiHitsSecTrkLayer0", "hEtaPhiHitsSecTrkLayer0;#phi;#eta", 300, 0, 6.28319, 40, -2, 2);
  TH2F* hEtaPhiHitsPrmTrkLayer1 = new TH2F("hEtaPhiHitsPrmTrkLayer1", "hEtaPhiHitsPrmTrkLayer1;#phi;#eta", 300, 0, 6.28319, 40, -2, 2);
  TH2F* hEtaPhiHitsSecTrkLayer1 = new TH2F("hEtaPhiHitsSecTrkLayer1", "hEtaPhiHitsSecTrkLayer1;#phi;#eta", 300, 0, 6.28319, 40, -2, 2);
  // Load Hits, which are stored per-event
  int nHits{0}, nHitsFromPrimaryTracks{0}, nHitsFromSecondaryTracks{0};
  Print(verbose, "\n\n----> Starting hits printouts ... ");
  for (int iEvt = 0; iEvt < nEvts; ++iEvt) {

    Print(verbose, "Event ", iEvt, ": ", hitsPerEvent[iEvt]->size(), " hits");
    for (int iHit = 0; iHit < hitsPerEvent[iEvt]->size(); ++iHit) {

      const auto& hit = (*hitsPerEvent[iEvt])[iHit];
      const int trackID = hit.GetTrackID();
      const int chipIndex = hit.GetDetectorID();
      allEvtsTrackData[iEvt][trackID].hitsByDetector[chipIndex].push_back({iHit, {}, {}});
      nHits++;

      // Fill histograms
      int hitLayer   = iotofGeom->getIOTOFLayer(hit.GetDetectorID());
      auto& mcTrack = (*mcTracksPerEvent[iEvt])[trackID];
      bool isPrimary = mcTrack.isPrimary();
      if (isPrimary) nHitsFromPrimaryTracks++;
      else           nHitsFromSecondaryTracks++;
      float genEta   = mcTrack.GetEta();
      float genPhi   = mcTrack.GetPhi();

      if (hitLayer == 0 && isPrimary)        { hEtaPhiHitsPrmTrkLayer0->Fill(genPhi, genEta); } 
      else if (hitLayer == 0 && !isPrimary)  { hEtaPhiHitsSecTrkLayer0->Fill(genPhi, genEta); }
      else if (hitLayer == 1 && isPrimary)   { hEtaPhiHitsPrmTrkLayer1->Fill(genPhi, genEta); }
      else                                   { hEtaPhiHitsSecTrkLayer1->Fill(genPhi, genEta); }

      // PrintHit(verbose, hit, iotofGeom);
    }
  }

  // Debug prints for digits, use MCCompLabel to get event ID (getEventID()), track ID (getTrackID())
  Print(verbose, "\n\n----> Starting digits printouts ... ");
  for (int iDigit = 0; iDigit < (int)digitsArray->size(); ++iDigit) {

    auto labels = digitsLabels.getLabels(iDigit);
    if (labels.empty())
      continue;
    const auto& evtTrackLabel = labels[0];
    if (!evtTrackLabel.isValid())
      continue;

    const int eventID = evtTrackLabel.getEventID();
    const int trackID = evtTrackLabel.getTrackID();

    if (eventID < 0 || eventID >= nEvts) {
      std::cerr << "WARNING: digit " << iDigit << " has invalid eventID=" << eventID << "\n";
      continue;
    }

    const auto& digit = (*digitsArray)[iDigit];
    const auto& digitLabels = digitsLabels.getLabels(iDigit);
    // PrintDigit(verbose, digit, digitLabels, iotofGeom, segmInfo);
    auto& hitList = allEvtsTrackData[eventID][trackID].hitsByDetector[digit.getChipIndex()];
    for (auto& hit : hitList) {
        hit.assocDigitIdxs.push_back(iDigit);
    }
  }

  // Debug prints for clusters, use MCCompLabel to get event ID (getEventID()), track ID (getTrackID())
  Print(verbose, "\n\n----> Starting clusters printouts ... ");
  for (int iCls = 0; iCls < (int)clustersArray->size(); ++iCls) {

    const auto& cls = (*clustersArray)[iCls];
    const auto& clsLabels = clustersLabelsArr->getLabels(iCls);

    if (clsLabels.empty())
      continue;
    const auto& evtTrackLabel = clsLabels[0];
    if (!evtTrackLabel.isValid())
      continue;

    const int eventID = evtTrackLabel.getEventID();
    const int trackID = evtTrackLabel.getTrackID();

    if (eventID < 0 || eventID >= nEvts) {
      std::cerr << "WARNING: cluster " << iCls << " has invalid eventID=" << eventID << "\n";
      continue;
    }

    // PrintCluster(verbose, cls, clsLabels, iotofGeom, segmInfo);
    auto& hitList = allEvtsTrackData[eventID][trackID].hitsByDetector[cls.getChipID()];
    for (auto& hit : hitList) {
        hit.assocClsIdxs.push_back(iCls);
    }
  }

  // Debug print of allEvtsTrackData structure
  for (int iEvt = 0; iEvt < nEvts; ++iEvt) {
    for (const auto& [trackID, trackData] : allEvtsTrackData[iEvt]) {
      Print(verbose, "\n\n\nEvent ", iEvt, ", Track ", trackID, ":");
      for (const auto& [chipID, hitsInfos] : trackData.hitsByDetector) {
        Print(verbose, "-----------\n", "Chip ", chipID, ": ", hitsInfos.size(), " hits");
        for (const auto& hitInfo : hitsInfos) {
          Print(verbose, "\nHit ", hitInfo.hitIdx, ": ", hitInfo.assocDigitIdxs.size(), " digits, ", hitInfo.assocClsIdxs.size(), " clusters");
          for (int iDigit=0; iDigit<hitInfo.assocDigitIdxs.size(); ++iDigit) {
            auto digit = (*digitsArray)[hitInfo.assocDigitIdxs[iDigit]];
            auto digitLabels = digitsLabels.getLabels(hitInfo.assocDigitIdxs[iDigit]);
            PrintDigit(verbose, digit, digitLabels, iotofGeom, segmInfo);
          }
          Print(verbose, "  Chip ", chipID, ": ", hitInfo.assocClsIdxs.size(), " clusters");
          for (int iCls=0; iCls<hitInfo.assocClsIdxs.size(); ++iCls) {
            auto cls = (*clustersArray)[hitInfo.assocClsIdxs[iCls]];
            auto clsLabels = clustersLabelsArr->getLabels(hitInfo.assocClsIdxs[iCls]);
            PrintCluster(verbose, cls, clsLabels, iotofGeom, segmInfo);
          }
        }
      }
    }
  }

  // Debug prints
  std::cout << "\n***********************************" << std::endl;
  Print(true, "Number of events:   ", nEvts);
  Print(true, "Number of hits:     ", nHits);
  Print(true, "-> from primary tracks:   ", nHitsFromPrimaryTracks);
  Print(true, "-> from secondary tracks: ", nHitsFromSecondaryTracks);
  Print(true, "Number of digits:   ", digitsArray->size());
  Print(true, "Number of digit labels:  ", digitsLabels.getNElements());
  Print(true, "Number of entries in digit tree: ", digitsTree->GetEntries());
  Print(true, "Number of clusters:   ", clustersArray->size());
  Print(true, "Number of clusters labels:  ", clustersLabelsArr->getNElements());
  Print(true, "Number of entries in cluster tree: ", clustersTree->GetEntries());
  std::cout << "***********************************\n" << std::endl;

  // Create vectors of digits with same chip index, cluster candidates
  TH2F* hCountHitMatchingType = new TH2F("hCountHitMatchingType", "hCountHitMatchingType;Hit matching type;#it{p}_{T}", 4, -0.5, 3.5, 50, 0, 10);
  hCountHitMatchingType->GetXaxis()->SetBinLabel(1, "Primary, 1 to 1");
  hCountHitMatchingType->GetXaxis()->SetBinLabel(2, "Secondary, 1 to 1");
  hCountHitMatchingType->GetXaxis()->SetBinLabel(3, "Primary, min distance");
  hCountHitMatchingType->GetXaxis()->SetBinLabel(4, "Secondary, min distance");

  std::vector<ClusterProperties> clustersProperties;
  clustersProperties.reserve(clustersArray->size()); // Pre-allocate memory

  for (int iCls = 0; iCls < (int)clustersArray->size(); ++iCls) {

    const auto& cluster = (*clustersArray)[iCls];

    // Cluster labels
    const auto& clsLabels = clustersLabelsArr->getLabels(iCls);
    std::cout << "Processing cluster " << iCls << " with " << clsLabels.size() << " MCCompLabels associated." << std::endl;
    if (clsLabels.empty()) {
      std::cout << "---> Empty cls label" << std::endl;
      continue;
    }

    const auto& firstEvtTrackLabel = clsLabels[0];
    if (!firstEvtTrackLabel.isValid()) {
      std::cout << "---> Invalid first evt-track label" << std::endl;
      continue;
    }
    const int eventID = firstEvtTrackLabel.getEventID();
    const int trackID = firstEvtTrackLabel.getTrackID();

    if (eventID < 0 || eventID >= nEvts) {
      std::cerr << "WARNING: cluster " << iCls << " has invalid eventID=" << eventID << "\n";
      continue;
    }

    ClusterProperties clsProps;
    clsProps.clsIdx = iCls;

    // Cluster geometric properties
    clsProps.chipID = cluster.getChipID();
    clsProps.layer = iotofGeom->getIOTOFLayer(cluster.getChipID());
    clsProps.rowStart = cluster.getRow();
    clsProps.rowSpan = cluster.getRowSpan();
    clsProps.colStart = cluster.getCol();
    clsProps.colSpan = cluster.getColSpan();
    clsProps.pattern = cluster.getPattern();
    clsProps.size = cluster.getSize();
    clsProps.topology = static_cast<Topologies>(cluster.getTopology());
    uint32_t clsTopoKey = (static_cast<uint32_t>(clsProps.rowSpan) << 24) |
                          (static_cast<uint32_t>(clsProps.colSpan) << 16) |
                           static_cast<uint32_t>(clsProps.pattern);
    clsProps.topoKey = clsTopoKey;
    TopologyInfo clsTopoInfo = topoClassifier.getTopologyFeatures(clsProps.topoKey);

    // Cluster association properties
    clsProps.eventID = eventID;
    clsProps.trackID = trackID;
    clsProps.isPrimary = false;
    clsProps.isFake = false;
    clsProps.isFakeDiffHits = false;
    clsProps.isFakeDiffTrks = false;
    clsProps.isFakeDiffEvts = false;
    clsProps.hitIdx = -1;

    // 1 to 1 hit-cluster correspondence, set eventID and trackID for the cluster
    if (clsLabels.size() > 1) {
      // Multiple hits associated with the cluster,
      // check consistency of track and event IDs across
      // all digits in the cluster to flag fake clusters
      for (int iLabel = 1; iLabel < clsLabels.size(); ++iLabel) {
        const auto& evtTrackLabel = clsLabels[iLabel];

        if (!evtTrackLabel.isValid()) {
          continue;
        }

        const int eventID = firstEvtTrackLabel.getEventID();
        const int trackID = firstEvtTrackLabel.getTrackID();

        if (eventID < 0 || eventID >= nEvts) {
          std::cerr << "WARNING: cluster " << iCls << " has invalid eventID=" << eventID << "\n";
          continue;
        }

        if (evtTrackLabel.getEventID() != eventID) {
          std::cout << "Cluster " << iCls << " has inconsistent event IDs across labels: " << evtTrackLabel.getEventID() << " != " << eventID << std::endl;
          clsProps.isFake = true;
          clsProps.isFakeDiffEvts = true;
        }
        if (evtTrackLabel.getTrackID() != trackID) {
          std::cout << "Cluster " << iCls << " has inconsistent track IDs across labels: " << evtTrackLabel.getTrackID() << " != " << trackID << std::endl;
          clsProps.isFake = true;
          clsProps.isFakeDiffTrks = true;
        }
      }
    }

    // Cluster-hit matching
    if (!clsProps.isFake) {

      const auto& mcTrack = (*mcTracksPerEvent[clsProps.eventID])[clsProps.trackID];
      clsProps.isPrimary = mcTrack.isPrimary();

      auto& chipHitsIdxs = allEvtsTrackData[clsProps.eventID][clsProps.trackID].hitsByDetector[clsProps.chipID];
      if (chipHitsIdxs.empty()) {
        clsProps.hitIdx = -1;
      } else if (chipHitsIdxs.size() == 1) {
        clsProps.hitIdx = 0;
        hCountHitMatchingType->Fill(clsProps.isPrimary ? 0 : 2, mcTrack.GetPt());
      } else {
        // Perform spatial matching for multi-hit candidates
        clsProps.hitIdx = FindBestMatchingHit(cluster, clsTopoInfo, chipHitsIdxs, hitsPerEvent[clsProps.eventID], digitsArray, iotofGeom, segmInfo);
        hCountHitMatchingType->Fill(clsProps.isPrimary ? 1 : 3, mcTrack.GetPt());
      }

      if (clsProps.hitIdx != -1) {
        chipHitsIdxs[clsProps.hitIdx].assocClsIdxs.push_back(clustersProperties.size());
      } else {
        clsProps.isFake = true;
        clsProps.isFakeDiffHits = true;
        std::cout << "Cluster " << iCls << " has no matching hit, marked as fake." << std::endl;
      }
    }

    // PrintCluster(verbose, cluster, digitsArray, digitsLabels, hitsPerEvent, iotofGeom, segmInfo);
    clustersProperties.push_back(clsProps);
  }
  Print(true, "----> Total number of clusters: ", clustersProperties.size());

  // QA printouts and histograms
  Print(true, "\n\n----> Starting QA logging ... ");
  const char* trackName[2] = {"Prm", "Sec"};

  // Count fake clusters
  TH1F* hCountFakeClusters[2][2];
  for (int layer = 0; layer < 2; ++layer) {
    for (int type = 0; type < 2; ++type) {
      hCountFakeClusters[layer][type] = new TH1F(Form("hCountFakeClusters%sTrkLayer%d", trackName[type], layer), Form("Fake Cluster Counter %s Trk Layer %d", trackName[type], layer), 6, -0.5, 5.5);
      hCountFakeClusters[layer][type]->GetXaxis()->SetBinLabel(1, "Total");
      hCountFakeClusters[layer][type]->GetXaxis()->SetBinLabel(2, "Real");
      hCountFakeClusters[layer][type]->GetXaxis()->SetBinLabel(3, "Fake");
      hCountFakeClusters[layer][type]->GetXaxis()->SetBinLabel(4, "Fake NoHit");
      hCountFakeClusters[layer][type]->GetXaxis()->SetBinLabel(5, "Fake DiffTrks");
      hCountFakeClusters[layer][type]->GetXaxis()->SetBinLabel(6, "Fake DiffEvts");
    }
  }

  // Loop over clusters and fill histograms
  for (const auto& cluster : clustersProperties) {
    int layer = cluster.layer;
    int type = cluster.isPrimary ? 0 : 1;
    hCountFakeClusters[layer][type]->Fill(0.f, 1); // Total clusters
    if (cluster.isFake) {
      hCountFakeClusters[layer][type]->Fill(2.f, 1); // Fake clusters
      if (cluster.isFakeDiffHits) {
        hCountFakeClusters[layer][type]->Fill(3.f, 1); // Fake NoHit
      }
      if (cluster.isFakeDiffTrks) {
        hCountFakeClusters[layer][type]->Fill(4.f, 1); // Fake DiffTrks
      }
      if (cluster.isFakeDiffEvts) {
        hCountFakeClusters[layer][type]->Fill(5.f, 1); // Fake DiffEvts
      }
    } else {
      hCountFakeClusters[layer][type]->Fill(1.f, 1); // Real clusters
    }
  }

  Print(true, "----> hCountFakeClusters filled");
  // Topology names
  const std::array<std::string, kNTopologies> topologyNames = {
      "kSingleDigit", "kLineOnRow", "kLineOnCol", "kDiagonal", "kSquare",
      "kUpperTriangleLeft", "kUpperTriangleRight", "kLowerTriangleLeft",
      "kLowerTriangleRight", "kSnake", "kSnakeRot90", "kSnakeRefl",
      "kSnakeRot90Refl", "kHuge", "kOther"};

  // Count topologies from frequency values in 
  // topologies dictionary and fill the summary histograms
  TH1F* hTopoSummaryDictionary = new TH1F("hTopoSummaryDictionary", "Cluster Topology Count Summary;;Counts", kNTopologies, 0, kNTopologies);
  for (const auto& [topoKey, topology] : topoClassifier.getTopologyMap()) {
    hTopoSummaryDictionary->Fill(topology.mTopology, topology.mFrequency);
  }

  TH2F *hTrueClsSizeVsEta[2][2], *hTrueClsSizeVsPhi[2][2], *hFakeClsSizeVsEta[2][2], *hFakeClsSizeVsPhi[2][2], 
       *hClustersEtaPhi[2][2], *hTopoVsEta[2][2], *hClsSizeVsTopo[2][2], *hXRes[2][2], *hYRes[2][2], *hZRes[2][2],
       *hTrackHitsXY[2][2], *hTrackDoubleHitsXY[2][2], *hTrackDoubleHitsPhiPt[2][2], *hTopoVsEtaPt[2][2][kNTopologies];
  TH1F *hNClustersFromHit[2][2], *hMeanTrueClsSizeVsEta[2][2], *hMeanTrueClsSizeVsPhi[2][2], *hMeanFakeClsSizeVsEta[2][2],
       *hMeanFakeClsSizeVsPhi[2][2], *hRmsXRes[2][2], *hRmsYRes[2][2], *hRmsZRes[2][2], *hMeanXRes[2][2], *hMeanYRes[2][2],
       *hMeanZRes[2][2];
  TH1F* hTopoSummaryTotal = new TH1F("hTopoSummaryTotal", "Cluster Topology Summary;;Counts", kNTopologies, 0, kNTopologies);
  TH1F* hTopoSummaryReal = new TH1F("hTopoSummaryReal", "Cluster Topology Summary;;Counts", kNTopologies, 0, kNTopologies);
  TH1F* hTopoSummaryFake = new TH1F("hTopoSummaryFake", "Cluster Topology Summary;;Counts", kNTopologies, 0, kNTopologies);

  Print(true, "----> Defining histograms");
  for (int layer = 0; layer < 2; ++layer) {
    for (int type = 0; type < 2; ++type) {
      hClustersEtaPhi[layer][type] = new TH2F(Form("hNClsVsEtaPhi%sTrkLayer%d", trackName[type], layer), "Cluster #eta vs #phi;#phi;#eta", 300, 0, 6.28319, 40, -2, 2);
      hTrueClsSizeVsEta[layer][type] = new TH2F(Form("hTrueClsSizeVsEta%sTrkLayer%d", trackName[type], layer), "True Cluster Size vs #eta;#eta", 300, -2, 2, 20, 0.5, 20.5);
      hTrueClsSizeVsPhi[layer][type] = new TH2F(Form("hTrueClsSizeVsPhi%sTrkLayer%d", trackName[type], layer), "True Cluster Size vs #phi;#phi", 300, 0, 6.28319, 20, 0.5, 20.5);
      hFakeClsSizeVsEta[layer][type] = new TH2F(Form("hFakeClsSizeVsEta%sTrkLayer%d", trackName[type], layer), "Fake Cluster Size vs #eta;#eta", 300, -2, 2, 20, 0.5, 20.5);
      hFakeClsSizeVsPhi[layer][type] = new TH2F(Form("hFakeClsSizeVsPhi%sTrkLayer%d", trackName[type], layer), "Fake Cluster Size vs #phi;#phi", 300, 0, 6.28319, 20, 0.5, 20.5);
      hNClustersFromHit[layer][type] = new TH1F(Form("hNClsPerHit%sTrkLayer%d", trackName[type], layer), ";N Cluster per Hit;Counts", 21, -0.5, 20.5);
      hMeanTrueClsSizeVsEta[layer][type] = new TH1F(Form("hMeanTrueClsSizeVsEta%sTrkLayer%d", trackName[type], layer), "Mean True Cluster Size vs #eta;#eta", 300, -2, 2);
      hMeanTrueClsSizeVsPhi[layer][type] = new TH1F(Form("hMeanTrueClsSizeVsPhi%sTrkLayer%d", trackName[type], layer), "Mean True Cluster Size vs #phi;#phi", 300, 0, 6.28319);
      hMeanFakeClsSizeVsEta[layer][type] = new TH1F(Form("hMeanFakeClsSizeVsEta%sTrkLayer%d", trackName[type], layer), "Mean Fake Cluster Size vs #eta;#eta", 300, -2, 2);
      hMeanFakeClsSizeVsPhi[layer][type] = new TH1F(Form("hMeanFakeClsSizeVsPhi%sTrkLayer%d", trackName[type], layer), "Mean Fake Cluster Size vs #phi;#phi", 300, 0, 6.28319);
      hTopoVsEta[layer][type] = new TH2F(Form("hClsSizeVsEtaTopo%sTrkLayer%d", trackName[type], layer), "Cluster Topology vs #eta;;#eta", kNTopologies, 0, kNTopologies, 20, -2, 2);
      hClsSizeVsTopo[layer][type] = new TH2F(Form("hClsSizeVsTopo%sTrkLayer%d", trackName[type], layer), "Cluster Topology vs N Digits;;N Digits", kNTopologies, 0, kNTopologies, 20, 0.5, 20.5);
      hXRes[layer][type] = new TH2F(Form("hDeltaXClsHit%sTrkLayer%d", trackName[type], layer), ";#Delta X;#eta", 1000, -0.2, 0.2, 20, -2, 2);
      hYRes[layer][type] = new TH2F(Form("hDeltaYClsHit%sTrkLayer%d", trackName[type], layer), ";#Delta Y;#eta", 1000, -0.2, 0.2, 20, -2, 2);
      hZRes[layer][type] = new TH2F(Form("hDeltaZClsHit%sTrkLayer%d", trackName[type], layer), ";#Delta Z;#eta", 1000, -0.2, 0.2, 20, -2, 2);
      hRmsXRes[layer][type] = new TH1F(Form("hRmsDeltaXClsHit%sTrkLayer%d", trackName[type], layer), ";#eta;RMS #Delta X", 20, -2, 2);
      hRmsYRes[layer][type] = new TH1F(Form("hRmsDeltaYClsHit%sTrkLayer%d", trackName[type], layer), ";#eta;RMS #Delta Y", 20, -2, 2);
      hRmsZRes[layer][type] = new TH1F(Form("hRmsDeltaZClsHit%sTrkLayer%d", trackName[type], layer), ";#eta;RMS #Delta Z", 20, -2, 2);
      hMeanXRes[layer][type] = new TH1F(Form("hMeanDeltaXClsHit%sTrkLayer%d", trackName[type], layer), ";#eta;Mean #Delta X", 20, -2, 2);
      hMeanYRes[layer][type] = new TH1F(Form("hMeanDeltaYClsHit%sTrkLayer%d", trackName[type], layer), ";#eta;Mean #Delta Y", 20, -2, 2);
      hMeanZRes[layer][type] = new TH1F(Form("hMeanDeltaZClsHit%sTrkLayer%d", trackName[type], layer), ";#eta;Mean #Delta Z", 20, -2, 2);

      if (layer == 0) {
        hTrackHitsXY[layer][type] = new TH2F(Form("hTrackHitsXY%sTrkLayer%d", trackName[type], layer), ";Hit X;Hit Y", 5000, -30, 30, 5000, -30, 30);
        hTrackDoubleHitsXY[layer][type] = new TH2F(Form("hTrackDoubleHitsXY%sTrkLayer%d", trackName[type], layer), ";Hit X;Hit Y", 5000, -30, 30, 5000, -30, 30);
      } else {
        hTrackHitsXY[layer][type] = new TH2F(Form("hTrackHitsXY%sTrkLayer%d", trackName[type], layer), ";Hit X;Hit Y", 10000, -100, 100, 10000, -100, 100);
        hTrackDoubleHitsXY[layer][type] = new TH2F(Form("hTrackDoubleHitsXY%sTrkLayer%d", trackName[type], layer), ";Hit X;Hit Y", 10000, -100, 100, 10000, -100, 100);
      }
      hTrackDoubleHitsPhiPt[layer][type] = new TH2F(Form("hTrackDoubleHitsPhiPt%sTrkLayer%d", trackName[type], layer), ";#phi;p_{T}", 3000, 0, 6.28319, 50, 0, 10);

      for (int topo = 0; topo < kNTopologies; ++topo) {
        hTopoSummaryReal->GetXaxis()->SetBinLabel(topo + 1, topologyNames[topo].c_str());
        hTopoSummaryFake->GetXaxis()->SetBinLabel(topo + 1, topologyNames[topo].c_str());
        hTopoSummaryTotal->GetXaxis()->SetBinLabel(topo + 1, topologyNames[topo].c_str());
        hTopoSummaryDictionary->GetXaxis()->SetBinLabel(topo + 1, topologyNames[topo].c_str());
        hTopoVsEta[layer][type]->GetXaxis()->SetBinLabel(topo + 1, topologyNames[topo].c_str());
        hClsSizeVsTopo[layer][type]->GetXaxis()->SetBinLabel(topo + 1, topologyNames[topo].c_str());
        hTopoVsEtaPt[layer][type][topo] = new TH2F(Form("h%sVsEtaPt_%sTrk_TrkLayer%d", topologyNames[topo].c_str(), trackName[type], layer), Form("Cluster Topology %s vs Eta and Pt;#eta;p_{T}", topologyNames[topo].c_str()), 100, -2, 2, 20, 0, 10);
      }
    }
  }

  // Loop over clusters
  Print(true, "----> Looping over clusters and filling histograms");
  for (const auto& cls : clustersProperties) {
    
    const int layer = cls.layer;
    const int topo  = static_cast<int>(cls.topology);
    
    const int chipID      = cls.chipID;
    const int eventID     = cls.eventID;
    const int trackID     = cls.trackID;

    const auto& mcTrack = (*mcTracksPerEvent[eventID])[trackID];
    const float eta     = mcTrack.GetEta();
    const float phi     = mcTrack.GetPhi();
    const float pt      = mcTrack.GetPt();
    const int type      = cls.isPrimary ? 0 : 1;
    const int size      = cls.size;
    
    hTopoVsEtaPt[layer][type][topo]->Fill(eta, pt);
    hTopoVsEta[layer][type]->Fill(topo, eta);
    
    hClsSizeVsTopo[layer][type]->Fill(topo, size);

    hTopoSummaryTotal->Fill(topo);
    if (cls.isFake) {
      hTopoSummaryFake->Fill(topo);
      hFakeClsSizeVsEta[layer][type]->Fill(eta, size);
      hFakeClsSizeVsPhi[layer][type]->Fill(phi, size);
    } else {
      hTopoSummaryReal->Fill(topo);
      hTrueClsSizeVsEta[layer][type]->Fill(eta, size);
      hTrueClsSizeVsPhi[layer][type]->Fill(phi, size);
    }

    if (cls.hitIdx < 0) {
      continue; // Skip clusters without a matching hit
    }
    const auto& hitData = allEvtsTrackData[cls.eventID][cls.trackID].hitsByDetector[cls.chipID][cls.hitIdx];
    auto& hit = (*hitsPerEvent[cls.eventID])[hitData.hitIdx];
    hNClustersFromHit[layer][type]->Fill(hitData.assocClsIdxs.size());
    if (hitData.assocClsIdxs.size() > 0)
      hClustersEtaPhi[layer][type]->Fill(phi, eta);

    o2::math_utils::Point3D<float> clusterPos;
    TopologyInfo clsTopoInfo = topoClassifier.getTopologyFeatures(cls.topoKey);
    auto clsFull = clustersArray->at(cls.clsIdx);
    GetClusterGlobalPos(clsFull, clsTopoInfo, clusterPos, iotofGeom, segmInfo);
    o2::math_utils::Point3D<float> avgPos;
    GetHitAvgPositionGlobal(hit, avgPos);
    hXRes[layer][type]->Fill(clusterPos.X() - avgPos.X(), eta);
    hYRes[layer][type]->Fill(clusterPos.Y() - avgPos.Y(), eta);
    hZRes[layer][type]->Fill(clusterPos.Z() - avgPos.Z(), eta);
  }

  // Fill means and RMS of cluster size and residuals
  for (int layer = 0; layer < 2; ++layer) {
    for (int type = 0; type < 2; ++type) {
      for (int etaBin = 1; etaBin <= hTrueClsSizeVsEta[layer][type]->GetNbinsX(); ++etaBin) {
        // Project 1D histogram to get mean cluster size for this eta bin
        TH1D* hClsSizeProj = hTrueClsSizeVsEta[layer][type]->ProjectionY(Form("hClsSizeProj_etaBin%d", etaBin), etaBin, etaBin);
        hMeanTrueClsSizeVsEta[layer][type]->SetBinContent(etaBin, hClsSizeProj->GetMean());
        hMeanTrueClsSizeVsEta[layer][type]->SetBinError(etaBin, hClsSizeProj->GetMeanError());
      }
      for (int phiBin = 1; phiBin <= hTrueClsSizeVsPhi[layer][type]->GetNbinsX(); ++phiBin) {
        // Project 1D histogram to get mean cluster size for this eta bin
        TH1D* hClsSizeProj = hTrueClsSizeVsPhi[layer][type]->ProjectionY(Form("hClsSizeProj_phiBin%d", phiBin), phiBin, phiBin);
        hMeanTrueClsSizeVsPhi[layer][type]->SetBinContent(phiBin, hClsSizeProj->GetMean());
        hMeanTrueClsSizeVsPhi[layer][type]->SetBinError(phiBin, hClsSizeProj->GetMeanError());
      }
      for (int etaBin = 1; etaBin <= hFakeClsSizeVsEta[layer][type]->GetNbinsX(); ++etaBin) {
        // Project 1D histogram to get mean cluster size for this eta bin
        TH1D* hClsSizeProj = hFakeClsSizeVsEta[layer][type]->ProjectionY(Form("hClsSizeProj_etaBin%d", etaBin), etaBin, etaBin);
        hMeanFakeClsSizeVsEta[layer][type]->SetBinContent(etaBin, hClsSizeProj->GetMean());
        hMeanFakeClsSizeVsEta[layer][type]->SetBinError(etaBin, hClsSizeProj->GetMeanError());
      }
      for (int phiBin = 1; phiBin <= hFakeClsSizeVsPhi[layer][type]->GetNbinsX(); ++phiBin) {
        // Project 1D histogram to get mean cluster size for this eta bin
        TH1D* hClsSizeProj = hFakeClsSizeVsPhi[layer][type]->ProjectionY(Form("hClsSizeProj_phiBin%d", phiBin), phiBin, phiBin);
        hMeanFakeClsSizeVsPhi[layer][type]->SetBinContent(phiBin, hClsSizeProj->GetMean());
        hMeanFakeClsSizeVsPhi[layer][type]->SetBinError(phiBin, hClsSizeProj->GetMeanError());
      }
      for (int etaBin = 1; etaBin <= hXRes[layer][type]->GetNbinsY(); ++etaBin) {
        TH1D* hXResProj = hXRes[layer][type]->ProjectionX(Form("hXResProj_etaBin%d", etaBin), etaBin, etaBin);
        TH1D* hYResProj = hYRes[layer][type]->ProjectionX(Form("hYResProj_etaBin%d", etaBin), etaBin, etaBin);
        TH1D* hZResProj = hZRes[layer][type]->ProjectionX(Form("hZResProj_etaBin%d", etaBin), etaBin, etaBin);
        hRmsXRes[layer][type]->SetBinContent(etaBin, hXResProj->GetRMS());
        hRmsYRes[layer][type]->SetBinContent(etaBin, hYResProj->GetRMS());
        hRmsZRes[layer][type]->SetBinContent(etaBin, hZResProj->GetRMS());
        hRmsXRes[layer][type]->SetBinError(etaBin, hXResProj->GetRMSError());
        hRmsYRes[layer][type]->SetBinError(etaBin, hYResProj->GetRMSError());
        hRmsZRes[layer][type]->SetBinError(etaBin, hZResProj->GetRMSError());
        hMeanXRes[layer][type]->SetBinContent(etaBin, hXResProj->GetMean());
        hMeanYRes[layer][type]->SetBinContent(etaBin, hYResProj->GetMean());
        hMeanZRes[layer][type]->SetBinContent(etaBin, hZResProj->GetMean());
        hMeanXRes[layer][type]->SetBinError(etaBin, hXResProj->GetMeanError());
        hMeanYRes[layer][type]->SetBinError(etaBin, hYResProj->GetMeanError());
        hMeanZRes[layer][type]->SetBinError(etaBin, hZResProj->GetMeanError());
      }
    }
  }

  Print(true, "----> Looping over generated particles");

  // Generated particles
  TH2F* hGenEtaPt[2] = {new TH2F("hGenEtaPtPrm", "Generated primary tracks;#eta;p_{T}", 100, -2, 2, 100, 0, 10),
                        new TH2F("hGenEtaPtSec", "Generated secondary tracks;#eta;p_{T}", 100, -2, 2, 100, 0, 10)};

  for (int iEvt = 0; iEvt < nEvts; ++iEvt) {
    for (const auto& mcTrack : *mcTracksPerEvent[iEvt]) {
      const int type = mcTrack.isPrimary() ? 0 : 1;
      hGenEtaPt[type]->Fill(mcTrack.GetEta(), mcTrack.GetPt());
    }
  }

  // Check eta and phi of tracks producing multiple hits, should reflect
  // overlaps between staves and validate the geometry implementation
  Print(true, "----> Looping over tracks producing multiple hits");
  for (int iEvt = 0; iEvt < nEvts; ++iEvt) {
    for (const auto& [trackID, trackData] : allEvtsTrackData[iEvt]) {

      const auto& mcTrack = (*mcTracksPerEvent[iEvt])[trackID];
      if (!mcTrack.isPrimary() || trackData.hitsByDetector.size() <= 1) {
        continue;
      }

      // Index 0 -> Layer 0, Index 1 -> Layer 1
      std::vector<int> distinctChips[2];

      for (const auto& [chipIdx, hitsVec] : trackData.hitsByDetector) {

        int layer{-1}, stave{-1}, subStave{-1}, module{-1}, chip{-1};
        iotofGeom->getIOTOFChipId(chipIdx, layer, stave, subStave, module, chip);

        // Check if current chip is a neighbor to any already accepted chip in this layer
        // Required because the same track can produce multiple hits in adjacent chips,
        // belonging to the same module/substave, therefore the double hit is not related
        // to the detector geometry
        const bool isNeighborToExisting = std::any_of(
          distinctChips[layer].begin(),
          distinctChips[layer].end(),
          [&](int existingChipIdx) {
            int layerA{-1}, staveA{-1}, subStaveA{-1}, moduleA{-1}, chipA{-1};
            iotofGeom->getIOTOFChipId(existingChipIdx, layerA, staveA, subStaveA, moduleA, chipA);

            // Reject adjacent modules in the same stave, substave
            if (layer == layerA && stave == staveA && subStave == subStaveA && std::abs(module - moduleA) <= 1) {
              return true;
            }
            // Reject adjacent chips with same stave, subStave, module but different chip index
            if (layer == layerA && stave == staveA && subStave == subStaveA && module == moduleA &&std::abs(chip - chipA) <= 1) {
              return true;
            }
            return false;
          }
        );

        // Keep chip ONLY IF it is not an immediate neighbor to an existing one
        if (!isNeighborToExisting) {
          distinctChips[layer].push_back(chipIdx);
        }
      }

      // Fill histograms with properties of tracks producing multiple hits
      for (int layer = 0; layer < 2; ++layer) {
        for (const auto& [chipIdx, hitsVec] : trackData.hitsByDetector) {
          if (iotofGeom->getIOTOFLayer(chipIdx) != layer) {
            continue;
          }

          for (const auto& hitData : hitsVec) {
            if (hitData.hitIdx < 0) {
              continue; // Skip if no matching hit
            }
            const auto& hit = (*hitsPerEvent[iEvt])[hitData.hitIdx];
            PrintHit(verbose, hit, iotofGeom);

            const int type = mcTrack.isPrimary() ? 0 : 1;
            hTrackHitsXY[layer][type]->Fill(hit.GetX(), hit.GetY());
          }
        }
      }

      // Fill histograms with properties of tracks producing multiple hits
      for (int layer = 0; layer < 2; ++layer) {
        if (distinctChips[layer].size() <= 1) {
          continue;
        }

        for (const auto& [chipIdx, hitsVec] : trackData.hitsByDetector) {
          if (iotofGeom->getIOTOFLayer(chipIdx) != layer) {
            continue;
          }

          for (const auto& hitData : hitsVec) {
            if (hitData.hitIdx < 0) {
              continue; // Skip if no matching hit
            }
            const auto& hit = (*hitsPerEvent[iEvt])[hitData.hitIdx];
            PrintHit(verbose, hit, iotofGeom);

            const int type = mcTrack.isPrimary() ? 0 : 1;
            if (mcTrack.GetPt() > 5.0f) {
              hTrackDoubleHitsXY[layer][type]->Fill(hit.GetX(), hit.GetY());
            }
            hTrackDoubleHitsPhiPt[layer][type]->Fill(mcTrack.GetPhi(), mcTrack.GetPt());
          }
        }
      }
    }
  }

  Print(true, "----> Writing histograms");
  // Output
  TFile* outFile = new TFile("CheckClusters.root", "RECREATE");
  for (int type = 0; type < 2; ++type) {
    hGenEtaPt[type]->Write();
  }

  hEtaPhiHitsPrmTrkLayer0->Write();
  hEtaPhiHitsSecTrkLayer0->Write();
  hEtaPhiHitsPrmTrkLayer1->Write();
  hEtaPhiHitsSecTrkLayer1->Write();
  hTopoSummaryReal->Write();
  hTopoSummaryFake->Write();
  hTopoSummaryTotal->Write();
  hTopoSummaryDictionary->Write();
  hCountHitMatchingType->Write();

  for (int layer = 0; layer < 2; ++layer) {

    for (int type = 0; type < 2; ++type) {
      outFile->mkdir(Form("%sTrkLayer%d", trackName[type], layer));
      outFile->mkdir(Form("%sTrkLayer%d/Topologies", trackName[type], layer));
      outFile->cd(Form("%sTrkLayer%d", trackName[type], layer));

      hCountFakeClusters[layer][type]->Write("hCountFakeClusters");

      hClustersEtaPhi[layer][type]->Write("hClustersEtaPhi");
      hTrueClsSizeVsEta[layer][type]->Write("hTrueClsSizeVsEta");
      hTrueClsSizeVsPhi[layer][type]->Write("hTrueClsSizeVsPhi");
      hFakeClsSizeVsEta[layer][type]->Write("hFakeClsSizeVsEta");
      hFakeClsSizeVsPhi[layer][type]->Write("hFakeClsSizeVsPhi");

      TH2F* hEfficiency = static_cast<TH2F*>(hClustersEtaPhi[layer][type]->Clone(Form("hClusterEfficiencyVsEtaPhi%sTrkLayer%d", trackName[type], layer)));
      TH2F* hHits = layer == 0 ? (type == 0 ? hEtaPhiHitsPrmTrkLayer0 : hEtaPhiHitsSecTrkLayer0)
                              : (type == 0 ? hEtaPhiHitsPrmTrkLayer1 : hEtaPhiHitsSecTrkLayer1);
      hEfficiency->Divide(hHits);
      hEfficiency->Write("hClsEfficiency");
      delete hEfficiency;

      hNClustersFromHit[layer][type]->Write("hNClustersFromHit");
      hClsSizeVsTopo[layer][type]->Write("hClsSizeVsTopo");
      hMeanTrueClsSizeVsEta[layer][type]->Write("hMeanTrueClsSizeVsEta");
      hMeanTrueClsSizeVsPhi[layer][type]->Write("hMeanTrueClsSizeVsPhi");
      hMeanFakeClsSizeVsEta[layer][type]->Write("hMeanFakeClsSizeVsEta");
      hMeanFakeClsSizeVsPhi[layer][type]->Write("hMeanFakeClsSizeVsPhi");
      hTopoVsEta[layer][type]->Write("hTopoVsEta");
      hXRes[layer][type]->Write("hXRes");
      hYRes[layer][type]->Write("hYRes");
      hZRes[layer][type]->Write("hZRes");
      hRmsXRes[layer][type]->Write("hRmsXRes");
      hRmsYRes[layer][type]->Write("hRmsYRes");
      hRmsZRes[layer][type]->Write("hRmsZRes");
      hMeanXRes[layer][type]->Write("hMeanXRes");
      hMeanYRes[layer][type]->Write("hMeanYRes");
      hMeanZRes[layer][type]->Write("hMeanZRes");

      if (type == 0) {
        hTrackHitsXY[layer][type]->Write("hTrackHitsXY");
        hTrackDoubleHitsXY[layer][type]->Write("hTrackDoubleHitsXY");
        hTrackDoubleHitsPhiPt[layer][type]->Write("hTrackDoubleHitsPhiPt");
      }

      outFile->cd(Form("%sTrkLayer%d/Topologies", trackName[type], layer));
      for (int topo = 0; topo < kNTopologies; ++topo) hTopoVsEtaPt[layer][type][topo]->Write(Form("%sVsEtaPt", topologyNames[topo].c_str())); 
    }
  }

  // Create canvas overlapping hTrackHitsXY and hTrackDoubleHitsXY with
  // different colors in a restricted range to visualize the double hits

  TCanvas* cTrackHitsXY[2][2];
  TCanvas* cTrackHitsXYZoom[2][2];
  for (int layer = 0; layer < 2; ++layer) {
    for (int type = 0; type < 2; ++type) {
      cTrackHitsXY[layer][type] = new TCanvas(
        Form("cTrackHitsXY%sTrkLayer%d", trackName[type], layer),
        Form("Track Hits XY %s Track Layer %d", trackName[type], layer),
        800, 600
      );

      // Constrain in a box (xMin, xMax, yMin, yMax) to visualize the double hits
      if (layer == 0) {
        hTrackHitsXY[layer][type]->GetXaxis()->SetRangeUser(-22, 0);
        hTrackHitsXY[layer][type]->GetYaxis()->SetRangeUser(-22, 0);
        hTrackDoubleHitsXY[layer][type]->GetXaxis()->SetRangeUser(-22, 0);
        hTrackDoubleHitsXY[layer][type]->GetYaxis()->SetRangeUser(-22, 0);
      } else {
        hTrackHitsXY[layer][type]->GetXaxis()->SetRangeUser(-50, -20);
        hTrackHitsXY[layer][type]->GetYaxis()->SetRangeUser(-95, -75);
        hTrackDoubleHitsXY[layer][type]->GetXaxis()->SetRangeUser(-50, -20);
        hTrackDoubleHitsXY[layer][type]->GetYaxis()->SetRangeUser(-95, -75);
      }

      // First histogram: normal track hits
      hTrackHitsXY[layer][type]->SetLineColor(kBlue);
      hTrackHitsXY[layer][type]->SetLineWidth(2);
      hTrackHitsXY[layer][type]->SetFillStyle(0);

      // Draw only the histogram contours.
      hTrackHitsXY[layer][type]->Draw("CONT3");

      // Second histogram: double hits
      hTrackDoubleHitsXY[layer][type]->SetLineColor(kRed);
      hTrackDoubleHitsXY[layer][type]->SetLineWidth(2);
      hTrackDoubleHitsXY[layer][type]->SetFillStyle(0);

      // Overlay the double-hit contours.
      hTrackDoubleHitsXY[layer][type]->Draw("CONT3 SAME");

      // Don't save stats panel
      gStyle->SetOptStat(0);

      // Save
      cTrackHitsXY[layer][type]->Write();
      cTrackHitsXY[layer][type]->SaveAs(Form("cTrackHitsXY%sTrkLayer%d.pdf", trackName[type], layer));
    }
  }


  // Check digit efficiency across pixel by print the local coordinates
  // of hits without any cluster and digit associated to them
  Print(true, "----> Checking digit efficiency across pixel");
  TH2F* hNotRecoHits[2][2];
  for (int layer = 0; layer < 2; ++layer) {
    for (int type = 0; type < 2; ++type) {
      hNotRecoHits[layer][type] = new TH2F(Form("hNotRecoHits%sTrkLayer%d", trackName[type], layer), "Hits with no clusters or digits", 6000, -3, 3, 600, 3, 3);
    }
  }
  for (int iEvt = 0; iEvt < nEvts; ++iEvt) {
    for (const auto& [trackID, trackData] : allEvtsTrackData[iEvt]) {
      const auto& mcTrack = (*mcTracksPerEvent[iEvt])[trackID];
      const int type = mcTrack.isPrimary() ? 0 : 1;

      for (const auto& [chipIdx, hitsVec] : trackData.hitsByDetector) {
        int layer{-1}, stave{-1}, subStave{-1}, module{-1}, chip{-1};
        iotofGeom->getIOTOFChipId(chipIdx, layer, stave, subStave, module, chip);

        for (const auto& hitData : hitsVec) {
          if (hitData.hitIdx < 0) {
            continue; // Skip if no matching hit
          }
          const auto& hit = (*hitsPerEvent[iEvt])[hitData.hitIdx];
          if (hitData.assocClsIdxs.empty() && hitData.assocDigitIdxs.empty()) {
            Print(verbose, "Hit with no associated clusters or digits:");
            o2::math_utils::Point3D<float> avgPos;
            GetHitAvgPositionLocal(hit, iotofGeom, avgPos);
            Print(verbose, Form("Local position: x = %.5f, y = %.5f, z = %.5f", avgPos.X(), avgPos.Y(), avgPos.Z()));
            hNotRecoHits[layer][type]->Fill(avgPos.X(), avgPos.Y());
          }
        }
      }
    }
  }
  // Write digit efficiency histograms
  for (int layer = 0; layer < 2; ++layer) {
    for (int type = 0; type < 2; ++type) {
      outFile->cd(Form("%sTrkLayer%d", trackName[type], layer));
      hNotRecoHits[layer][type]->Write();
    }
  }

  outFile->Close();
  delete outFile;


  // // Print all properties of fake clusters
  // for (const auto& cluster : clusters) {
  //   if (cluster.isFakeDiffHits || cluster.isFakeDiffTrks || cluster.isFakeDiffEvts) {
  //     std::cout << "\n\n\nFake cluster properties: " << std::endl;
  //     PrintCluster(true, cluster, digitsArray, digitsLabels, hitsPerEvent, iotofGeom, segmInfo);
  //   }
  // }

}
