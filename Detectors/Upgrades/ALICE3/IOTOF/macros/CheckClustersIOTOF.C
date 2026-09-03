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
/// \brief QA macro for TF3 clusters

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <algorithm>
#include <set>

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
  o2::math_utils::Point3D<float> avgPos;
  GetHitAvgPositionGlobal(hit, avgPos);
  std::cout << "Hit: detectorID = " << hit.GetDetectorID() << ", avgPos = (" << avgPos.X() << ", "
            << avgPos.Y() << ", " << avgPos.Z() << ")" << ", layer = " << layer << ", stave = " << stave
            << ", subStave = " << subStave << ", module = " << module << ", chip = " << chip << ", trackID = "
            << hit.GetTrackID() << ", X = " << hit.GetX() << ", Y = " << hit.GetY() << ", Z = " << hit.GetZ()
            << ", time = " << hit.GetTime()
            << std::endl;
}


// Fare residuo in-chip
void GetClusterGlobalPos(const o2::iotof::Cluster& cluster,
                         TopologyInfo topoInfo,
                         o2::math_utils::Point3D<float>& globalPos,
                         o2::iotof::GeometryTGeo* iotofGeom,
                         o2::iotof::Segmentation* segmInfo){

  std::cout << "Computing cluster global position for cluster with bottom left corner at (row = " << cluster.getRow() << ", col = " << cluster.getCol() << "), chipID = " << cluster.getChipID() << std::endl;
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;
  int rowCOG = cluster.getRow() + topoInfo.mOffsetXToCOG;
  int colCOG = cluster.getCol() + topoInfo.mOffsetZToCOG;
  topoInfo.print();
  std::cout << "Cluster COG at (row = " << rowCOG << ", col = " << colCOG << ")" << std::endl;
  segmInfo->detectorToLocal(rowCOG, colCOG, x, z, cluster.getChipID());
  globalPos = iotofGeom->getMatrixL2G(cluster.getChipID())(o2::math_utils::Point3D<float>{x, 0.f, z});
}


void PrintCluster(bool verbose,
                  const o2::iotof::Cluster& cluster,
                  auto clsLabel,
                  TopologyInfo topoInfo,
                  o2::iotof::GeometryTGeo* iotofGeom,
                  o2::iotof::Segmentation* segmInfo) {
  if (!verbose) {
    return;
  }

  if (clsLabel.empty())
    return;

  o2::math_utils::Point3D<float> clsPos;
  GetClusterGlobalPos(cluster, topoInfo, clsPos, iotofGeom, segmInfo);

  int layer{-1}, stave{-1}, subStave{-1}, module{-1}, chip{-1};
  iotofGeom->getIOTOFChipId(cluster.getChipID(), layer, stave, subStave, module, chip);

  std::cout << "Cluster: chipID=" << cluster.getChipID() << ", row=" << cluster.getRow() << ", col=" << cluster.getCol()
            << ", layer=" << layer << ", stave=" << stave << ", subStave=" << subStave
            << ", module=" << module << ", chip=" << chip
            << ", rowSpan=" << cluster.getRowSpan() << ", colSpan=" << cluster.getColSpan()
            << ", size=" << cluster.getSize() << ", labels=" << clsLabel.size() << " MCCompLabels"
            << ", topology=" << cluster.getTopology() << ", time=" << cluster.getTime()
            << std::endl;
  // for (int iLabel = 0; iLabel < clsLabel.size(); ++iLabel) {
  //   const auto& evtTrackLabel = clsLabel[iLabel];
  //   if (!evtTrackLabel.isValid())
  //     continue;

  //   const int eventID = evtTrackLabel.getEventID();
  //   const int trackID = evtTrackLabel.getTrackID();
  //   std::cout << "    Contributing track to cls, label " << iLabel << ", eventID=" << eventID << ", trackID=" << trackID << std::endl;
  // }
}


template <typename... Args>
void Print(bool verbose, Args&&... args) {
  if (!verbose) {
    return;
  }

  (std::cout << ... << std::forward<Args>(args)) << std::endl;
}


void GetClusterLocalPos(const o2::iotof::Cluster& cluster,
                        TopologyInfo topoInfo,
                        o2::math_utils::Point3D<float>& localPos,
                        o2::iotof::GeometryTGeo* iotofGeom,
                        o2::iotof::Segmentation* segmInfo){

  std::cout << "Computing cluster global position for cluster with bottom left corner at (row = " << cluster.getRow() << ", col = " << cluster.getCol() << "), chipID = " << cluster.getChipID() << std::endl;
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;
  int rowCOG = cluster.getRow() + topoInfo.mOffsetXToCOG;
  int colCOG = cluster.getCol() + topoInfo.mOffsetZToCOG;
  topoInfo.print();
  std::cout << "Cluster COG at (row = " << rowCOG << ", col = " << colCOG << ")" << std::endl;
  segmInfo->detectorToLocal(rowCOG, colCOG, x, z, cluster.getChipID());
  localPos = o2::math_utils::Point3D<float>{x, 0.f, z};
}


int FindBestMatchingHit(const o2::iotof::Cluster& cluster,
                        TopologyInfo topoInfo,
                        const std::vector<int>& chipHitsIdxs,
                        const std::vector<o2::itsmft::Hit>* evtChipHits,
                        o2::iotof::GeometryTGeo* iotofGeom,
                        o2::iotof::Segmentation* segmInfo){
    int bestHitIdx = -1;
    float minDistanceSq = std::numeric_limits<float>::max();
    o2::math_utils::Point3D<float> clsPos;
    GetClusterGlobalPos(cluster, topoInfo, clsPos, iotofGeom, segmInfo);

    for (int i = 0; i < chipHitsIdxs.size(); ++i) {
        const auto& hit = (*evtChipHits)[chipHitsIdxs[i]];

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
  int nAssocPrimaries = 0;    // More than one primary MC particle from the same event is associated to the cluster
  bool isShared = false;    // More than one primary MC particle from the same event is associated to the cluster
  bool isFake = false;      // More than one primary MC particle from different events is associated to the cluster
  int hitIdx = -1;
  Topologies topology = kOther;
  uint32_t topoKey = 0;
};

struct DetectorData {
  std::vector<int> hitIndicesL0;
  std::vector<int> hitIndicesL1;
  std::vector<int> clsIndicesL0;
  std::vector<int> clsIndicesL1;
};


void CheckClustersIOTOF(std::string kinefile = "o2sim_Kine.root",
                        std::string hitfile = "o2sim_HitsTF3.root",
                        std::string clsFilePath = "tf3clusters.root",
                        std::string clsFileTopoPath = "TF3ClustersTopologies.root",
                        std::string inputGeomPath = "o2sim_geometry.root",
                        std::string geomCfgStr = "IOTOFBase.segmentedInnerTOF=true;IOTOFBase.segmentedOuterTOF=true;IOTOFBase.enableForwardTOF=false;IOTOFBase.enableBackwardTOF=false;",
                        bool verbose = false)
{
  Print(verbose, "CheckClustersTopologiesIOTOF: kinefile = ", kinefile, ", hitfile = ", hitfile, ", clsFilePath = ", clsFilePath, ", inputGeomPath = ", inputGeomPath);
  gStyle->SetPalette(55);

  o2::conf::ConfigurableParam::updateFromString(geomCfgStr);

  auto segmInfo = o2::iotof::Segmentation::Instance();

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeomPath);
  auto* iotofGeom = o2::iotof::GeometryTGeo::Instance();
  iotofGeom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  // Cluster topologies dictionary
  TFile* clsTopoFile = TFile::Open(clsFileTopoPath.data(), "READ");
  auto* clsTopoMapPtr = clsTopoFile->Get<std::unordered_map<uint32_t, o2::iotof::TopologyInfo>>("TF3ClusterTopologies");
  if (clsTopoMapPtr) {
    std::cout << "Loaded " << clsTopoMapPtr->size() << " entries from " << clsFileTopoPath << std::endl;
  } else {
    std::cerr << "Failed to load TF3ClusterTopologies from " << clsFileTopoPath << std::endl;
  }

  // Construct map directly from the vector pairs
  std::unordered_map<uint32_t, o2::iotof::TopologyInfo> topoMap(clsTopoMapPtr->begin(), clsTopoMapPtr->end());
  std::cout << "\nTopologies summary:" << std::endl;
  TopologyClassifier topoClassifier(std::move(topoMap));
  topoClassifier.print();
  std::cout << std::endl;
  clsTopoFile->Close();

  // Sorted topology map by spanRow, spanCol, and then by bitmask for better organization in the output file
  auto topologyMap = topoClassifier.getTopologyMap();
  std::vector<std::pair<uint32_t, TopologyInfo>> sortedTopoMap(topologyMap.begin(), topologyMap.end());
  std::sort(sortedTopoMap.begin(), sortedTopoMap.end(), [](const auto& a, const auto& b) {
    int topoA = a.second.mTopology;
    int topoB = b.second.mTopology;
    uint8_t spanRowA = (a.first >> 24) & 0xFF;
    uint8_t spanColA = (a.first >> 16) & 0xFF;
    uint8_t spanRowB = (b.first >> 24) & 0xFF;
    uint8_t spanColB = (b.first >> 16) & 0xFF;
    int nPixelsA = a.second.mNPixels;
    int nPixelsB = b.second.mNPixels;
    int frequencyA = a.second.mFrequency;
    int frequencyB = b.second.mFrequency;
    if (topoA != topoB) return topoA < topoB;
    if (frequencyA != frequencyB) return frequencyA > frequencyB;
    if (spanRowA != spanRowB) return spanRowA < spanRowB;
    if (spanColA != spanColB) return spanColA < spanColB;
    if (nPixelsA != nPixelsB) return nPixelsA < nPixelsB;
    return a.first < b.first; // Finally sort by bitmask if spans are equal
  });

  // Print the sorted topology map
  for (const auto& entry : sortedTopoMap) {
    const uint32_t key = entry.first;
    const TopologyInfo& topoInfo = entry.second;

    uint8_t spanRow = (key >> 24) & 0xFF;
    uint8_t spanCol = (key >> 16) & 0xFF;
    uint16_t bitmask = key & 0xFFFF;

    LOG(info) << "Key: " << key
              << ", SpanRow: " << static_cast<int>(spanRow)
              << ", SpanCol: " << static_cast<int>(spanCol)
              << ", Bitmask: " << std::bitset<16>(bitmask);
    topoInfo.print();
    LOG(info) << "";
  }

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

  // Clusters information
  TFile* clsFile = TFile::Open(clsFilePath.data());
  TTree* clustersTree = (TTree*)clsFile->Get("o2sim");
  std::vector<o2::iotof::Cluster>* clustersArray = nullptr;
  std::vector<unsigned char>* clustersPatternsArray = nullptr;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clustersLabelsArr = nullptr;

  clustersTree->SetBranchAddress("TF3Cluster", &clustersArray);
  clustersTree->SetBranchAddress("TF3ClusterPatt", &clustersPatternsArray);
  clustersTree->SetBranchAddress("TF3ClusterMCTruth", &clustersLabelsArr);

  clustersTree->GetEntry(0);
  o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> clustersLabels;

  // Store hit, cluster and MC particles properties for all tracks in all events
  std::unordered_map<uint64_t, DetectorData> tracksHitCls;

  // Load hits and MC tracks, which are stored per-event
  // Generated particles
  TH2F* hGenEtaPt[2] = {new TH2F("hGenEtaPtPrm", "Generated primary tracks;#eta;p_{T}", 40, -2, 2, 100, 0, 10),
                        new TH2F("hGenEtaPtSec", "Generated secondary tracks;#eta;p_{T}", 40, -2, 2, 100, 0, 10)};
  for (int iEvt = 0; iEvt < nEvts; ++iEvt) {
    for (int iTrack = 0; iTrack < mcTracksPerEvent[iEvt]->size(); ++iTrack) {
      const auto& mcTrack = (*mcTracksPerEvent[iEvt])[iTrack];
      // if (!mcTrack.isPrimary())
      //   continue;
      const int type = mcTrack.isPrimary() ? 0 : 1;
      hGenEtaPt[type]->Fill(mcTrack.GetEta(), mcTrack.GetPt());
      const uint64_t trackKey = (static_cast<uint64_t>(iEvt) << 32) | static_cast<uint64_t>(iTrack);
      tracksHitCls[trackKey] = DetectorData();
    }
  }

  int nHits{0}, nHitsFromPrimaryTracks{0}, nHitsFromSecondaryTracks{0};
  TH2F* hEtaPhiHitsPrmTrkLayer0 = new TH2F("hEtaPhiHitsPrmTrkLayer0", "hEtaPhiHitsPrmTrkLayer0;#phi;#eta", 64, 0, 6.28319, 40, -2, 2);
  TH2F* hEtaPhiHitsSecTrkLayer0 = new TH2F("hEtaPhiHitsSecTrkLayer0", "hEtaPhiHitsSecTrkLayer0;#phi;#eta", 64, 0, 6.28319, 40, -2, 2);
  TH2F* hEtaPhiHitsPrmTrkLayer1 = new TH2F("hEtaPhiHitsPrmTrkLayer1", "hEtaPhiHitsPrmTrkLayer1;#phi;#eta", 64, 0, 6.28319, 40, -2, 2);
  TH2F* hEtaPhiHitsSecTrkLayer1 = new TH2F("hEtaPhiHitsSecTrkLayer1", "hEtaPhiHitsSecTrkLayer1;#phi;#eta", 64, 0, 6.28319, 40, -2, 2);
  TH2F* hEtaPtHitsPrmTrkLayer0 = new TH2F("hEtaPtHitsPrmLayer0", "Generated primary tracks;#eta;p_{T}", 40, -2, 2, 50, 0, 10);
  TH2F* hEtaPtHitsSecTrkLayer0 = new TH2F("hEtaPtHitsSecLayer0", "Generated secondary tracks;#eta;p_{T}", 40, -2, 2, 50, 0, 10);
  TH2F* hEtaPtHitsPrmTrkLayer1 = new TH2F("hEtaPtHitsPrmLayer1", "Generated primary tracks;#eta;p_{T}", 40, -2, 2, 50, 0, 10);
  TH2F* hEtaPtHitsSecTrkLayer1 = new TH2F("hEtaPtHitsSecLayer1", "Generated secondary tracks;#eta;p_{T}", 40, -2, 2, 50, 0, 10);
  for (int iEvt = 0; iEvt < nEvts; ++iEvt) {
    for (int iHit = 0; iHit < hitsPerEvent[iEvt]->size(); ++iHit) {
      nHits++;
      const auto& hit = (*hitsPerEvent[iEvt])[iHit];
      const int trackID = hit.GetTrackID();
      const int chipIndex = hit.GetDetectorID();
      const uint64_t trackKey = (static_cast<uint64_t>(iEvt) << 32) | static_cast<uint64_t>(trackID);
      if (tracksHitCls.find(trackKey) == tracksHitCls.end()) {
        continue;
      }
      int layer = iotofGeom->getIOTOFLayer(hit.GetDetectorID());
      if (layer == 0) {
        tracksHitCls[trackKey].hitIndicesL0.push_back(iHit);
      } else if (layer == 1) {
        tracksHitCls[trackKey].hitIndicesL1.push_back(iHit);
      }

      auto &mcTrack = (*mcTracksPerEvent[iEvt])[trackID];
      bool isPrimary = mcTrack.isPrimary();
      if (isPrimary) nHitsFromPrimaryTracks++;
      else           nHitsFromSecondaryTracks++;
  
      float genEta   = mcTrack.GetEta();
      float genPhi   = mcTrack.GetPhi();
      float genPt    = mcTrack.GetPt();

      int hitLayer = iotofGeom->getIOTOFLayer(hit.GetDetectorID());
      if (hitLayer == 0 && isPrimary)        {
        hEtaPhiHitsPrmTrkLayer0->Fill(genPhi, genEta);
        hEtaPtHitsPrmTrkLayer0->Fill(genEta, genPt);
      }
      else if (hitLayer == 0 && !isPrimary)  {
        hEtaPhiHitsSecTrkLayer0->Fill(genPhi, genEta);
        hEtaPtHitsSecTrkLayer0->Fill(genEta, genPt);
      }
      else if (hitLayer == 1 && isPrimary)   {
        hEtaPhiHitsPrmTrkLayer1->Fill(genPhi, genEta);
        hEtaPtHitsPrmTrkLayer1->Fill(genEta, genPt);
      }
      else {
        hEtaPhiHitsSecTrkLayer1->Fill(genPhi, genEta);
        hEtaPtHitsSecTrkLayer1->Fill(genEta, genPt);
      }
    }
  }

  TH2F* hEtaPhiClsPrmTrkLayer0 = new TH2F("hEtaPhiClsPrmTrkLayer0", "hEtaPhiClsPrmTrkLayer0;#phi;#eta", 64, 0, 6.28319, 40, -2, 2);
  TH2F* hEtaPhiClsSecTrkLayer0 = new TH2F("hEtaPhiClsSecTrkLayer0", "hEtaPhiClsSecTrkLayer0;#phi;#eta", 64, 0, 6.28319, 40, -2, 2);
  TH2F* hEtaPhiClsPrmTrkLayer1 = new TH2F("hEtaPhiClsPrmTrkLayer1", "hEtaPhiClsPrmTrkLayer1;#phi;#eta", 64, 0, 6.28319, 40, -2, 2);
  TH2F* hEtaPhiClsSecTrkLayer1 = new TH2F("hEtaPhiClsSecTrkLayer1", "hEtaPhiClsSecTrkLayer1;#phi;#eta", 64, 0, 6.28319, 40, -2, 2);
  for (int iCls = 0; iCls < clustersArray->size(); ++iCls) {
    const auto& cls = (*clustersArray)[iCls];
    const auto& clsLabels = clustersLabelsArr->getLabels(iCls);
    if (clsLabels.empty())
      continue;

      // Link the cluster to the primary track
    int clsEventID{-1}, clsTrackID{-1};
    bool hasValidLabels{false};
    int nAssoc{0};
    for (int iLabel=0; iLabel<clsLabels.size(); ++iLabel) {

      const auto& label = clsLabels[iLabel];
      if (!label.isValid())
        continue;
      hasValidLabels = true;

      int iEvtID = label.getEventID();
      int iTrkID = label.getTrackID();

      auto &iMcTrack = (*mcTracksPerEvent[iEvtID])[iTrkID];

      // Do not update the track label of the cluster
      // if there are multiple labels and at least one
      // of them is a primary track
      if (!iMcTrack.isPrimary() && nAssoc > 0) {
        continue;
      }
      clsEventID = iEvtID;
      clsTrackID = iTrkID;
      nAssoc++;
    }

    if (!hasValidLabels) {
      std::cerr << "WARNING: cluster " << iCls << " has no valid labels\n";
      continue;
    }

    if (clsEventID < 0 || clsEventID >= nEvts) {
      std::cerr << "WARNING: cluster " << iCls << " has invalid eventID=" << clsEventID << "\n";
      continue;
    }

    const auto& mcTrack = (*mcTracksPerEvent[clsEventID])[clsTrackID];
    float genEta   = mcTrack.GetEta();
    float genPhi   = mcTrack.GetPhi();
    bool isPrimary = mcTrack.isPrimary();

    const uint64_t trackKey = (static_cast<uint64_t>(clsEventID) << 32) | static_cast<uint64_t>(clsTrackID);
    if (tracksHitCls.find(trackKey) == tracksHitCls.end()) {
      continue;
    }
    int clsLayer = iotofGeom->getIOTOFLayer(cls.getChipID());
    if (clsLayer == 0) {
      tracksHitCls[trackKey].clsIndicesL0.push_back(iCls);
    } else if (clsLayer == 1) {
      tracksHitCls[trackKey].clsIndicesL1.push_back(iCls);
    }

    if (clsLayer == 0 && isPrimary)        { hEtaPhiClsPrmTrkLayer0->Fill(genPhi, genEta); }
    else if (clsLayer == 0 && !isPrimary)  { hEtaPhiClsSecTrkLayer0->Fill(genPhi, genEta); }
    else if (clsLayer == 1 && isPrimary)   { hEtaPhiClsPrmTrkLayer1->Fill(genPhi, genEta); }
    else                                   { hEtaPhiClsSecTrkLayer1->Fill(genPhi, genEta); }

  }

  // Create vectors of digits with same chip index, cluster candidates
  TH2F* hCountHitMatchingType = new TH2F("hCountHitMatchingType", "hCountHitMatchingType;Hit matching type;#it{p}_{T}", 4, -0.5, 3.5, 50, 0, 10);
  hCountHitMatchingType->GetXaxis()->SetBinLabel(1, "Primary, 1 to 1");
  hCountHitMatchingType->GetXaxis()->SetBinLabel(2, "Secondary, 1 to 1");
  hCountHitMatchingType->GetXaxis()->SetBinLabel(3, "Primary, min distance");
  hCountHitMatchingType->GetXaxis()->SetBinLabel(4, "Secondary, min distance");

  std::vector<ClusterProperties> clustersProperties;
  clustersProperties.reserve(clustersArray->size()); // Pre-allocate memory
  Print(verbose, "\n\n----> Starting clusters printouts ... ");

  for (int iCls = 0; iCls < (int)clustersArray->size(); ++iCls) {

    const auto& cls = (*clustersArray)[iCls];
    const auto& clsLabels = clustersLabelsArr->getLabels(iCls);

    if (clsLabels.empty())
      continue;

    // Update with primary track if multiple labels are present
    int nAssocPrimaries{0}, nAssocSecondaries{0};
    int clsEventID{-1}, clsTrackID{-1};
    bool hasValidLabels{false};
    std::set<int> uniqueEventIDs;

    for (int iLabel=0; iLabel<clsLabels.size(); ++iLabel) {

      const auto& label = clsLabels[iLabel];

      if (!label.isValid())
        continue;
      hasValidLabels = true;

      int iEvtID = label.getEventID();
      int iTrkID = label.getTrackID();
      uniqueEventIDs.insert(iEvtID);

      auto &iMcTrack = (*mcTracksPerEvent[iEvtID])[iTrkID];
      bool isPrimary = iMcTrack.isPrimary();
      if (isPrimary) {
        nAssocPrimaries++;
      } else {
        nAssocSecondaries++;
      }

      // Do not update the track label of the cluster
      // if there are multiple labels and at least one
      // of them is a primary track
      if (!isPrimary && nAssocPrimaries > 0) {
        continue;
      }
      clsEventID = iEvtID;
      clsTrackID = iTrkID;
    }

    if (clsEventID < 0 || clsEventID >= nEvts) {
      std::cerr << "WARNING: cluster " << iCls << " has invalid eventID=" << clsEventID << "\n";
      continue;
    }

    const auto& mcTrack = (*mcTracksPerEvent[clsEventID])[clsTrackID];

    ClusterProperties clsProps;
    clsProps.clsIdx = iCls;

    // Cluster geometric properties
    clsProps.chipID   = cls.getChipID();
    clsProps.layer    = iotofGeom->getIOTOFLayer(cls.getChipID());
    clsProps.rowStart = cls.getRow();
    clsProps.rowSpan  = cls.getRowSpan();
    clsProps.colStart = cls.getCol();
    clsProps.colSpan  = cls.getColSpan();
    clsProps.pattern  = cls.getPattern();
    clsProps.size     = cls.getSize();
    clsProps.topology = static_cast<Topologies>(cls.getTopology());
    uint32_t clsTopoKey = (static_cast<uint32_t>(clsProps.rowSpan) << 24) |
                          (static_cast<uint32_t>(clsProps.colSpan) << 16) |
                           static_cast<uint32_t>(clsProps.pattern);
    clsProps.topoKey = clsTopoKey;
    TopologyInfo clsTopoInfo = topoClassifier.getTopologyFeatures(clsProps.topoKey);

    // Cluster association properties
    clsProps.eventID = clsEventID;
    clsProps.trackID = clsTrackID;
    clsProps.nAssocPrimaries = nAssocPrimaries;
    clsProps.isShared = (nAssocPrimaries > 1);
    clsProps.isFake = (uniqueEventIDs.size() > 1);
    clsProps.hitIdx = -1;
    clsProps.isPrimary = mcTrack.isPrimary();
    float genPt    = mcTrack.GetPt();

    uint64_t trackKey = (static_cast<uint64_t>(clsEventID) << 32) | static_cast<uint64_t>(clsTrackID);

    // Get hits in the chip associated to the cluster's track and event
    std::vector<int> chipHitsIdxs;
    if (clsProps.layer == 0) {
      chipHitsIdxs = tracksHitCls[trackKey].hitIndicesL0;
    } else if (clsProps.layer == 1) {
      chipHitsIdxs = tracksHitCls[trackKey].hitIndicesL1;
    }
    if (chipHitsIdxs.size() == 0) {
      std::cout << "No hits by this track and event in this chip: " << clsProps.chipID << std::endl;
      continue;
    } if (chipHitsIdxs.size() == 1) {
      clsProps.hitIdx = 0;
      hCountHitMatchingType->Fill(clsProps.isPrimary ? 0 : 2, genPt);
    } else {
      // Perform spatial matching for chips with multiple hits
      clsProps.hitIdx = FindBestMatchingHit(cls, clsTopoInfo, chipHitsIdxs, hitsPerEvent[clsProps.eventID], iotofGeom, segmInfo);
      hCountHitMatchingType->Fill(clsProps.isPrimary ? 1 : 3, genPt);
    }

    // // Print cluster information
    // PrintCluster(verbose, cluster, digitsArray, digitsLabels, hitsPerEvent, iotofGeom, segmInfo);
    clustersProperties.push_back(clsProps);

  }
  Print(true, "----> Total number of clusters: ", clustersProperties.size());

  // Print features of all clusters and hits of the particles in the events
  for (const auto& trackProperties : tracksHitCls) {
    const uint64_t trackKey = trackProperties.first;
    const auto eventID = static_cast<int>(trackKey >> 32);
    const auto trackID = static_cast<int>(trackKey & 0xFFFFFFFF);
    Print(verbose, "\n\n");
    PrintMcTrack(verbose, (*mcTracksPerEvent[eventID])[trackID]); 

    Print(verbose, "Layer 0");
    for (const auto& hitIdx : trackProperties.second.hitIndicesL0) {
      const auto& hit = (*hitsPerEvent[eventID])[hitIdx];
      PrintHit(verbose, hit, iotofGeom);
    }
    for (const auto& clsIdx : trackProperties.second.clsIndicesL0) {
      const auto& cls = (*clustersArray)[clsIdx];
      const auto& clsLabels = clustersLabelsArr->getLabels(clsIdx);
      uint32_t clsTopoKey = (static_cast<uint32_t>(cls.getRowSpan()) << 24) |
                            (static_cast<uint32_t>(cls.getColSpan()) << 16) |
                             static_cast<uint32_t>(cls.getPattern());
      TopologyInfo clsTopoInfo = topoClassifier.getTopologyFeatures(clsTopoKey);
      PrintCluster(verbose, cls, clsLabels, clsTopoInfo, iotofGeom, segmInfo);
    }
    Print(verbose, "Layer 1");
    for (const auto& hitIdx : trackProperties.second.hitIndicesL1) {
      const auto& hit = (*hitsPerEvent[eventID])[hitIdx];
      PrintHit(verbose, hit, iotofGeom);
    }
    for (const auto& clsIdx : trackProperties.second.clsIndicesL1) {
      const auto& cls = (*clustersArray)[clsIdx];
      const auto& clsLabels = clustersLabelsArr->getLabels(clsIdx);
      uint32_t clsTopoKey = (static_cast<uint32_t>(cls.getRowSpan()) << 24) |
                            (static_cast<uint32_t>(cls.getColSpan()) << 16) |
                             static_cast<uint32_t>(cls.getPattern());
      TopologyInfo clsTopoInfo = topoClassifier.getTopologyFeatures(clsTopoKey);
      PrintCluster(verbose, cls, clsLabels, clsTopoInfo, iotofGeom, segmInfo);
    }
  }

  // Debug prints
  std::cout << "\n***********************************" << std::endl;
  Print(true, "Number of events:   ", nEvts);
  Print(true, "Number of hits:     ", nHits);
  Print(true, "-> from primary tracks:   ", nHitsFromPrimaryTracks);
  Print(true, "-> from secondary tracks: ", nHitsFromSecondaryTracks);
  Print(true, "Number of clusters:   ", clustersArray->size());
  Print(true, "Number of clusters labels:  ", clustersLabelsArr->getNElements());
  Print(true, "Number of entries in cluster tree: ", clustersTree->GetEntries());
  std::cout << "***********************************\n" << std::endl;

  // QA printouts and histograms
  Print(true, "\n\n----> Starting QA logging ... ");
  const char* trackName[2] = {"Prm", "Sec"};

  // Output
  TFile* outFile = new TFile("CheckClusters.root", "RECREATE");
  for (int type = 0; type < 2; ++type) {
    hGenEtaPt[type]->Write();
  }

  hEtaPhiHitsPrmTrkLayer0->Write();
  hEtaPhiHitsSecTrkLayer0->Write();
  hEtaPhiHitsPrmTrkLayer1->Write();
  hEtaPhiHitsSecTrkLayer1->Write();
  hEtaPtHitsPrmTrkLayer0->Write();
  hEtaPtHitsSecTrkLayer0->Write();
  hEtaPtHitsPrmTrkLayer1->Write();
  hEtaPtHitsSecTrkLayer1->Write();
  hEtaPhiClsPrmTrkLayer0->Write();
  hEtaPhiClsSecTrkLayer0->Write();
  hEtaPhiClsPrmTrkLayer1->Write();
  hEtaPhiClsSecTrkLayer1->Write();


  // Count fake clusters
  TH1F* hCountClsTypes[2][2];
  for (int layer = 0; layer < 2; ++layer) {
    for (int type = 0; type < 2; ++type) {
      hCountClsTypes[layer][type] = new TH1F(Form("hCountClsTypes%sTrkLayer%d", trackName[type], layer), Form("Cluster Counter %s Trk Layer %d", trackName[type], layer), 4, -0.5, 3.5);
      hCountClsTypes[layer][type]->GetXaxis()->SetBinLabel(1, "Total");
      hCountClsTypes[layer][type]->GetXaxis()->SetBinLabel(2, "Real");
      hCountClsTypes[layer][type]->GetXaxis()->SetBinLabel(3, "Shared");
      hCountClsTypes[layer][type]->GetXaxis()->SetBinLabel(4, "Fake");
    }
  }

  // Loop over clusters and fill histograms
  for (const auto& cluster : clustersProperties) {
    int layer = cluster.layer;
    int type = cluster.isPrimary ? 0 : 1;
    hCountClsTypes[layer][type]->Fill(0.f, 1); // Total clusters
    if (cluster.isShared) {
      hCountClsTypes[layer][type]->Fill(2.f, 1); // Shared clusters
      continue;
    }
    if (cluster.isFake) {
      hCountClsTypes[layer][type]->Fill(3.f, 1); // Fake clusters
      continue;
    }
    hCountClsTypes[layer][type]->Fill(1.f, 1); // Real clusters
  }

  // Topology names
  const std::array<std::string, kNTopologies> topologyNames = {
      "kSingleDigit", "kLineOnRow", "kLineOnCol", "kSquare", "kRectangle", "kDiagonal",
      "kLowerTriangleLeft", "kLowerTriangleRight", "kUpperTriangleLeft", "kUpperTriangleRight",
      "kSnake", "kSnakeRefl", "kSnakeRot90", "kSnakeRot90Refl", "kHuge", "kOther"};

  // Count topologies from frequency values in 
  // topologies dictionary and fill the summary histograms
  TH1F* hTopoSummaryDictionary = new TH1F("hTopoSummaryDictionary", "Cluster Topology Count Summary;;Counts", kNTopologies, 0, kNTopologies);
  for (const auto& [topoKey, topology] : topoClassifier.getTopologyMap()) {
    hTopoSummaryDictionary->Fill(topology.mTopology, topology.mFrequency);
  }

  TH2F *hTrueClsSizeVsEta[2][2], *hTrueClsSizeVsPhi[2][2], *hFakeClsSizeVsEta[2][2], *hFakeClsSizeVsPhi[2][2],
       *hClustersEtaPhi[2][2], *hTopoVsEta[2][2], *hClsSizeVsTopo[2][2],
       *hXResVsEta[2][2], *hYResVsEta[2][2], *hZResVsEta[2][2], *hXResVsTopo[2][2], *hYResVsTopo[2][2], *hZResVsTopo[2][2],
       *hTrackHitsXY[2][2], *hTrackDoubleHitsXY[2][2], *hTrackDoubleHitsPhiPt[2][2], *hTopoVsEtaPt[2][2][kNTopologies],
       *hRecoClsEtaPt[2][2];
  TH1F *hMeanTrueClsSizeVsEta[2][2], *hMeanTrueClsSizeVsPhi[2][2], *hMeanFakeClsSizeVsEta[2][2], *hMeanFakeClsSizeVsPhi[2][2],
       *hRmsXResVsEta[2][2], *hRmsYResVsEta[2][2], *hRmsZResVsEta[2][2], *hMeanXResVsEta[2][2], *hMeanYResVsEta[2][2], *hMeanZResVsEta[2][2],
       *hRmsXResVsTopo[2][2], *hRmsYResVsTopo[2][2], *hRmsZResVsTopo[2][2], *hMeanXResVsTopo[2][2], *hMeanYResVsTopo[2][2], *hMeanZResVsTopo[2][2];
  TH1F* hTopoSummaryTotal = new TH1F("hTopoSummaryTotal", "Cluster Topology Summary;;Counts", kNTopologies, 0, kNTopologies);
  TH1F* hTopoSummaryReal = new TH1F("hTopoSummaryReal", "Cluster Topology Summary;;Counts", kNTopologies, 0, kNTopologies);
  TH1F* hTopoSummaryFake = new TH1F("hTopoSummaryFake", "Cluster Topology Summary;;Counts", kNTopologies, 0, kNTopologies);

  for (int layer = 0; layer < 2; ++layer) {
    for (int type = 0; type < 2; ++type) {
      hClustersEtaPhi[layer][type] = new TH2F(Form("hNClsVsEtaPhi%sTrkLayer%d", trackName[type], layer), "Cluster #eta vs #phi;#phi;#eta", 64, 0, 6.28319, 40, -2, 2);
      hTrueClsSizeVsEta[layer][type] = new TH2F(Form("hTrueClsSizeVsEta%sTrkLayer%d", trackName[type], layer), "True Cluster Size vs #eta;#eta", 40, -2, 2, 20, 0.5, 20.5);
      hTrueClsSizeVsPhi[layer][type] = new TH2F(Form("hTrueClsSizeVsPhi%sTrkLayer%d", trackName[type], layer), "True Cluster Size vs #phi;#phi", 64, 0, 6.28319, 20, 0.5, 20.5);
      hFakeClsSizeVsEta[layer][type] = new TH2F(Form("hFakeClsSizeVsEta%sTrkLayer%d", trackName[type], layer), "Fake Cluster Size vs #eta;#eta", 40, -2, 2, 20, 0.5, 20.5);
      hFakeClsSizeVsPhi[layer][type] = new TH2F(Form("hFakeClsSizeVsPhi%sTrkLayer%d", trackName[type], layer), "Fake Cluster Size vs #phi;#phi", 64, 0, 6.28319, 20, 0.5, 20.5);
      hMeanTrueClsSizeVsEta[layer][type] = new TH1F(Form("hMeanTrueClsSizeVsEta%sTrkLayer%d", trackName[type], layer), "Mean True Cluster Size vs #eta;#eta", 40, -2, 2);
      hMeanTrueClsSizeVsPhi[layer][type] = new TH1F(Form("hMeanTrueClsSizeVsPhi%sTrkLayer%d", trackName[type], layer), "Mean True Cluster Size vs #phi;#phi", 64, 0, 6.28319);
      hMeanFakeClsSizeVsEta[layer][type] = new TH1F(Form("hMeanFakeClsSizeVsEta%sTrkLayer%d", trackName[type], layer), "Mean Fake Cluster Size vs #eta;#eta", 40, -2, 2);
      hMeanFakeClsSizeVsPhi[layer][type] = new TH1F(Form("hMeanFakeClsSizeVsPhi%sTrkLayer%d", trackName[type], layer), "Mean Fake Cluster Size vs #phi;#phi", 64, 0, 6.28319);
      hRecoClsEtaPt[layer][type] = new TH2F(Form("hRecoClsEtaPt%sTrkLayer%d", trackName[type], layer), "Reconstructed Cluster vs p_{T};#eta;p_{T}", 40, -2, 2, 50, 0, 10);
      hTopoVsEta[layer][type] = new TH2F(Form("hClsSizeVsEtaTopo%sTrkLayer%d", trackName[type], layer), "Cluster Topology vs #eta;;#eta", kNTopologies, 0, kNTopologies, 40, -2, 2);
      hClsSizeVsTopo[layer][type] = new TH2F(Form("hClsSizeVsTopo%sTrkLayer%d", trackName[type], layer), "Cluster Topology vs N Digits;;N Digits", kNTopologies, 0, kNTopologies, 20, 0.5, 20.5);
      hXResVsEta[layer][type] = new TH2F(Form("hDeltaXClsHit%sTrkLayer%d", trackName[type], layer), ";#Delta X;#eta", 1000, -2, 2, 40, -2, 2);
      hYResVsEta[layer][type] = new TH2F(Form("hDeltaYClsHit%sTrkLayer%d", trackName[type], layer), ";#Delta Y;#eta", 1000, -2, 2, 40, -2, 2);
      hZResVsEta[layer][type] = new TH2F(Form("hDeltaZClsHit%sTrkLayer%d", trackName[type], layer), ";#Delta Z;#eta", 1000, -2, 2, 40, -2, 2);
      hRmsXResVsEta[layer][type] = new TH1F(Form("hRmsDeltaXClsHit%sTrkLayer%d", trackName[type], layer), ";#eta;RMS #Delta X", 40, -2, 2);
      hRmsYResVsEta[layer][type] = new TH1F(Form("hRmsDeltaYClsHit%sTrkLayer%d", trackName[type], layer), ";#eta;RMS #Delta Y", 40, -2, 2);
      hRmsZResVsEta[layer][type] = new TH1F(Form("hRmsDeltaZClsHit%sTrkLayer%d", trackName[type], layer), ";#eta;RMS #Delta Z", 40, -2, 2);
      hMeanXResVsEta[layer][type] = new TH1F(Form("hMeanDeltaXClsHit%sTrkLayer%d", trackName[type], layer), ";#eta;Mean #Delta X", 40, -2, 2);
      hMeanYResVsEta[layer][type] = new TH1F(Form("hMeanDeltaYClsHit%sTrkLayer%d", trackName[type], layer), ";#eta;Mean #Delta Y", 40, -2, 2);
      hMeanZResVsEta[layer][type] = new TH1F(Form("hMeanDeltaZClsHit%sTrkLayer%d", trackName[type], layer), ";#eta;Mean #Delta Z", 40, -2, 2);
      hXResVsTopo[layer][type] = new TH2F(Form("hDeltaXClsHit%sTrkLayer%d", trackName[type], layer), ";#Delta X;Cluster Topology", 1000, -2, 2, kNTopologies, 0, kNTopologies);
      hYResVsTopo[layer][type] = new TH2F(Form("hDeltaYClsHit%sTrkLayer%d", trackName[type], layer), ";#Delta Y;Cluster Topology", 1000, -2, 2, kNTopologies, 0, kNTopologies);
      hZResVsTopo[layer][type] = new TH2F(Form("hDeltaZClsHit%sTrkLayer%d", trackName[type], layer), ";#Delta Z;Cluster Topology", 1000, -2, 2, kNTopologies, 0, kNTopologies);
      hRmsXResVsTopo[layer][type] = new TH1F(Form("hRmsDeltaXClsHit%sTrkLayer%d", trackName[type], layer), ";Cluster Topology;RMS #Delta X", kNTopologies, 0, kNTopologies);
      hRmsYResVsTopo[layer][type] = new TH1F(Form("hRmsDeltaYClsHit%sTrkLayer%d", trackName[type], layer), ";Cluster Topology;RMS #Delta Y", kNTopologies, 0, kNTopologies);
      hRmsZResVsTopo[layer][type] = new TH1F(Form("hRmsDeltaZClsHit%sTrkLayer%d", trackName[type], layer), ";Cluster Topology;RMS #Delta Z", kNTopologies, 0, kNTopologies);
      hMeanXResVsTopo[layer][type] = new TH1F(Form("hMeanDeltaXClsHit%sTrkLayer%d", trackName[type], layer), ";Cluster Topology;Mean #Delta X", kNTopologies, 0, kNTopologies);
      hMeanYResVsTopo[layer][type] = new TH1F(Form("hMeanDeltaYClsHit%sTrkLayer%d", trackName[type], layer), ";Cluster Topology;Mean #Delta Y", kNTopologies, 0, kNTopologies);
      hMeanZResVsTopo[layer][type] = new TH1F(Form("hMeanDeltaZClsHit%sTrkLayer%d", trackName[type], layer), ";Cluster Topology;Mean #Delta Z", kNTopologies, 0, kNTopologies);

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
        hTopoVsEtaPt[layer][type][topo] = new TH2F(Form("h%sVsEtaPt_%sTrk_TrkLayer%d", topologyNames[topo].c_str(), trackName[type], layer), Form("Cluster Topology %s vs Eta and Pt;#eta;p_{T}", topologyNames[topo].c_str()), 40, -2, 2, 20, 0, 10);
      }
    }
  }

  // Check digit efficiency across pixel by print the local coordinates
  // of hits without any cluster and digit associated to them
  TH2F* hNotRecoHits[2][2];
  for (int l = 0; l < 2; ++l) {
    for (int t = 0; t < 2; ++t) {
      hNotRecoHits[l][t] = new TH2F(
        Form("hNotRecoHits_layer%d_type%d", l, t),
        Form("Local Position Unreconstructed Hits L%d Type%d;x (cm);y (cm)", l, t),
        1000, -2., 2., // Adjust binning/ranges to your sensor dimensions
        1000, -2., 2.
      );
    }
  }

  for (const auto& trackProperties : tracksHitCls) {
    int eventID = static_cast<int>(trackProperties.first >> 32);
    int trackID = static_cast<int>(trackProperties.first & 0xFFFFFFFF);
    const auto& mcTrack = (*mcTracksPerEvent[eventID])[trackID];

    if (!mcTrack.isPrimary()) { continue; }
    const int type = 0;

    if (trackProperties.second.hitIndicesL0.empty() && trackProperties.second.hitIndicesL1.empty()) {
      continue; // Skip tracks without hits in both layers
    }

    // Helper lambda to check if two chips are adjacent
    auto areChipsAdjacent = [&](int chipID1, int chipID2) -> bool {
      int l1{-1}, s1{-1}, ss1{-1}, m1{-1}, c1{-1};
      int l2{-1}, s2{-1}, ss2{-1}, m2{-1}, c2{-1};

      iotofGeom->getIOTOFChipId(chipID1, l1, s1, ss1, m1, c1);
      iotofGeom->getIOTOFChipId(chipID2, l2, s2, ss2, m2, c2);

      bool sameLayer = (l1 == l2);
      bool sameStave = (s1 == s2);
      bool sameSubStave = (ss1 == ss2);
      bool sameModule = (m1 == m2);
      bool sameChip = (c1 == c2);

      // // Same chip check
      // if (sameLayer && sameStave && sameSubStave && sameModule && sameChip) {
      //   return true;
      // }
      // Adjacent module check on same stave & substave
      if (sameLayer && sameStave && sameSubStave && std::abs(m1 - m2) <= 1) {
        return true;
      }
      // Adjacent chip check on same module
      if (sameLayer && sameStave && sameSubStave && sameModule && std::abs(c1 - c2) <= 1) {
        return true;
      }
      return false;
    };

    // Helper lambda to check a layer's cluster list
    auto hasNonAdjacentDoubleClusters = [&](const std::vector<int>& hitIndices) -> bool {
      if (hitIndices.size() < 2) return false;

      // Collect unique chip IDs for this layer
      std::vector<int> chips;
      for (int hitIdx : hitIndices) {
        chips.push_back((*hitsPerEvent[eventID])[hitIdx].GetDetectorID());
      }
      std::sort(chips.begin(), chips.end());
      chips.erase(std::unique(chips.begin(), chips.end()), chips.end());

      if (chips.size() < 2) return false; // All clusters are on the exact same chip

      // Check if ANY pair of chips is non-adjacent
      for (size_t i = 0; i < chips.size(); ++i) {
        for (size_t j = i + 1; j < chips.size(); ++j) {
          if (!areChipsAdjacent(chips[i], chips[j])) {
            return true; // Found double clusters on non-adjacent chips!
          }
        }
      }
      return false;
    };

    // Evaluate for L0 and L1 directly using DetectorData
    bool hasDoubleClustersL0 = hasNonAdjacentDoubleClusters(trackProperties.second.hitIndicesL0);
    for (const auto& hitIdx : trackProperties.second.hitIndicesL0) {
      const auto& hit = (*hitsPerEvent[eventID])[hitIdx];
      PrintHit(verbose, hit, iotofGeom);
      hTrackHitsXY[0][type]->Fill(hit.GetX(), hit.GetY());

      // Check for non-reconstructed hits
      if (trackProperties.second.clsIndicesL0.empty()) {
        o2::math_utils::Point3D<float> avgPos;
        GetHitAvgPositionLocal(hit, iotofGeom, avgPos);
        hNotRecoHits[0][type]->Fill(avgPos.X(), avgPos.Z());
      }
      if (hasDoubleClustersL0) {
        hTrackDoubleHitsPhiPt[0][type]->Fill(mcTrack.GetPhi(), mcTrack.GetPt());
        if (mcTrack.GetPt() > 5.0f) {
          hTrackDoubleHitsXY[0][type]->Fill(hit.GetX(), hit.GetY());
        }
      }
    }
    bool hasDoubleClustersL1 = hasNonAdjacentDoubleClusters(trackProperties.second.hitIndicesL1);
    for (const auto& hitIdx : trackProperties.second.hitIndicesL1) {
      const auto& hit = (*hitsPerEvent[eventID])[hitIdx];
      PrintHit(verbose, hit, iotofGeom);
      hTrackHitsXY[1][type]->Fill(hit.GetX(), hit.GetY());

      // Check for non-reconstructed hits
      if (trackProperties.second.clsIndicesL1.empty()) {
        o2::math_utils::Point3D<float> avgPos;
        GetHitAvgPositionLocal(hit, iotofGeom, avgPos);
        hNotRecoHits[1][type]->Fill(avgPos.X(), avgPos.Z());
      }
      if (hasDoubleClustersL1) {
        hTrackDoubleHitsPhiPt[1][type]->Fill(mcTrack.GetPhi(), mcTrack.GetPt());
        if (mcTrack.GetPt() > 5.0f) {
          hTrackDoubleHitsXY[1][type]->Fill(hit.GetX(), hit.GetY());
        }
      }
    }
  } // end event loop

  // Loop over clusters
  Print(true, "----> Looping over clusters and filling histograms");
  std::cout << "\n\n\n\n\n\n\n" << std::endl;
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
      continue; // Skip clusters without a matching hit
    }

    hTopoSummaryReal->Fill(topo);
    hTrueClsSizeVsEta[layer][type]->Fill(eta, size);
    hTrueClsSizeVsPhi[layer][type]->Fill(phi, size);
    float weight = cls.isShared ? cls.nAssocPrimaries : 1.0f; // Weight for shared clusters
    hClustersEtaPhi[layer][type]->Fill(phi, eta, weight);
    hRecoClsEtaPt[layer][type]->Fill(eta, pt, weight);

    int hitIdx{-1};
    uint64_t trackKey = (static_cast<uint64_t>(eventID) << 32) | static_cast<uint64_t>(trackID);
    if (layer == 0) {
      hitIdx = tracksHitCls[trackKey].hitIndicesL0[cls.hitIdx];
    } else if (layer == 1) {
      hitIdx = tracksHitCls[trackKey].hitIndicesL1[cls.hitIdx];
    }
    auto& hit = (*hitsPerEvent[cls.eventID])[hitIdx];

    o2::math_utils::Point3D<float> clusterPos;
    TopologyInfo clsTopoInfo = topoClassifier.getTopologyFeatures(cls.topoKey);
    auto clsFull = clustersArray->at(cls.clsIdx);
    GetClusterLocalPos(clsFull, clsTopoInfo, clusterPos, iotofGeom, segmInfo);
    o2::math_utils::Point3D<float> avgPos;
    GetHitAvgPositionLocal(hit, iotofGeom, avgPos);
    // if (clusterPos.X() - avgPos.X() > 1) {

    // }
    // PrintHit(true, hit, iotofGeom);
    // std::cout << "Hit average position " << avgPos.X() << ", " << avgPos.Y() << ", " << avgPos.Z() << std::endl;
    // std::cout << std::endl;
    hXResVsEta[layer][type]->Fill(clusterPos.X() - avgPos.X(), eta);
    hYResVsEta[layer][type]->Fill(clusterPos.Y() - avgPos.Y(), eta);
    hZResVsEta[layer][type]->Fill(clusterPos.Z() - avgPos.Z(), eta);
    hXResVsTopo[layer][type]->Fill(clusterPos.X() - avgPos.X(), topo);
    hYResVsTopo[layer][type]->Fill(clusterPos.Y() - avgPos.Y(), topo);
    hZResVsTopo[layer][type]->Fill(clusterPos.Z() - avgPos.Z(), topo);
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
      for (int etaBin = 1; etaBin <= hXResVsEta[layer][type]->GetNbinsY(); ++etaBin) {
        TH1D* hXResProjVsEta = hXResVsEta[layer][type]->ProjectionX(Form("hXResProj_etaBin%d", etaBin), etaBin, etaBin);
        TH1D* hYResProjVsEta = hYResVsEta[layer][type]->ProjectionX(Form("hYResProj_etaBin%d", etaBin), etaBin, etaBin);
        TH1D* hZResProjVsEta = hZResVsEta[layer][type]->ProjectionX(Form("hZResProj_etaBin%d", etaBin), etaBin, etaBin);
        hRmsXResVsEta[layer][type]->SetBinContent(etaBin, hXResProjVsEta->GetRMS());
        hRmsYResVsEta[layer][type]->SetBinContent(etaBin, hYResProjVsEta->GetRMS());
        hRmsZResVsEta[layer][type]->SetBinContent(etaBin, hZResProjVsEta->GetRMS());
        hRmsXResVsEta[layer][type]->SetBinError(etaBin, hXResProjVsEta->GetRMSError());
        hRmsYResVsEta[layer][type]->SetBinError(etaBin, hYResProjVsEta->GetRMSError());
        hRmsZResVsEta[layer][type]->SetBinError(etaBin, hZResProjVsEta->GetRMSError());
        hMeanXResVsEta[layer][type]->SetBinContent(etaBin, hXResProjVsEta->GetMean());
        hMeanYResVsEta[layer][type]->SetBinContent(etaBin, hYResProjVsEta->GetMean());
        hMeanZResVsEta[layer][type]->SetBinContent(etaBin, hZResProjVsEta->GetMean());
        hMeanXResVsEta[layer][type]->SetBinError(etaBin, hXResProjVsEta->GetMeanError());
        hMeanYResVsEta[layer][type]->SetBinError(etaBin, hYResProjVsEta->GetMeanError());
        hMeanZResVsEta[layer][type]->SetBinError(etaBin, hZResProjVsEta->GetMeanError());
      }
      for (int topoBin = 1; topoBin <= hXResVsTopo[layer][type]->GetNbinsY(); ++topoBin) {
        TH1D* hXResProjVsTopo = hXResVsTopo[layer][type]->ProjectionX(Form("hXResProj_topoBin%d", topoBin), topoBin, topoBin);
        TH1D* hYResProjVsTopo = hYResVsTopo[layer][type]->ProjectionX(Form("hYResProj_topoBin%d", topoBin), topoBin, topoBin);
        TH1D* hZResProjVsTopo = hZResVsTopo[layer][type]->ProjectionX(Form("hZResProj_topoBin%d", topoBin), topoBin, topoBin);
        hRmsXResVsTopo[layer][type]->SetBinContent(topoBin, hXResProjVsTopo->GetRMS());
        hRmsYResVsTopo[layer][type]->SetBinContent(topoBin, hYResProjVsTopo->GetRMS());
        hRmsZResVsTopo[layer][type]->SetBinContent(topoBin, hZResProjVsTopo->GetRMS());
        hRmsXResVsTopo[layer][type]->SetBinError(topoBin, hXResProjVsTopo->GetRMSError());
        hRmsYResVsTopo[layer][type]->SetBinError(topoBin, hYResProjVsTopo->GetRMSError());
        hRmsZResVsTopo[layer][type]->SetBinError(topoBin, hZResProjVsTopo->GetRMSError());
        hMeanXResVsTopo[layer][type]->SetBinContent(topoBin, hXResProjVsTopo->GetMean());
        hMeanYResVsTopo[layer][type]->SetBinContent(topoBin, hYResProjVsTopo->GetMean());
        hMeanZResVsTopo[layer][type]->SetBinContent(topoBin, hZResProjVsTopo->GetMean());
        hMeanXResVsTopo[layer][type]->SetBinError(topoBin, hXResProjVsTopo->GetMeanError());
        hMeanYResVsTopo[layer][type]->SetBinError(topoBin, hYResProjVsTopo->GetMeanError());
        hMeanZResVsTopo[layer][type]->SetBinError(topoBin, hZResProjVsTopo->GetMeanError());
      }
    }
  }

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

      hCountClsTypes[layer][type]->Write("hCountClsTypes");

      hClustersEtaPhi[layer][type]->Write("hClustersEtaPhi");
      hTrueClsSizeVsEta[layer][type]->Write("hTrueClsSizeVsEta");
      hTrueClsSizeVsPhi[layer][type]->Write("hTrueClsSizeVsPhi");
      hFakeClsSizeVsEta[layer][type]->Write("hFakeClsSizeVsEta");
      hFakeClsSizeVsPhi[layer][type]->Write("hFakeClsSizeVsPhi");

      TH2F* hEffEtaPhi = static_cast<TH2F*>(hClustersEtaPhi[layer][type]->Clone(Form("hClsEffVsEtaPhi%sTrkLayer%d", trackName[type], layer)));
      TH2F* hEtaPhiHits = layer == 0 ? (type == 0 ? hEtaPhiHitsPrmTrkLayer0 : hEtaPhiHitsSecTrkLayer0)
                                     : (type == 0 ? hEtaPhiHitsPrmTrkLayer1 : hEtaPhiHitsSecTrkLayer1);
      hEffEtaPhi->Divide(hEtaPhiHits);
      hEffEtaPhi->Write("hClsEffEtaPhi");
      delete hEffEtaPhi;

      TH2F* hEffEtaPt = static_cast<TH2F*>(hRecoClsEtaPt[layer][type]->Clone(Form("hClsEffVsEtaPhi%sTrkLayer%d", trackName[type], layer)));
      TH2F* hEtaPtHits = layer == 0 ? (type == 0 ? hEtaPtHitsPrmTrkLayer0 : hEtaPtHitsSecTrkLayer0)
                                    : (type == 0 ? hEtaPtHitsPrmTrkLayer1 : hEtaPtHitsSecTrkLayer1);
      hEffEtaPt->Divide(hEtaPtHits);
      hEffEtaPt->Write("hClsEffEtaPt");
      delete hEffEtaPt;

      // Compute Efficiency vs Eta
      TH1D* hClustersEta = hClustersEtaPhi[layer][type]->ProjectionY(Form("hClsEta_%sTrkLayer%d", trackName[type], layer));
      TH1D* hHitsEta     = hEtaPhiHits->ProjectionY(Form("hHitsEta_%sTrkLayer%d", trackName[type], layer));

      TH1F* hEffVsEta = static_cast<TH1F*>(hClustersEta->Clone(Form("hClsEffVsEta%sTrkLayer%d", trackName[type], layer)));
      hEffVsEta->Divide(hHitsEta);

      // Compute proper binomial uncertainties
      for (int bin = 1; bin <= hEffVsEta->GetNbinsX(); ++bin) {
        double eff   = hEffVsEta->GetBinContent(bin);
        double nHits = hHitsEta->GetBinContent(bin);

        if (nHits > 0) {
          // Clamp eff between 0 and 1 to prevent sqrt of negative numbers due to numerical precision
          eff = std::clamp(eff, 0.0, 1.0);
          double err = std::sqrt(eff * (1.0 - eff) / nHits);
          hEffVsEta->SetBinError(bin, err);
        } else {
          hEffVsEta->SetBinError(bin, 0);
        }
      }
      hEffVsEta->Write("hClsEffVsEta");

      TH1D* hClustersPt = hRecoClsEtaPt[layer][type]->ProjectionY(Form("hClsPt_%sTrkLayer%d", trackName[type], layer));
      TH1D* hHitsPt     = hEtaPtHits->ProjectionY(Form("hHitsPt_%sTrkLayer%d", trackName[type], layer));

      TH1F* hEffVsPt = static_cast<TH1F*>(hClustersPt->Clone(Form("hClsEffVsPt%sTrkLayer%d", trackName[type], layer)));
      hEffVsPt->Divide(hHitsPt);
      // Compute proper binomial uncertainties
      for (int bin = 1; bin <= hEffVsPt->GetNbinsX(); ++bin) {
        double eff   = hEffVsPt->GetBinContent(bin);
        double nHits = hHitsPt->GetBinContent(bin);

        if (nHits > 0) {
          // Clamp eff between 0 and 1 to prevent sqrt of negative numbers due to numerical precision
          eff = std::clamp(eff, 0.0, 1.0);
          double err = std::sqrt(eff * (1.0 - eff) / nHits);
          hEffVsPt->SetBinError(bin, err);
        } else {
          hEffVsPt->SetBinError(bin, 0);
        }
      }
      hEffVsPt->Write("hClsEffVsPt");

      // Compute Efficiency vs Phi
      TH1D* hClustersPhi = hClustersEtaPhi[layer][type]->ProjectionX(Form("hClsPhi_%sTrkLayer%d", trackName[type], layer));
      TH1D* hHitsPhi     = hEtaPhiHits->ProjectionX(Form("hHitsPhi_%sTrkLayer%d", trackName[type], layer));

      TH1F* hEffVsPhi = static_cast<TH1F*>(hClustersPhi->Clone(Form("hClsEffVsPhi%sTrkLayer%d", trackName[type], layer)));
      hEffVsPhi->Divide(hHitsPhi);

      // Compute proper binomial uncertainties
      for (int bin = 1; bin <= hEffVsPhi->GetNbinsX(); ++bin) {
        double eff   = hEffVsPhi->GetBinContent(bin);
        double nHits = hHitsPhi->GetBinContent(bin);

        if (nHits > 0) {
          eff = std::clamp(eff, 0.0, 1.0);
          double err = std::sqrt(eff * (1.0 - eff) / nHits);
          hEffVsPhi->SetBinError(bin, err);
        } else {
          hEffVsPhi->SetBinError(bin, 0);
        }
      }
      hEffVsPhi->Write("hClsEffVsPhi");

      hClsSizeVsTopo[layer][type]->Write("hClsSizeVsTopo");
      hMeanTrueClsSizeVsEta[layer][type]->Write("hMeanTrueClsSizeVsEta");
      hMeanTrueClsSizeVsPhi[layer][type]->Write("hMeanTrueClsSizeVsPhi");
      hMeanFakeClsSizeVsEta[layer][type]->Write("hMeanFakeClsSizeVsEta");
      hMeanFakeClsSizeVsPhi[layer][type]->Write("hMeanFakeClsSizeVsPhi");
      hTopoVsEta[layer][type]->Write("hTopoVsEta");
      hXResVsEta[layer][type]->Write("hXResVsEta");
      hYResVsEta[layer][type]->Write("hYResVsEta");
      hZResVsEta[layer][type]->Write("hZResVsEta");
      hRmsXResVsEta[layer][type]->Write("hRmsXResVsEta");
      hRmsYResVsEta[layer][type]->Write("hRmsYResVsEta");
      hRmsZResVsEta[layer][type]->Write("hRmsZResVsEta");
      hMeanXResVsEta[layer][type]->Write("hMeanXResVsEta");
      hMeanYResVsEta[layer][type]->Write("hMeanYResVsEta");
      hMeanZResVsEta[layer][type]->Write("hMeanZResVsEta");
      hXResVsTopo[layer][type]->Write("hXResVsTopo");
      hYResVsTopo[layer][type]->Write("hYResVsTopo");
      hZResVsTopo[layer][type]->Write("hZResVsTopo");
      hRmsXResVsTopo[layer][type]->Write("hRmsXResVsTopo");
      hRmsYResVsTopo[layer][type]->Write("hRmsYResVsTopo");
      hRmsZResVsTopo[layer][type]->Write("hRmsZResVsTopo");
      hMeanXResVsTopo[layer][type]->Write("hMeanXResVsTopo");
      hMeanYResVsTopo[layer][type]->Write("hMeanYResVsTopo");
      hMeanZResVsTopo[layer][type]->Write("hMeanZResVsTopo");

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

  // Write digit efficiency histograms
  for (int layer = 0; layer < 2; ++layer) {
    for (int type = 0; type < 2; ++type) {
      outFile->cd(Form("%sTrkLayer%d", trackName[type], layer));
      hNotRecoHits[layer][type]->Write();
    }
  }

  // Map all found cluster topologies to histograms
  // Create directories of all topologies
  for (const auto& topoName : topologyNames) {
    outFile->mkdir(Form("TopologyDictionary/All/%s", topoName.c_str()));
    outFile->mkdir(Form("TopologyDictionary/Real/%s", topoName.c_str()));
    outFile->mkdir(Form("TopologyDictionary/Fake/%s", topoName.c_str()));
  }
  for (int iMapEntry = 0; iMapEntry < sortedTopoMap.size(); ++iMapEntry) {
    const auto& [topoKey, topology] = sortedTopoMap[iMapEntry];
    std::string topoName = topologyNames[topology.mTopology];
    int spanRow = topology.mSizeX;
    int spanCol = topology.mSizeZ;
    uint16_t bitmask = topology.mPattern;

    TH2F* hTopoDisplayAll = new TH2F(Form("spanRow_%i_spanCol_%i_key_%i_all", spanRow, spanCol, topoKey), Form("Cluster Topology %s;Row;Column", topoName.c_str()), 
                                     spanRow + 2, -1.5, spanRow + 0.5, spanCol + 2, -1.5, spanCol + 0.5);
    TH2F* hTopoDisplayReal = new TH2F(Form("spanRow_%i_spanCol_%i_key_%i_real", spanRow, spanCol, topoKey), Form("Cluster Topology %s;Row;Column", topoName.c_str()), 
                                      spanRow + 2, -1.5, spanRow + 0.5, spanCol + 2, -1.5, spanCol + 0.5);
    TH2F* hTopoDisplayFake = new TH2F(Form("spanRow_%i_spanCol_%i_key_%i_fake", spanRow, spanCol, topoKey), Form("Cluster Topology %s;Row;Column", topoName.c_str()), 
                                      spanRow + 2, -1.5, spanRow + 0.5, spanCol + 2, -1.5, spanCol + 0.5);

    TH2F* hTopoCOGAll = new TH2F(Form("spanRow_%i_spanCol_%i_key_%i_all_COG", spanRow, spanCol, topoKey), Form("Cluster Topology %s;Row;Column", topoName.c_str()), 
                                      spanRow + 2, -1.5, spanRow + 0.5, spanCol + 2, -1.5, spanCol + 0.5);
    TH2F* hTopoCOGReal = new TH2F(Form("spanRow_%i_spanCol_%i_key_%i_real_COG", spanRow, spanCol, topoKey), Form("Cluster Topology %s;Row;Column", topoName.c_str()), 
                                       spanRow + 2, -1.5, spanRow + 0.5, spanCol + 2, -1.5, spanCol + 0.5);
    TH2F* hTopoCOGFake = new TH2F(Form("spanRow_%i_spanCol_%i_key_%i_fake_COG", spanRow, spanCol, topoKey), Form("Cluster Topology %s;Row;Column", topoName.c_str()), 
                                       spanRow + 2, -1.5, spanRow + 0.5, spanCol + 2, -1.5, spanCol + 0.5);

    int frequency = topology.mFrequency;
    hTopoCOGAll->Fill(topology.mOffsetXToCOG, topology.mOffsetZToCOG, frequency);
    int countFakeThisTopo = std::count_if(clustersProperties.begin(), clustersProperties.end(),
                                          [topoKey](const ClusterProperties& cls)
                                          { return cls.topoKey == topoKey && cls.isFake; });
    hTopoCOGAll->Fill(topology.mOffsetXToCOG, topology.mOffsetZToCOG, countFakeThisTopo);
    int countRealThisTopo = std::count_if(clustersProperties.begin(), clustersProperties.end(),
                                          [topoKey](const ClusterProperties& cls)
                                          { return cls.topoKey == topoKey && !cls.isFake; });
    hTopoCOGAll->Fill(topology.mOffsetXToCOG, topology.mOffsetZToCOG, countRealThisTopo);

      // Loop over the bits of bitmask and fill the histogram
    for (int row = 0; row < spanRow; ++row) {
      for (int col = 0; col < spanCol; ++col) {
        int bitIndex = row * spanCol + col;
        if (bitmask & (1 << bitIndex)) {
          hTopoDisplayAll->Fill(row, col, frequency);
          if (countRealThisTopo > 0) {
            hTopoDisplayReal->Fill(row, col, countFakeThisTopo);
          }
          if (countFakeThisTopo > 0) {
            hTopoDisplayFake->Fill(row, col, countRealThisTopo);
          }
        }
      }
    }
    outFile->cd(Form("TopologyDictionary/All/%s", topoName.c_str()));
    hTopoDisplayAll->Write();
    hTopoCOGAll->Write();
    if (countRealThisTopo > 0) {
      outFile->cd(Form("TopologyDictionary/Real/%s", topoName.c_str()));
      hTopoDisplayReal->Write();
      hTopoCOGReal->Write();
    }
    if (countFakeThisTopo > 0) {
      outFile->cd(Form("TopologyDictionary/Fake/%s", topoName.c_str()));
      hTopoDisplayFake->Write();
      hTopoCOGFake->Write();
    }
    delete hTopoDisplayAll;
    delete hTopoDisplayReal;
    delete hTopoDisplayFake;
    delete hTopoCOGAll;
    delete hTopoCOGReal;
    delete hTopoCOGFake;
  }

  outFile->Close();
  delete outFile;

  // Print all hits without any cluster associated to them
  Print(verbose, "----> Printing all hits without any cluster associated to them");
  for (const auto& trackProperties : tracksHitCls) {
    int eventID = static_cast<int>(trackProperties.first >> 32);
    int trackID = static_cast<int>(trackProperties.first & 0xFFFFFFFF);
    const auto& mcTrack = (*mcTracksPerEvent[eventID])[trackID];
    if (!mcTrack.isPrimary()) { continue; }

    if (!(trackProperties.second.hitIndicesL0.size() > 0 && trackProperties.second.clsIndicesL0.empty()) ||
        !(trackProperties.second.hitIndicesL1.size() > 0 && trackProperties.second.clsIndicesL1.empty())) {
      continue; // Skip tracks with reconstructed clusters
    }

    Print(verbose, "\nTrack has not reconstructed clusters for hits in both layers. Printing track and hit information:");
    PrintMcTrack(verbose, mcTrack);
    if (trackProperties.second.clsIndicesL0.empty()) {
      for (const auto& hitIdx : trackProperties.second.hitIndicesL0) {
        const auto& hit = (*hitsPerEvent[eventID])[hitIdx];
        PrintHit(verbose, hit, iotofGeom);
      }
    }
    if (trackProperties.second.clsIndicesL1.empty()) {
      for (const auto& hitIdx : trackProperties.second.hitIndicesL1) {
        const auto& hit = (*hitsPerEvent[eventID])[hitIdx];
        PrintHit(verbose, hit, iotofGeom);
      }
    }
  }
}
