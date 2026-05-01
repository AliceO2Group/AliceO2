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
///
/// \file TimeFrame.cxx
/// \brief TRK TimeFrame implementation
///

#include "ALICE3GlobalReconstruction/TimeFrame.h"
#include "TRKReconstruction/Clusterer.h"
#include "TRKSimulation/Hit.h"
#include "TRKBase/GeometryTGeo.h"
#include "TRKBase/SegmentationChip.h"
#include "Framework/Logger.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "Steer/MCKinematicsReader.h"
#include <TTree.h>
#include <TRandom3.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory_resource>
#include <ranges>
#include <vector>
#include <array>

namespace o2::trk
{

template <int nLayers>
void TimeFrame<nLayers>::initTimingTables(const std::array<o2::its::LayerTiming, nLayers>& timings)
{
  if (mTimingTablesInitialised) {
    return;
  }
  typename o2::its::TimeFrame<nLayers>::ROFOverlapTableN rofOverlapTable;
  typename o2::its::TimeFrame<nLayers>::ROFVertexLookupTableN rofVertexLookupTable;
  typename o2::its::TimeFrame<nLayers>::ROFMaskTableN rofMaskTable;
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    rofOverlapTable.defineLayer(iLayer, timings[iLayer]);
    rofVertexLookupTable.defineLayer(iLayer, timings[iLayer]);
    rofMaskTable.defineLayer(iLayer, timings[iLayer]);
  }
  rofOverlapTable.init();
  rofVertexLookupTable.init();
  rofMaskTable.init();
  // ALICE3 TRK currently runs without per-ROF selection — no FastMultEst,
  // no PhysTrigger, no UPC iteration on this detector yet. Enable every ROF
  // so useMultiplictyMask() acts as an effective no-op for the tracker until
  // a real selector is ported (see ITS TrackingInterface.cxx for the
  // reference implementation).
  rofMaskTable.resetMask(1u);
  this->setROFOverlapTable(std::move(rofOverlapTable));
  this->setROFVertexLookupTable(std::move(rofVertexLookupTable));
  this->setMultiplicityCutMask(std::move(rofMaskTable));
  this->useMultiplictyMask();
  mTimingTablesInitialised = true;

  // One-shot log of resolved per-layer timing and mask state.
  const auto maskView = this->getROFMaskView();
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    LOGP(info, "TRK timing initialised: layer {}: {}", iLayer, timings[iLayer].asString());
    LOGP(info, "TRK ROF mask: {}", maskView.asString(iLayer));
  }
}

template <int nLayers>
void TimeFrame<nLayers>::deriveAndInitTiming(const std::array<gsl::span<const o2::trk::ROFRecord>, nLayers>& layerROFs)
{
  if (mTimingTablesInitialised) {
    return;
  }

  // Anchor the TF at the earliest first-ROF BCData seen across layers, so
  // every layer's mROFBias is non-negative and intra-anchor BC values stay
  // bounded (LayerTiming uses uint32 BC counters). For non-staggered TRK
  // every layer is anchored at BC=0 (mROFBias=0); per-layer biases only
  // diverge once the digitiser actually staggers the readout.
  o2::InteractionRecord anchor{0, 0};
  bool haveAnchor = false;
  for (const auto& span : layerROFs) {
    if (span.empty()) {
      continue;
    }
    const auto& first = span.front().getBCData();
    if (!haveAnchor || first.toLong() < anchor.toLong()) {
      anchor = first;
      haveAnchor = true;
    }
  }
  mTFAnchorIR = anchor;
  const int64_t anchorBC = anchor.toLong();

  std::array<o2::its::LayerTiming, nLayers> timings{};
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    const auto& span = layerROFs[iLayer];
    auto& t = timings[iLayer];
    t.mNROFsTF = static_cast<o2::its::LayerTiming::BCType>(span.size());

    if (span.size() >= 2) {
      const int64_t delta = span[1].getBCData().toLong() - span[0].getBCData().toLong();
      if (delta > 0) {
        t.mROFLength = static_cast<o2::its::LayerTiming::BCType>(delta);
      } else {
        LOGP(warning, "TRK layer {}: non-positive BC delta between rofs[0] and rofs[1] ({}); falling back to mROFLength=1", iLayer, delta);
        t.mROFLength = 1;
      }
    } else {
      if (span.size() == 1) {
        LOGP(warning, "TRK layer {}: only one input ROF — cannot derive mROFLength; falling back to mROFLength=1", iLayer);
      }
      t.mROFLength = 1;
    }

    if (!span.empty()) {
      const int64_t bias = span.front().getBCData().toLong() - anchorBC;
      // bias is guaranteed >= 0 by the anchor choice above
      t.mROFBias = static_cast<o2::its::LayerTiming::BCType>(bias);
    }
    t.mROFDelay = 0;
    t.mROFAddTimeErr = 0;
  }

  initTimingTables(timings);
}

template <int nLayers>
int TimeFrame<nLayers>::loadROFsFromHitTree(TTree* hitsTree, GeometryTGeo* gman, const nlohmann::json& config)
{
  constexpr std::array<int, 2> startLayer{0, 3};
  const Long64_t nEvents = hitsTree->GetEntries();

  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L) | o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  std::vector<o2::trk::Hit>* trkHit = nullptr;
  hitsTree->SetBranchAddress("TRKHit", &trkHit);

  const int inROFpileup{config.contains("inROFpileup") ? config["inROFpileup"].get<int>() : 1};

  // Calculate number of ROFs and initialize data structures
  const int nRofs = (nEvents + inROFpileup - 1) / inROFpileup;
  // Hit-tree path has no real BCData, so the timing stays at the placeholder
  // (mROFLength = 1 BC, no delay/bias). For timing-sensitive work use the
  // cluster path, which derives mROFLength from the input ROF BC stamps.
  std::array<o2::its::LayerTiming, nLayers> timings{};
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    timings[iLayer].mNROFsTF = static_cast<o2::its::LayerTiming::BCType>(nRofs);
    timings[iLayer].mROFLength = 1;
  }
  this->initTimingTables(timings);
  const auto& timing = this->getROFOverlapTableView().getLayer(0);
  if (timing.mNROFsTF != static_cast<o2::its::LayerTiming::BCType>(nRofs)) {
    LOGP(fatal, "TRK: inconsistent number of ROFs across TFs: timing has {}, hit-tree path produced {}", timing.mNROFsTF, nRofs);
  }

  // Reset and prepare ROF data structures
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    this->mMinR[iLayer] = std::numeric_limits<float>::max();
    this->mMaxR[iLayer] = std::numeric_limits<float>::lowest();
    this->mROFramesClusters[iLayer].clear();
    this->mROFramesClusters[iLayer].resize(nRofs + 1, 0);
    this->mUnsortedClusters[iLayer].clear();
    this->mTrackingFrameInfo[iLayer].clear();
    this->mClusterExternalIndices[iLayer].clear();
    this->mClusterSize[iLayer].clear();
  }

  // Pre-count hits to reserve memory efficiently
  std::array<int, nLayers> clusterCountPerLayer{};
  for (Long64_t iEvent = 0; iEvent < nEvents; ++iEvent) {
    hitsTree->GetEntry(iEvent);
    for (const auto& hit : *trkHit) {
      if (gman->getDisk(hit.GetDetectorID()) != -1) {
        continue; // skip non-barrel hits
      }
      int subDetID = gman->getSubDetID(hit.GetDetectorID());
      const int layer = startLayer[subDetID] + gman->getLayer(hit.GetDetectorID());
      if (layer >= nLayers) {
        continue;
      }
      ++clusterCountPerLayer[layer];
    }
  }

  // Reserve memory for all layers
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    this->mUnsortedClusters[iLayer].reserve(clusterCountPerLayer[iLayer]);
    this->mTrackingFrameInfo[iLayer].reserve(clusterCountPerLayer[iLayer]);
    this->mClusterExternalIndices[iLayer].reserve(clusterCountPerLayer[iLayer]);
    this->mClusterSize[iLayer].reserve(clusterCountPerLayer[iLayer]);
  }

  std::array<float, 11> resolution{0.001, 0.001, 0.001, 0.001, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004};
  if (config["geometry"]["pitch"].size() == nLayers) {
    for (int iLayer{0}; iLayer < config["geometry"]["pitch"].size(); ++iLayer) {
      LOGP(info, "Setting resolution for layer {} from config", iLayer);
      LOGP(info, "Layer {} pitch {} cm", iLayer, config["geometry"]["pitch"][iLayer].get<float>());
      resolution[iLayer] = config["geometry"]["pitch"][iLayer].get<float>() / std::sqrt(12.f);
    }
  }
  LOGP(info, "Number of active parts in VD: {}", gman->getNumberOfActivePartsVD());

  int hitCounter{0};
  auto labels = new dataformats::MCTruthContainer<MCCompLabel>();

  int iRof{0}; // Current ROF index
  for (Long64_t iEvent = 0; iEvent < nEvents; ++iEvent) {
    hitsTree->GetEntry(iEvent);

    for (auto& hit : *trkHit) {
      if (gman->getDisk(hit.GetDetectorID()) != -1) {
        continue; // skip non-barrel hits for this test
      }
      int subDetID = gman->getSubDetID(hit.GetDetectorID());
      const int layer = startLayer[subDetID] + gman->getLayer(hit.GetDetectorID());

      float alpha{0.f};
      o2::math_utils::Point3D<float> gloXYZ;
      o2::math_utils::Point3D<float> trkXYZ;
      float r{0.f};
      if (layer >= nLayers) {
        continue;
      }
      if (layer >= 3) {
        int chipID = hit.GetDetectorID();
        alpha = gman->getSensorRefAlphaMLOT(chipID);
        const o2::math_utils::Transform3D& l2g = gman->getMatrixL2G(chipID);
        auto locXYZ = l2g ^ (hit.GetPos());
        locXYZ.SetX(locXYZ.X() + gRandom->Gaus(0.0, resolution[layer]));
        locXYZ.SetZ(locXYZ.Z() + gRandom->Gaus(0.0, resolution[layer]));
        gloXYZ = gman->getMatrixL2G(chipID) * locXYZ;
        trkXYZ = gman->getMatrixT2L(chipID - gman->getNumberOfActivePartsVD()) ^ locXYZ;
        r = std::hypot(gloXYZ.X(), gloXYZ.Y());
      } else {
        const auto& hitPos = hit.GetPos();
        r = std::hypot(hitPos.X(), hitPos.Y());
        alpha = std::atan2(hitPos.Y(), hitPos.X()) + gRandom->Gaus(0.0, resolution[layer] / r);
        o2::math_utils::bringTo02Pi(alpha);
        gloXYZ.SetX(r * std::cos(alpha));
        gloXYZ.SetY(r * std::sin(alpha));
        gloXYZ.SetZ(hitPos.Z() + gRandom->Gaus(0.0, resolution[layer]));
        trkXYZ.SetX(r);
        trkXYZ.SetY(0.f);
        trkXYZ.SetZ(gloXYZ.Z());
      }
      this->mMinR[layer] = std::min(this->mMinR[layer], r);
      this->mMaxR[layer] = std::max(this->mMaxR[layer], r);
      this->addTrackingFrameInfoToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), trkXYZ.x(), alpha,
                                        std::array<float, 2>{trkXYZ.y(), trkXYZ.z()},
                                        std::array<float, 3>{resolution[layer] * resolution[layer], 0., resolution[layer] * resolution[layer]});
      /// Rotate to the global frame
      this->addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), this->mUnsortedClusters[layer].size());
      this->addClusterExternalIndexToLayer(layer, hitCounter);
      this->mClusterSize[layer].push_back(1); // For compatibility with cluster-based tracking, set cluster size to 1 for hits
      MCCompLabel label{hit.GetTrackID(), static_cast<int>(iEvent), 0};
      labels->addElement(hitCounter, label);
      hitCounter++;
    }
    trkHit->clear();

    // Update ROF structure when we complete an ROF or reach the last event
    if ((iEvent + 1) % inROFpileup == 0 || iEvent == nEvents - 1) {
      iRof++;
      for (unsigned int iLayer{0}; iLayer < this->mUnsortedClusters.size(); ++iLayer) {
        this->mROFramesClusters[iLayer][iRof] = this->mUnsortedClusters[iLayer].size(); // effectively calculating an exclusive sum
      }
      // Update primary vertices ROF structure
    }
    this->mClusterLabels[0] = labels;
  }
  return nRofs;
}

template <int nLayers>
int TimeFrame<nLayers>::loadROFrameData(gsl::span<const o2::trk::ROFRecord> rofs,
                                        gsl::span<const o2::trk::Cluster> clusters,
                                        gsl::span<const unsigned char> patterns,
                                        const dataformats::MCTruthContainer<MCCompLabel>* mcLabels,
                                        float yPlaneMLOT)
{
  constexpr std::array<int, 2> startLayer{0, 3};
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L) | o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  const int nRofs = rofs.size();
  // Per-layer LayerTiming (and mROFLength / mROFBias) must already be in place;
  // the cluster path requires the caller (TrackerSpec) to invoke
  // deriveAndInitTiming() with the per-layer ROF spans first. We deliberately
  // do not derive timing from the merged `rofs` span here — that would lose
  // per-layer information needed for staggered readouts.
  if (!mTimingTablesInitialised) {
    LOGP(fatal, "TRK::loadROFrameData: timing tables not initialised — call deriveAndInitTiming() first");
  }
  const auto& timing = this->getROFOverlapTableView().getLayer(0);
  if (timing.mNROFsTF != static_cast<o2::its::LayerTiming::BCType>(nRofs)) {
    LOGP(fatal, "TRK: inconsistent number of ROFs across TFs: timing has {}, cluster path received {}", timing.mNROFsTF, nRofs);
  }

  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    this->mMinR[iLayer] = std::numeric_limits<float>::max();
    this->mMaxR[iLayer] = std::numeric_limits<float>::lowest();
    this->mROFramesClusters[iLayer].clear();
    this->mROFramesClusters[iLayer].resize(nRofs + 1, 0);
    this->mUnsortedClusters[iLayer].clear();
    this->mTrackingFrameInfo[iLayer].clear();
    this->mClusterExternalIndices[iLayer].clear();
    this->mClusterSize[iLayer].clear();
  }

  std::array<int, nLayers> clusterCountPerLayer{};
  for (const auto& c : clusters) {
    if (c.subDetID < 0 || c.subDetID > 1) {
      continue;
    }
    if (c.disk != -1) {
      continue; // skip non-barrel clusters for now
    }
    const int layer = startLayer[c.subDetID] + c.layer;
    if (layer < 0 || layer >= nLayers) {
      continue;
    }
    ++clusterCountPerLayer[layer];
  }

  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    this->mUnsortedClusters[iLayer].reserve(clusterCountPerLayer[iLayer]);
    this->mTrackingFrameInfo[iLayer].reserve(clusterCountPerLayer[iLayer]);
    this->mClusterExternalIndices[iLayer].reserve(clusterCountPerLayer[iLayer]);
    this->mClusterSize[iLayer].reserve(clusterCountPerLayer[iLayer]);
  }

  const uint8_t* pattPtr = patterns.data();
  const uint8_t* pattEnd = pattPtr + patterns.size();

  for (size_t iRof{0}; iRof < rofs.size(); ++iRof) {
    const auto& rof = rofs[iRof];
    const int first = rof.getFirstEntry();
    const int last = first + rof.getNEntries();

    for (int clusterId{first}; clusterId < last; ++clusterId) {
      // Parse the pattern header up-front so we always know how many bytes
      // this cluster occupies in the pattern stream. The stream is keyed
      // per-cluster: every `continue` below MUST advance pattPtr by
      // pattAdvance, otherwise the next cluster decodes from a stale
      // offset and the whole layer's geometry is corrupted.
      if (pattPtr + 2 > pattEnd) {
        LOGP(error, "Pattern stream exhausted while decoding cluster {}", clusterId);
        break;
      }
      const uint8_t* pattForCluster = pattPtr;
      const int nBytes = (pattForCluster[0] * pattForCluster[1] + 7) / 8;
      if (pattPtr + 2 + nBytes > pattEnd) {
        LOGP(error, "Pattern stream truncated for cluster {}", clusterId);
        break;
      }
      const int pattAdvance = 2 + nBytes;

      if (clusterId < 0 || clusterId >= static_cast<int>(clusters.size())) {
        LOGP(warning, "Skipping out-of-range cluster id {} for ROF {}", clusterId, iRof);
        pattPtr += pattAdvance;
        continue;
      }

      const auto& c = clusters[clusterId];
      if (c.subDetID < 0 || c.subDetID > 1 || c.disk != -1) {
        pattPtr += pattAdvance;
        continue;
      }

      const int layer = startLayer[c.subDetID] + c.layer;
      if (layer < 0 || layer >= nLayers) {
        LOGP(error, "Skipping cluster with invalid layer {} (subDetID {}, layer {})", layer, c.subDetID, c.layer);
        pattPtr += pattAdvance;
        continue;
      }

      auto locXYZ = Clusterer::getClusterLocalCoordinates(c, pattForCluster, yPlaneMLOT);
      pattPtr += pattAdvance;

      const auto gloXYZ = geom->getMatrixL2G(c.chipID) * locXYZ;

      float alpha{0.f};
      o2::math_utils::Point3D<float> trkXYZ;
      if (c.subDetID == 1) {
        alpha = geom->getSensorRefAlphaMLOT(c.chipID);
        trkXYZ = geom->getMatrixT2L(c.chipID - geom->getNumberOfActivePartsVD()) ^ locXYZ;
      } else {
        const float r = std::hypot(gloXYZ.X(), gloXYZ.Y());
        alpha = std::atan2(gloXYZ.Y(), gloXYZ.X());
        o2::math_utils::bringTo02Pi(alpha);
        trkXYZ.SetX(r);
        trkXYZ.SetY(0.f);
        trkXYZ.SetZ(gloXYZ.Z());
      }

      const float r = std::hypot(gloXYZ.X(), gloXYZ.Y());
      this->mMinR[layer] = std::min(this->mMinR[layer], r);
      this->mMaxR[layer] = std::max(this->mMaxR[layer], r);

      const float sigmaY2 = (c.subDetID == 0)
                              ? 0.25f * SegmentationChip::PitchRowVD * SegmentationChip::PitchRowVD
                              : 0.25f * SegmentationChip::PitchRowMLOT * SegmentationChip::PitchRowMLOT;
      const float sigmaZ2 = (c.subDetID == 0)
                              ? 0.25f * SegmentationChip::PitchColVD * SegmentationChip::PitchColVD
                              : 0.25f * SegmentationChip::PitchColMLOT * SegmentationChip::PitchColMLOT;

      this->addTrackingFrameInfoToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), trkXYZ.x(), alpha,
                                        std::array<float, 2>{trkXYZ.y(), trkXYZ.z()},
                                        std::array<float, 3>{sigmaY2, 0.f, sigmaZ2});
      this->addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), this->mUnsortedClusters[layer].size());
      this->addClusterExternalIndexToLayer(layer, clusterId);
      this->mClusterSize[layer].push_back(std::clamp(static_cast<unsigned int>(c.size), 0u, 255u));
    }

    for (unsigned int iL{0}; iL < this->mUnsortedClusters.size(); ++iL) {
      this->mROFramesClusters[iL][iRof + 1] = this->mUnsortedClusters[iL].size();
    }
  }

  for (auto i = 0; i < this->mNTrackletsPerCluster.size(); ++i) {
    this->mNTrackletsPerCluster[i].resize(this->mUnsortedClusters[1].size());
    this->mNTrackletsPerClusterSum[i].resize(this->mUnsortedClusters[1].size() + 1);
  }

  if (mcLabels != nullptr) {
    this->mClusterLabels[0] = mcLabels;
  }

  return nRofs;
}

template <int nLayers>
void TimeFrame<nLayers>::getPrimaryVerticesFromMC(TTree* mcHeaderTree, int nRofs, Long64_t nEvents, int inROFpileup)
{
  auto mcheader = new o2::dataformats::MCEventHeader;
  mcHeaderTree->SetBranchAddress("MCEventHeader.", &mcheader);

  this->mPrimaryVertices.clear();
  this->mPrimaryVerticesLabels.clear();

  // Vertex timestamps live in the clock layer's intra-anchor BC frame
  // (anchor = mTFAnchorIR set by deriveAndInitTiming, or {0,0} for the
  // hit-tree placeholder). TimeEstBC interval [t0, t0+tE] covers one full
  // clock-layer ROF so that clockLayer.getROF(vertex.getTimeStamp().lower())
  // round-trips back to iRof.
  const auto& clockLayer = this->getROFOverlapTableView().getClockLayer();
  const auto rofLength = clockLayer.mROFLength;

  int iRof{0};
  for (Long64_t iEvent = 0; iEvent < nEvents; ++iEvent) {
    mcHeaderTree->GetEntry(iEvent);
    o2::its::Vertex vertex;
    vertex.setTimeStamp(o2::its::TimeEstBC{
      clockLayer.getROFStartInBC(iRof),
      static_cast<o2::its::TimeStampErrorType>(rofLength)});
    vertex.setXYZ(mcheader->GetX(), mcheader->GetY(), mcheader->GetZ());
    vertex.setNContributors(30);
    vertex.setChi2(0.f);
    LOGP(debug, "ROF {}: Added primary vertex at ({}, {}, {})", iRof, mcheader->GetX(), mcheader->GetY(), mcheader->GetZ());
    this->addPrimaryVertex(vertex);
    this->addPrimaryVertexLabel({o2::MCCompLabel{o2::MCCompLabel::maxTrackID(), static_cast<int>(iEvent), 0, false}, 1.f});
    if ((iEvent + 1) % inROFpileup == 0 || iEvent == nEvents - 1) {
      iRof++;
    }
  }
  this->updateROFVertexLookupTable();
}



// Explicit template instantiation for TRK with 11 layers

template <int nLayers>
void TimeFrame<nLayers>::addTruthSeedingVertices(gsl::span<const o2::trk::ROFRecord> rofs)
{
  LOGP(info, "TRK: using truth seeds as vertices from DigitizationContext");
  this->mPrimaryVertices.clear();
  this->mPrimaryVerticesLabels.clear();

  const auto dc = o2::steer::DigitizationContext::loadFromFile("collisioncontext.root");
  const auto irs = dc->getEventRecords();
  o2::steer::MCKinematicsReader mcReader(dc);

  // Pre-compute ROF start BC (as absolute long) for binary search
  std::vector<int64_t> rofStartBC(rofs.size());
  for (size_t i = 0; i < rofs.size(); ++i) {
    rofStartBC[i] = rofs[i].getBCData().toLong();
  }

  // Vertex timestamps live in the clock layer's intra-anchor BC frame
  // (anchor = mTFAnchorIR set by deriveAndInitTiming). TimeEstBC interval
  // covers one full clock-layer ROF, so
  // clockLayer.getROF(vertex.getTimeStamp().lower()) maps back to rofId.
  const auto& clockLayer = this->getROFOverlapTableView().getClockLayer();
  const auto rofLength = clockLayer.mROFLength;

  using Vertex = o2::its::Vertex;
  struct VertInfo {
    std::pmr::vector<Vertex> vertices;
    std::pmr::vector<int> srcs;
    std::pmr::vector<int> events;
  };
  std::map<int, VertInfo> vertMap;

  const int iSrc = 0; // primary collision generator source
  auto eveId2colId = dc->getCollisionIndicesForSource(iSrc);
  for (int iEve{0}; iEve < mcReader.getNEvents(iSrc); ++iEve) {
    const auto& ir = irs[eveId2colId[iEve]];
    if (!ir.isDummy()) {
      const auto& eve = mcReader.getMCEventHeader(iSrc, iEve);
      const int64_t evBC = ir.toLong();
      // Find ROF: last ROF whose start BC <= evBC
      auto it = std::upper_bound(rofStartBC.begin(), rofStartBC.end(), evBC);
      if (it != rofStartBC.begin()) {
        --it;
        int rofId = static_cast<int>(std::distance(rofStartBC.begin(), it));
        auto* mr = this->mMemoryPool.get();
        if (!vertMap.contains(rofId)) {
          vertMap[rofId] = {
            .vertices = std::pmr::vector<Vertex>(mr),
            .srcs = std::pmr::vector<int>(mr),
            .events = std::pmr::vector<int>(mr),
          };
        }
        Vertex vert;
        vert.setTimeStamp(o2::its::TimeEstBC{
          clockLayer.getROFStartInBC(rofId),
          static_cast<o2::its::TimeStampErrorType>(rofLength)});
        vert.setNContributors(std::max(1L, std::ranges::count_if(
                                            mcReader.getTracks(iSrc, iEve),
                                            [](const auto& trk) {
                                              return trk.isPrimary() && trk.GetPt() > 0.05 && std::abs(trk.GetEta()) < 1.1;
                                            })));
        vert.setXYZ((float)eve.GetX(), (float)eve.GetY(), (float)eve.GetZ());
        vert.setChi2(1);
        constexpr float cov = 50e-9f;
        vert.setCov(cov, cov, cov, cov, cov, cov);
        vertMap[rofId].vertices.push_back(vert);
        vertMap[rofId].srcs.push_back(iSrc);
        vertMap[rofId].events.push_back(iEve);
      }
    }
    mcReader.releaseTracksForSourceAndEvent(iSrc, iEve);
  }

  size_t nVerts{0};
  auto* mr = this->mMemoryPool.get();
  for (int iROF{0}; iROF < static_cast<int>(rofs.size()); ++iROF) {
    std::pmr::vector<Vertex> verts(mr);
    std::pmr::vector<std::pair<o2::MCCompLabel, float>> polls(mr);
    if (vertMap.contains(iROF)) {
      const auto& info = vertMap[iROF];
      verts = info.vertices;
      nVerts += verts.size();
      for (size_t i{0}; i < verts.size(); ++i) {
        o2::MCCompLabel lbl(o2::MCCompLabel::maxTrackID(), info.events[i], info.srcs[i], false);
        polls.emplace_back(lbl, 1.f);
      }
    }
    for (const auto& vert : verts) {
      this->addPrimaryVertex(vert);
    }
    for (const auto& label : polls) {
      this->addPrimaryVertexLabel(label);
    }
  }
  this->updateROFVertexLookupTable();
  LOGP(info, "TRK truth seeding: {}/{} ROFs with {} vertices -> <NV>={:.2f}",
       vertMap.size(), rofs.size(), nVerts,
       vertMap.size() > 0 ? (float)nVerts / (float)vertMap.size() : 0.f);
}

template class TimeFrame<11>;

} // namespace o2::trk
