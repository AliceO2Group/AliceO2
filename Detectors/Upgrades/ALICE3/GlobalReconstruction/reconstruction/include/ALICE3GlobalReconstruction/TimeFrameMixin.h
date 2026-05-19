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
/// \file TimeFrameMixin.h
/// \brief Shared TRK TimeFrame helpers for CPU and GPU backends.
///

#ifndef ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAMEMIXIN_H
#define ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAMEMIXIN_H

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsTRK/Cluster.h"
#include "DataFormatsTRK/ROFRecord.h"
#include "ITStracking/ROFLookupTables.h"
#include "ITStracking/TimeFrame.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "Steer/MCKinematicsReader.h"
#include "TRKReconstruction/Clusterer.h"
#include "TRKSimulation/Hit.h"
#include "TRKBase/GeometryTGeo.h"
#include "TRKBase/SegmentationChip.h"
#include "Framework/Logger.h"

#include <Rtypes.h>
#include <TTree.h>
#include <TRandom3.h>
#include <gsl/span>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <ranges>
#include <vector>

#include <nlohmann/json.hpp>

namespace o2::trk
{

template <int nLayers, class Base>
class TimeFrameMixin : public Base
{
 public:
  TimeFrameMixin() = default;
  ~TimeFrameMixin() override = default;

  int loadROFsFromHitTree(TTree* hitsTree, GeometryTGeo* gman, const nlohmann::json& config);

  int loadROFrameData(const std::array<gsl::span<const o2::trk::ROFRecord>, nLayers>& layerROFs,
                      const std::array<gsl::span<const o2::trk::Cluster>, nLayers>& layerClusters,
                      const std::array<gsl::span<const unsigned char>, nLayers>& layerPatterns,
                      const std::array<const dataformats::MCTruthContainer<MCCompLabel>*, nLayers>* mcLabels = nullptr,
                      float yPlaneMLOT = 0.f);

  void getPrimaryVerticesFromMC(TTree* mcHeaderTree, int nRofs, Long64_t nEvents, int inROFpileup);

  void addTruthSeedingVertices();

  void deriveAndInitTiming(const std::array<gsl::span<const o2::trk::ROFRecord>, nLayers>& layerROFs);

  const o2::InteractionRecord& getTFAnchorIR() const noexcept { return mTFAnchorIR; }

 protected:
  void initTimingTables(const std::array<o2::its::LayerTiming, nLayers>& timings);
  void updateHostROFVertexLookupTable();

  bool mTimingTablesInitialised{false};
  o2::InteractionRecord mTFAnchorIR{0, 0};
};

template <int nLayers, class Base>
void TimeFrameMixin<nLayers, Base>::updateHostROFVertexLookupTable()
{
  static_cast<o2::its::TimeFrame<nLayers>*>(this)->updateROFVertexLookupTable();
}

template <int nLayers, class Base>
void TimeFrameMixin<nLayers, Base>::initTimingTables(const std::array<o2::its::LayerTiming, nLayers>& timings)
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
  rofMaskTable.resetMask(1u);
  this->setROFOverlapTable(std::move(rofOverlapTable));
  this->setROFVertexLookupTable(std::move(rofVertexLookupTable));
  this->setMultiplicityCutMask(std::move(rofMaskTable));
  this->useMultiplictyMask();
  mTimingTablesInitialised = true;

  const auto maskView = this->getROFMaskView();
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    LOGP(info, "TRK timing initialised: layer {}: {}", iLayer, timings[iLayer].asString());
    LOGP(info, "TRK ROF mask: {}", maskView.asString(iLayer));
  }
}

template <int nLayers, class Base>
void TimeFrameMixin<nLayers, Base>::deriveAndInitTiming(const std::array<gsl::span<const o2::trk::ROFRecord>, nLayers>& layerROFs)
{
  if (mTimingTablesInitialised) {
    return;
  }

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
      t.mROFBias = static_cast<o2::its::LayerTiming::BCType>(bias);
    }
    t.mROFDelay = 0;
    t.mROFAddTimeErr = 0;
  }

  initTimingTables(timings);
}

template <int nLayers, class Base>
int TimeFrameMixin<nLayers, Base>::loadROFsFromHitTree(TTree* hitsTree, GeometryTGeo* gman, const nlohmann::json& config)
{
  constexpr std::array<int, 2> startLayer{0, 3};
  const Long64_t nEvents = hitsTree->GetEntries();
  this->setIsStaggered(true);

  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L) | o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  std::vector<o2::trk::Hit>* trkHit = nullptr;
  hitsTree->SetBranchAddress("TRKHit", &trkHit);

  const int inROFpileup{config.contains("inROFpileup") ? config["inROFpileup"].get<int>() : 1};

  const int nRofs = (nEvents + inROFpileup - 1) / inROFpileup;
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
  for (Long64_t iEvent = 0; iEvent < nEvents; ++iEvent) {
    hitsTree->GetEntry(iEvent);
    for (const auto& hit : *trkHit) {
      if (gman->getDisk(hit.GetDetectorID()) != -1) {
        continue;
      }
      int subDetID = gman->getSubDetID(hit.GetDetectorID());
      const int layer = startLayer[subDetID] + gman->getLayer(hit.GetDetectorID());
      if (layer >= nLayers) {
        continue;
      }
      ++clusterCountPerLayer[layer];
    }
  }

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

  std::array<int, nLayers> hitCounterPerLayer{};
  std::array<dataformats::MCTruthContainer<MCCompLabel>*, nLayers> labels{};
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    labels[iLayer] = new dataformats::MCTruthContainer<MCCompLabel>();
    this->mClusterLabels[iLayer] = labels[iLayer];
  }

  int iRof{0};
  for (Long64_t iEvent = 0; iEvent < nEvents; ++iEvent) {
    hitsTree->GetEntry(iEvent);

    for (auto& hit : *trkHit) {
      if (gman->getDisk(hit.GetDetectorID()) != -1) {
        continue;
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
      this->addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), this->mUnsortedClusters[layer].size());
      const int layerHitCounter = hitCounterPerLayer[layer]++;
      this->addClusterExternalIndexToLayer(layer, layerHitCounter);
      this->mClusterSize[layer].push_back(1);
      MCCompLabel label{hit.GetTrackID(), static_cast<int>(iEvent), 0};
      labels[layer]->addElement(layerHitCounter, label);
    }
    trkHit->clear();

    if ((iEvent + 1) % inROFpileup == 0 || iEvent == nEvents - 1) {
      iRof++;
      for (unsigned int iLayer{0}; iLayer < this->mUnsortedClusters.size(); ++iLayer) {
        this->mROFramesClusters[iLayer][iRof] = this->mUnsortedClusters[iLayer].size();
      }
    }
  }
  return nRofs;
}

template <int nLayers, class Base>
int TimeFrameMixin<nLayers, Base>::loadROFrameData(const std::array<gsl::span<const o2::trk::ROFRecord>, nLayers>& layerROFs,
                                                   const std::array<gsl::span<const o2::trk::Cluster>, nLayers>& layerClusters,
                                                   const std::array<gsl::span<const unsigned char>, nLayers>& layerPatterns,
                                                   const std::array<const dataformats::MCTruthContainer<MCCompLabel>*, nLayers>* mcLabels,
                                                   float yPlaneMLOT)
{
  constexpr std::array<int, 2> startLayer{0, 3};
  this->setIsStaggered(true);
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L) | o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  if (!mTimingTablesInitialised) {
    LOGP(fatal, "TRK::loadROFrameData: timing tables not initialised — call deriveAndInitTiming() first");
  }
  int nRofs{0};
  for (const auto& rofs : layerROFs) {
    nRofs = std::max(nRofs, static_cast<int>(rofs.size()));
  }

  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    const auto& timing = this->getROFOverlapTableView().getLayer(iLayer);
    if (timing.mNROFsTF != static_cast<o2::its::LayerTiming::BCType>(layerROFs[iLayer].size())) {
      LOGP(fatal, "TRK: inconsistent number of ROFs on layer {}: timing has {}, cluster path received {}", iLayer, timing.mNROFsTF, layerROFs[iLayer].size());
    }
    this->mMinR[iLayer] = std::numeric_limits<float>::max();
    this->mMaxR[iLayer] = std::numeric_limits<float>::lowest();
    this->mROFramesClusters[iLayer].clear();
    this->mROFramesClusters[iLayer].resize(layerROFs[iLayer].size() + 1, 0);
    this->mUnsortedClusters[iLayer].clear();
    this->mTrackingFrameInfo[iLayer].clear();
    this->mClusterExternalIndices[iLayer].clear();
    this->mClusterSize[iLayer].clear();
    this->mUnsortedClusters[iLayer].reserve(layerClusters[iLayer].size());
    this->mTrackingFrameInfo[iLayer].reserve(layerClusters[iLayer].size());
    this->mClusterExternalIndices[iLayer].reserve(layerClusters[iLayer].size());
    this->mClusterSize[iLayer].reserve(layerClusters[iLayer].size());
  }

  std::array<std::vector<size_t>, nLayers> patternOffsetsPerLayer;
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    auto& offsets = patternOffsetsPerLayer[iLayer];
    offsets.resize(layerClusters[iLayer].size(), std::numeric_limits<size_t>::max());
    size_t pattPos = 0;
    bool validPatterns = true;
    for (size_t clusterId{0}; clusterId < layerClusters[iLayer].size(); ++clusterId) {
      if (pattPos + 2 > layerPatterns[iLayer].size()) {
        validPatterns = false;
        break;
      }
      offsets[clusterId] = pattPos;
      const uint8_t rowSpan = layerPatterns[iLayer][pattPos];
      const uint8_t colSpan = layerPatterns[iLayer][pattPos + 1];
      const size_t nBytes = (size_t(rowSpan) * colSpan + 7) / 8;
      if (pattPos + 2 + nBytes > layerPatterns[iLayer].size()) {
        validPatterns = false;
        break;
      }
      pattPos += 2 + nBytes;
    }
    if (!validPatterns || pattPos != layerPatterns[iLayer].size()) {
      LOGP(fatal, "Malformed TRK pattern stream for layer {}: {} bytes for {} clusters",
           iLayer, layerPatterns[iLayer].size(), layerClusters[iLayer].size());
    }
  }

  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    for (size_t iRof{0}; iRof < layerROFs[iLayer].size(); ++iRof) {
      const auto& rof = layerROFs[iLayer][iRof];
      const int first = rof.getFirstEntry();
      const int last = first + rof.getNEntries();

      for (int clusterId{first}; clusterId < last; ++clusterId) {
        if (clusterId < 0 || clusterId >= static_cast<int>(layerClusters[iLayer].size())) {
          LOGP(warning, "Skipping out-of-range TRK cluster {} on layer {}", clusterId, iLayer);
          continue;
        }

        const auto& c = layerClusters[iLayer][clusterId];
        if (c.subDetID < 0 || c.subDetID > 1 || c.disk != -1) {
          continue;
        }

        const int clusterLayer = startLayer[c.subDetID] + c.layer;
        if (clusterLayer != iLayer) {
          LOGP(error, "Skipping cluster from layer {} found in TRK layer stream {}", clusterLayer, iLayer);
          continue;
        }

        const auto pattOffset = patternOffsetsPerLayer[iLayer][clusterId];
        const uint8_t* pattForCluster = layerPatterns[iLayer].data() + pattOffset;
        auto locXYZ = Clusterer::getClusterLocalCoordinates(c, pattForCluster, yPlaneMLOT);

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
        this->mMinR[iLayer] = std::min(this->mMinR[iLayer], r);
        this->mMaxR[iLayer] = std::max(this->mMaxR[iLayer], r);

        const float sigmaY2 = (c.subDetID == 0)
                                ? 0.25f * SegmentationChip::PitchRowVD * SegmentationChip::PitchRowVD
                                : 0.25f * SegmentationChip::PitchRowMLOT * SegmentationChip::PitchRowMLOT;
        const float sigmaZ2 = (c.subDetID == 0)
                                ? 0.25f * SegmentationChip::PitchColVD * SegmentationChip::PitchColVD
                                : 0.25f * SegmentationChip::PitchColMLOT * SegmentationChip::PitchColMLOT;

        this->addTrackingFrameInfoToLayer(iLayer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), trkXYZ.x(), alpha,
                                          std::array<float, 2>{trkXYZ.y(), trkXYZ.z()},
                                          std::array<float, 3>{sigmaY2, 0.f, sigmaZ2});
        this->addClusterToLayer(iLayer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), this->mUnsortedClusters[iLayer].size());
        this->addClusterExternalIndexToLayer(iLayer, clusterId);
        this->mClusterSize[iLayer].push_back(std::clamp(static_cast<unsigned int>(c.size), 0u, 255u));
      }

      this->mROFramesClusters[iLayer][iRof + 1] = this->mUnsortedClusters[iLayer].size();
    }
  }

  for (auto i = 0; i < this->mNTrackletsPerCluster.size(); ++i) {
    this->mNTrackletsPerCluster[i].resize(this->mUnsortedClusters[1].size());
    this->mNTrackletsPerClusterSum[i].resize(this->mUnsortedClusters[1].size() + 1);
  }

  if (mcLabels != nullptr) {
    for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
      this->mClusterLabels[iLayer] = (*mcLabels)[iLayer];
    }
  }

  return nRofs;
}

template <int nLayers, class Base>
void TimeFrameMixin<nLayers, Base>::getPrimaryVerticesFromMC(TTree* mcHeaderTree, int nRofs, Long64_t nEvents, int inROFpileup)
{
  auto mcheader = new o2::dataformats::MCEventHeader;
  mcHeaderTree->SetBranchAddress("MCEventHeader.", &mcheader);

  this->mPrimaryVertices.clear();
  this->mPrimaryVerticesLabels.clear();

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
  updateHostROFVertexLookupTable();
}

template <int nLayers, class Base>
void TimeFrameMixin<nLayers, Base>::addTruthSeedingVertices()
{
  LOGP(info, "TRK: using truth seeds as vertices from DigitizationContext");
  this->mPrimaryVertices.clear();
  this->mPrimaryVerticesLabels.clear();

  const auto dc = o2::steer::DigitizationContext::loadFromFile("collisioncontext.root");
  const auto irs = dc->getEventRecords();
  o2::steer::MCKinematicsReader mcReader(dc);

  const int64_t anchorBC = mTFAnchorIR.toLong();
  const auto& clockLayer = this->getROFOverlapTableView().getClockLayer();
  const auto rofLength = clockLayer.mROFLength;

  using Vertex = o2::its::Vertex;
  struct VertEntry {
    int64_t bc;
    Vertex vertex;
    int event;
  };
  std::vector<VertEntry> entries;

  const int iSrc = 0;
  auto eveId2colId = dc->getCollisionIndicesForSource(iSrc);
  for (int iEve{0}; iEve < mcReader.getNEvents(iSrc); ++iEve) {
    const auto& ir = irs[eveId2colId[iEve]];
    if (!ir.isDummy()) {
      const auto& eve = mcReader.getMCEventHeader(iSrc, iEve);
      const int64_t evBC = ir.toLong() - anchorBC;
      if (evBC >= 0) {
        Vertex vert;
        vert.setTimeStamp(o2::its::TimeEstBC{
          static_cast<o2::its::TimeStampType>(evBC),
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
        entries.push_back({evBC, vert, iEve});
      }
    }
    mcReader.releaseTracksForSourceAndEvent(iSrc, iEve);
  }

  // Sort by BC so the lookup table binary search works correctly
  std::ranges::sort(entries, {}, &VertEntry::bc);

  for (const auto& e : entries) {
    this->addPrimaryVertex(e.vertex);
    o2::MCCompLabel lbl(o2::MCCompLabel::maxTrackID(), e.event, iSrc, false);
    this->addPrimaryVertexLabel({lbl, 1.f});
  }
  updateHostROFVertexLookupTable();
  LOGP(info, "TRK truth seeding: added {} vertices", entries.size());
}

} // namespace o2::trk

#endif // ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAMEMIXIN_H
