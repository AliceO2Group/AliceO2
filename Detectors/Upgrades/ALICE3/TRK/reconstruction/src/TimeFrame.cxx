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

#include "TRKReconstruction/TimeFrame.h"
#include "TRKSimulation/Hit.h"
#include "TRKBase/GeometryTGeo.h"
#include "Framework/Logger.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include <TTree.h>
#include <TRandom3.h>
#include <vector>
#include <array>

using o2::its::clearResizeBoundedVector;

namespace o2::trk
{

template <int NLayers>
int TimeFrame<NLayers>::loadROFsFromHitTree(TTree* hitsTree, GeometryTGeo* gman, const nlohmann::json& config)
{
  constexpr std::array<int, 2> startLayer{0, 3};
  const Long64_t nEvents = hitsTree->GetEntries();

  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L) | o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  std::vector<o2::trk::Hit>* trkHit = nullptr;
  hitsTree->SetBranchAddress("TRKHit", &trkHit);

  const int inROFpileup{config.contains("inROFpileup") ? config["inROFpileup"].get<int>() : 1};

  // Calculate number of ROFs
  const int nRofs = (nEvents + inROFpileup - 1) / inROFpileup;

  // Set up ROF timing for all layers (no staggering in TRK simulation, all layers read out together)
  constexpr uint32_t rofLength = 198; // ROF length in BC
  o2::its::ROFOverlapTable<NLayers> overlapTable;
  for (int iLayer = 0; iLayer < NLayers; ++iLayer) {
    overlapTable.defineLayer(iLayer, nRofs, rofLength, 0, 0, 0);
  }
  overlapTable.init();
  this->setROFOverlapTable(overlapTable);

  // Set up the vertex lookup table timing (pre-allocate, vertices will be filled later)
  o2::its::ROFVertexLookupTable<NLayers> vtxLookupTable;
  for (int iLayer = 0; iLayer < NLayers; ++iLayer) {
    vtxLookupTable.defineLayer(iLayer, nRofs, rofLength, 0, 0, 0);
  }
  vtxLookupTable.init(); // pre-allocate without vertices
  this->setROFVertexLookupTable(vtxLookupTable);

  // Reset and prepare ROF data structures
  for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
    this->mMinR[iLayer] = std::numeric_limits<float>::max();
    this->mMaxR[iLayer] = std::numeric_limits<float>::lowest();
    this->mROFramesClusters[iLayer].clear();
    this->mROFramesClusters[iLayer].resize(nRofs + 1, 0);
    this->mUnsortedClusters[iLayer].clear();
    this->mTrackingFrameInfo[iLayer].clear();
    this->mClusterExternalIndices[iLayer].clear();
  }

  // Pre-count hits to reserve memory efficiently
  std::array<int, NLayers> clusterCountPerLayer{};
  for (Long64_t iEvent = 0; iEvent < nEvents; ++iEvent) {
    hitsTree->GetEntry(iEvent);
    for (const auto& hit : *trkHit) {
      if (gman->getDisk(hit.GetDetectorID()) != -1) {
        continue; // skip non-barrel hits
      }
      int subDetID = gman->getSubDetID(hit.GetDetectorID());
      const int layer = startLayer[subDetID] + gman->getLayer(hit.GetDetectorID());
      if (layer >= NLayers) {
        continue;
      }
      ++clusterCountPerLayer[layer];
    }
  }

  // Reserve memory for all layers (mClusterSize is now per-layer)
  for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
    this->mUnsortedClusters[iLayer].reserve(clusterCountPerLayer[iLayer]);
    this->mTrackingFrameInfo[iLayer].reserve(clusterCountPerLayer[iLayer]);
    this->mClusterExternalIndices[iLayer].reserve(clusterCountPerLayer[iLayer]);
    clearResizeBoundedVector(this->mClusterSize[iLayer], clusterCountPerLayer[iLayer], this->mMemoryPool.get());
  }

  std::array<float, 11> resolution{0.001, 0.001, 0.001, 0.001, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004};
  if (config["geometry"]["pitch"].size() == static_cast<size_t>(NLayers)) {
    for (size_t iLayer{0}; iLayer < config["geometry"]["pitch"].size(); ++iLayer) {
      LOGP(info, "Setting resolution for layer {} from config", iLayer);
      LOGP(info, "Layer {} pitch {} cm", iLayer, config["geometry"]["pitch"][iLayer].get<float>());
      resolution[iLayer] = config["geometry"]["pitch"][iLayer].get<float>() / std::sqrt(12.f);
    }
  }
  LOGP(info, "Number of active parts in VD: {}", gman->getNumberOfActivePartsVD());

  // One shared MC label container for all layers
  auto* labels = new dataformats::MCTruthContainer<MCCompLabel>();

  int hitCounter{0};
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
      if (layer >= NLayers) {
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
      const int clusterIdxInLayer = this->mUnsortedClusters[layer].size();
      this->addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), clusterIdxInLayer);
      this->addClusterExternalIndexToLayer(layer, hitCounter);
      MCCompLabel label{hit.GetTrackID(), static_cast<int>(iEvent), 0};
      labels->addElement(hitCounter, label);
      this->mClusterSize[layer][clusterIdxInLayer] = 1;
      hitCounter++;
    }
    trkHit->clear();

    // Update ROF structure when we complete an ROF or reach the last event
    if ((iEvent + 1) % inROFpileup == 0 || iEvent == nEvents - 1) {
      iRof++;
      for (unsigned int iLayer{0}; iLayer < this->mUnsortedClusters.size(); ++iLayer) {
        this->mROFramesClusters[iLayer][iRof] = this->mUnsortedClusters[iLayer].size(); // effectively calculating an exclusive sum
      }
    }
  }

  // Set the shared labels container for all layers
  for (int iLayer = 0; iLayer < NLayers; ++iLayer) {
    this->mClusterLabels[iLayer] = labels;
  }

  return nRofs;
}

template <int NLayers>
void TimeFrame<NLayers>::getPrimaryVerticesFromMC(TTree* mcHeaderTree, int nRofs, Long64_t nEvents, int inROFpileup, uint32_t rofLength)
{
  auto mcheader = new o2::dataformats::MCEventHeader;
  mcHeaderTree->SetBranchAddress("MCEventHeader.", &mcheader);

  this->mPrimaryVertices.clear();

  int iRof{0};
  for (Long64_t iEvent = 0; iEvent < nEvents; ++iEvent) {
    mcHeaderTree->GetEntry(iEvent);
    o2::its::Vertex vertex;
    vertex.setXYZ(mcheader->GetX(), mcheader->GetY(), mcheader->GetZ());
    vertex.setNContributors(30);
    vertex.setChi2(0.f);

    // Set proper BC timestamp for vertex-ROF compatibility
    // The vertex timestamp is set to the center of its ROF with half-ROF as error
    const uint32_t rofCenter = static_cast<uint32_t>(rofLength * iRof + rofLength / 2);
    const uint16_t rofHalf = static_cast<uint16_t>(rofLength / 2);
    vertex.setTimeStamp({rofCenter, rofHalf});

    LOGP(debug, "ROF {}: Added primary vertex at ({}, {}, {}) with BC timestamp [{}, +/-{}]",
         iRof, mcheader->GetX(), mcheader->GetY(), mcheader->GetZ(), rofCenter, rofHalf);
    this->addPrimaryVertex(vertex);
    if ((iEvent + 1) % inROFpileup == 0 || iEvent == nEvents - 1) {
      iRof++;
    }
  }
  this->mMultiplicityCutMask.resetMask(1u); /// all ROFs are valid with MC primary vertices.

  // Update the vertex lookup table with the newly added vertices
  this->updateROFVertexLookupTable();
}

// Explicit template instantiation for TRK with 11 layers
template class TimeFrame<11>;

} // namespace o2::trk
