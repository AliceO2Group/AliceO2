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
/// \brief
///

#include <numeric>

#include "Framework/Logger.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/MathUtils.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITStracking/BoundedAllocator.h"

namespace
{
struct ClusterHelper {
  float phi;
  float r;
  int bin;
  int ind;
};
} // namespace

namespace o2::its
{

constexpr float DefClusErrorRow = o2::itsmft::SegmentationAlpide::PitchRow * 0.5;
constexpr float DefClusErrorCol = o2::itsmft::SegmentationAlpide::PitchCol * 0.5;
constexpr float DefClusError2Row = DefClusErrorRow * DefClusErrorRow;
constexpr float DefClusError2Col = DefClusErrorCol * DefClusErrorCol;

template <int NLayers>
void TimeFrame<NLayers>::addPrimaryVertex(const Vertex& vert)
{
  mPrimaryVertices.emplace_back(vert);
  if (!isBeamPositionOverridden) {
    const float w = vert.getNContributors();
    mBeamPos[0] = (mBeamPos[0] * mBeamPosWeight + vert.getX() * w) / (mBeamPosWeight + w);
    mBeamPos[1] = (mBeamPos[1] * mBeamPosWeight + vert.getY() * w) / (mBeamPosWeight + w);
    mBeamPosWeight += w;
  }
}

template <int NLayers>
void TimeFrame<NLayers>::loadROFrameData(gsl::span<const o2::itsmft::ROFRecord> rofs,
                                         gsl::span<const itsmft::CompClusterExt> clusters,
                                         gsl::span<const unsigned char>::iterator& pattIt,
                                         const itsmft::TopologyDictionary* dict,
                                         int layer,
                                         const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  resetROFrameData(layer);
  prepareROFrameData(clusters, layer);

  // check for missing/empty/unset rofs
  // the code requires consistent monotonically increasing input without gaps
  const auto& timing = mROFOverlapTableView.getLayer(layer >= 0 ? layer : 0);
  if (timing.mNROFsTF != rofs.size()) {
    LOGP(fatal, "Received inconsistent number of rofs on layer:{} expected:{} received:{}", layer, timing.mNROFsTF, rofs.size());
  }

  for (int32_t iRof{0}; iRof < rofs.size(); ++iRof) {
    const auto& rof = rofs[iRof];
    for (int clusterId{rof.getFirstEntry()}; clusterId < rof.getFirstEntry() + rof.getNEntries(); ++clusterId) {
      const auto& c = clusters[clusterId];
      int lay = geom->getLayer(c.getSensorID());
      auto pattID = c.getPatternID();
      o2::math_utils::Point3D<float> locXYZ;
      float sigmaY2 = DefClusError2Row, sigmaZ2 = DefClusError2Col, sigmaYZ = 0; // Dummy COG errors (about half pixel size)
      unsigned int clusterSize{0};
      if (pattID != itsmft::CompCluster::InvalidPatternID) {
        sigmaY2 = dict->getErr2X(pattID);
        sigmaZ2 = dict->getErr2Z(pattID);
        if (!dict->isGroup(pattID)) {
          locXYZ = dict->getClusterCoordinates(c);
          clusterSize = dict->getNpixels(pattID);
        } else {
          o2::itsmft::ClusterPattern patt(pattIt);
          locXYZ = dict->getClusterCoordinates(c, patt);
          clusterSize = patt.getNPixels();
        }
      } else {
        o2::itsmft::ClusterPattern patt(pattIt);
        locXYZ = dict->getClusterCoordinates(c, patt, false);
        clusterSize = patt.getNPixels();
      }
      mClusterSize[layer >= 0 ? layer : 0][clusterId] = std::clamp(clusterSize, 0u, 255u);
      auto sensorID = c.getSensorID();
      // Inverse transformation to the local --> tracking
      auto trkXYZ = geom->getMatrixT2L(sensorID) ^ locXYZ;
      // Transformation to the local --> global
      auto gloXYZ = geom->getMatrixL2G(sensorID) * locXYZ;
      addTrackingFrameInfoToLayer(lay, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), trkXYZ.x(), geom->getSensorRefAlpha(sensorID),
                                  std::array<float, 2>{trkXYZ.y(), trkXYZ.z()},
                                  std::array<float, 3>{sigmaY2, sigmaYZ, sigmaZ2});
      /// Rotate to the global frame
      addClusterToLayer(lay, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), mUnsortedClusters[lay].size());
      addClusterExternalIndexToLayer(lay, clusterId);
    }
    // effectively calculating an exclusive sum
    if (layer >= 0) {
      mROFramesClusters[layer][iRof + 1] = mUnsortedClusters[layer].size();
    } else {
      for (unsigned int iL{0}; iL < mUnsortedClusters.size(); ++iL) {
        mROFramesClusters[iL][iRof + 1] = mUnsortedClusters[iL].size();
      }
    }
  }

  if (layer == 1 || layer == -1) {
    for (auto i = 0; i < mNTrackletsPerCluster.size(); ++i) {
      mNTrackletsPerCluster[i].resize(mUnsortedClusters[1].size());
      mNTrackletsPerClusterSum[i].resize(mUnsortedClusters[1].size() + 1);
    }
  }

  if (mcLabels != nullptr) {
    mClusterLabels[layer >= 0 ? layer : 0] = mcLabels;
  } else {
    mClusterLabels[layer >= 0 ? layer : 0] = nullptr;
  }
}

template <int NLayers>
void TimeFrame<NLayers>::resetROFrameData(int layer)
{
  if (layer >= 0) {
    deepVectorClear(mUnsortedClusters[layer], getMaybeFrameworkHostResource());
    deepVectorClear(mTrackingFrameInfo[layer], getMaybeFrameworkHostResource());
    deepVectorClear(mClusterExternalIndices[layer], mMemoryPool.get());
    clearResizeBoundedVector(mROFramesClusters[layer], mROFOverlapTableView.getLayer(layer).mNROFsTF + 1, getMaybeFrameworkHostResource());
  } else {
    for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
      deepVectorClear(mUnsortedClusters[iLayer], getMaybeFrameworkHostResource());
      deepVectorClear(mTrackingFrameInfo[iLayer], getMaybeFrameworkHostResource());
      deepVectorClear(mClusterExternalIndices[iLayer], mMemoryPool.get());
      clearResizeBoundedVector(mROFramesClusters[iLayer], mROFOverlapTableView.getLayer(iLayer).mNROFsTF + 1, getMaybeFrameworkHostResource());
    }
  }
}

template <int NLayers>
void TimeFrame<NLayers>::prepareROFrameData(gsl::span<const itsmft::CompClusterExt> clusters, int layer)
{
  if (layer >= 0) {
    mUnsortedClusters[layer].reserve(clusters.size());
    mTrackingFrameInfo[layer].reserve(clusters.size());
    mClusterExternalIndices[layer].reserve(clusters.size());
    clearResizeBoundedVector(mClusterSize[layer], clusters.size(), mMemoryPool.get());
  } else {
    auto* geom = GeometryTGeo::Instance();
    clearResizeBoundedVector(mClusterSize[0], clusters.size(), mMemoryPool.get());
    std::array<size_t, NLayers> clusterCountPerLayer{0};
    for (const auto& cls : clusters) {
      ++clusterCountPerLayer[geom->getLayer(cls.getChipID())];
    }
    for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
      mUnsortedClusters[iLayer].reserve(clusterCountPerLayer[iLayer]);
      mTrackingFrameInfo[iLayer].reserve(clusterCountPerLayer[iLayer]);
      mClusterExternalIndices[iLayer].reserve(clusterCountPerLayer[iLayer]);
    }
  }
}

template <int NLayers>
void TimeFrame<NLayers>::prepareClusters(const TrackingParameters& trkParam, const int maxLayers)
{
  const int numBins{trkParam.PhiBins * trkParam.ZBins};
  const int stride{numBins + 1};
  bounded_vector<ClusterHelper> cHelper(mMemoryPool.get());
  bounded_vector<int> clsPerBin(numBins, 0, mMemoryPool.get());
  bounded_vector<int> lutPerBin(numBins, 0, mMemoryPool.get());
  for (int iLayer{0}, stopLayer = std::min(trkParam.NLayers, maxLayers); iLayer < stopLayer; ++iLayer) {
    for (int rof{0}; rof < getNrof(iLayer); ++rof) {
      if (!mROFMaskView.isROFEnabled(iLayer, rof)) {
        continue;
      }
      const auto& unsortedClusters{getUnsortedClustersOnLayer(rof, iLayer)};
      const int clustersNum{static_cast<int>(unsortedClusters.size())};
      auto* tableBase = mIndexTables[iLayer].data() + rof * stride;

      cHelper.resize(clustersNum);

      for (int iCluster{0}; iCluster < clustersNum; ++iCluster) {
        const Cluster& c = unsortedClusters[iCluster];
        ClusterHelper& h = cHelper[iCluster];

        const float x = c.xCoordinate - mBeamPos[0];
        const float y = c.yCoordinate - mBeamPos[1];
        const float z = c.zCoordinate;

        float phi = math_utils::computePhi(x, y);
        int zBin{mIndexTableUtils.getZBinIndex(iLayer, z)};
        if (zBin < 0 || zBin >= trkParam.ZBins) {
          zBin = std::clamp(zBin, 0, trkParam.ZBins - 1);
          mBogusClusters[iLayer]++;
        }
        int bin = mIndexTableUtils.getBinIndex(zBin, mIndexTableUtils.getPhiBinIndex(phi));
        h.phi = phi;
        h.r = math_utils::hypot(x, y);
        mMinR[iLayer] = o2::gpu::GPUCommonMath::Min(h.r, mMinR[iLayer]);
        mMaxR[iLayer] = o2::gpu::GPUCommonMath::Max(h.r, mMaxR[iLayer]);
        h.bin = bin;
        h.ind = clsPerBin[bin]++;
      }
      std::exclusive_scan(clsPerBin.begin(), clsPerBin.end(), lutPerBin.begin(), 0);

      auto clusters2beSorted{getClustersOnLayer(rof, iLayer)};
      for (int iCluster{0}; iCluster < clustersNum; ++iCluster) {
        const ClusterHelper& h = cHelper[iCluster];
        Cluster& c = clusters2beSorted[lutPerBin[h.bin] + h.ind];

        c = unsortedClusters[iCluster];
        c.phi = h.phi;
        c.radius = h.r;
        c.indexTableBinIndex = h.bin;
      }
      std::copy_n(lutPerBin.data(), clsPerBin.size(), tableBase);
      std::fill_n(tableBase + clsPerBin.size(), stride - clsPerBin.size(), clustersNum);

      std::fill(clsPerBin.begin(), clsPerBin.end(), 0);
      cHelper.clear();
    }
  }
}

template <int NLayers>
void TimeFrame<NLayers>::initialise(const TrackingParameters& trkParam, const int maxLayers)
{
  if (trkParam.PassFlags[IterationStep::FirstPass]) {
    deepVectorClear(mTracks);
    deepVectorClear(mTracksLabel);
    deepVectorClear(mLines);
    deepVectorClear(mLinesLabels);
    if (trkParam.PassFlags[IterationStep::ResetVertices]) {
      deepVectorClear(mPrimaryVertices);
      deepVectorClear(mPrimaryVerticesLabels);
    }
    clearResizeBoundedVector(mLinesLabels, getNrof(1), mMemoryPool.get());
    clearResizeBoundedVector(mCells, trkParam.CellsPerRoad(), mMemoryPool.get());
    clearResizeBoundedVector(mCellsLookupTable, trkParam.CellsPerRoad() - 1, mMemoryPool.get());
    clearResizeBoundedVector(mCellsNeighbours, trkParam.CellsPerRoad() - 1, mMemoryPool.get());
    clearResizeBoundedVector(mCellsNeighboursLUT, trkParam.CellsPerRoad() - 1, mMemoryPool.get());
    clearResizeBoundedVector(mCellLabels, trkParam.CellsPerRoad(), mMemoryPool.get());
    clearResizeBoundedVector(mTracklets, std::min(trkParam.TrackletsPerRoad(), maxLayers - 1), mMemoryPool.get());
    clearResizeBoundedVector(mTrackletLabels, trkParam.TrackletsPerRoad(), mMemoryPool.get());
    clearResizeBoundedVector(mTrackletsLookupTable, trkParam.TrackletsPerRoad(), mMemoryPool.get());
    mIndexTableUtils.setTrackingParameters(trkParam);
    clearResizeBoundedVector(mPositionResolution, trkParam.NLayers, mMemoryPool.get());
    clearResizeBoundedVector(mBogusClusters, trkParam.NLayers, mMemoryPool.get());
    deepVectorClear(mTrackletClusters);
    for (unsigned int iLayer{0}; iLayer < std::min((int)mClusters.size(), maxLayers); ++iLayer) {
      clearResizeBoundedVector(mClusters[iLayer], mUnsortedClusters[iLayer].size(), getMaybeFrameworkHostResource(maxLayers != NLayers));
      clearResizeBoundedVector(mUsedClusters[iLayer], mUnsortedClusters[iLayer].size(), getMaybeFrameworkHostResource(maxLayers != NLayers));
      mPositionResolution[iLayer] = o2::gpu::CAMath::Sqrt((0.5f * (trkParam.SystErrorZ2[iLayer] + trkParam.SystErrorY2[iLayer])) + (trkParam.LayerResolution[iLayer] * trkParam.LayerResolution[iLayer]));
    }
    clearResizeBoundedVector(mLines, getNrof(1), mMemoryPool.get());
    clearResizeBoundedVector(mTrackletClusters, getNrof(1), mMemoryPool.get());

    for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
      clearResizeBoundedVector(mIndexTables[iLayer], getNrof(iLayer) * ((trkParam.ZBins * trkParam.PhiBins) + 1), getMaybeFrameworkHostResource());
    }
    for (int iLayer{0}; iLayer < trkParam.NLayers; ++iLayer) {
      if (trkParam.SystErrorY2[iLayer] > 0.f || trkParam.SystErrorZ2[iLayer] > 0.f) {
        for (auto& tfInfo : mTrackingFrameInfo[iLayer]) {
          /// Account for alignment systematics in the cluster covariance matrix
          tfInfo.covarianceTrackingFrame[0] += trkParam.SystErrorY2[iLayer];
          tfInfo.covarianceTrackingFrame[2] += trkParam.SystErrorZ2[iLayer];
        }
      }
    }

    mMinR.fill(std::numeric_limits<float>::max());
    mMaxR.fill(std::numeric_limits<float>::min());
  }
  mNTrackletsPerROF.resize(2);
  for (auto& v : mNTrackletsPerROF) {
    v = bounded_vector<int>(getNrof(1) + 1, 0, mMemoryPool.get());
  }
  if (trkParam.PassFlags[IterationStep::RebuildClusterLUT]) {
    prepareClusters(trkParam, maxLayers);
  }
  mTotalTracklets = {0, 0};
  if (maxLayers < trkParam.NLayers) { // Vertexer only, but in both iterations
    for (size_t iLayer{0}; iLayer < maxLayers; ++iLayer) {
      deepVectorClear(mUsedClusters[iLayer]);
      clearResizeBoundedVector(mUsedClusters[iLayer], mUnsortedClusters[iLayer].size(), mMemoryPool.get());
    }
  }

  mMSangles.resize(trkParam.NLayers);
  mPhiCuts.resize(mClusters.size() - 1, 0.f);
  float oneOverR{0.001f * 0.3f * std::abs(mBz) / trkParam.TrackletMinPt};
  for (unsigned int iLayer{0}; iLayer < NLayers; ++iLayer) {
    mMSangles[iLayer] = math_utils::MSangle(0.14f, trkParam.TrackletMinPt, trkParam.LayerxX0[iLayer]);
    mPositionResolution[iLayer] = o2::gpu::CAMath::Sqrt((0.5f * (trkParam.SystErrorZ2[iLayer] + trkParam.SystErrorY2[iLayer])) + (trkParam.LayerResolution[iLayer] * trkParam.LayerResolution[iLayer]));
    if (iLayer < mClusters.size() - 1) {
      const float& r1 = trkParam.LayerRadii[iLayer];
      const float& r2 = trkParam.LayerRadii[iLayer + 1];
      oneOverR = (0.5 * oneOverR >= 1.f / r2) ? (2.f / r2) - o2::constants::math::Almost0 : oneOverR;
      const float res1 = o2::gpu::CAMath::Hypot(trkParam.PVres, mPositionResolution[iLayer]);
      const float res2 = o2::gpu::CAMath::Hypot(trkParam.PVres, mPositionResolution[iLayer + 1]);
      const float cosTheta1half = o2::gpu::CAMath::Sqrt(1.f - math_utils::Sq(0.5f * r1 * oneOverR));
      const float cosTheta2half = o2::gpu::CAMath::Sqrt(1.f - math_utils::Sq(0.5f * r2 * oneOverR));
      float x = (r2 * cosTheta1half) - (r1 * cosTheta2half);
      float delta = o2::gpu::CAMath::Sqrt(1.f / (1.f - 0.25f * math_utils::Sq(x * oneOverR)) * (math_utils::Sq((0.25f * r1 * r2 * math_utils::Sq(oneOverR) / cosTheta2half) + cosTheta1half) * math_utils::Sq(res1) + math_utils::Sq((0.25f * r1 * r2 * math_utils::Sq(oneOverR) / cosTheta1half) + cosTheta2half) * math_utils::Sq(res2)));
      /// the expression std::asin(0.5f * x * oneOverR) is equivalent to std::aCos(0.5f * r1 * oneOverR) - std::acos(0.5 * r2 * oneOverR)
      mPhiCuts[iLayer] = std::min(o2::gpu::CAMath::ASin(0.5f * x * oneOverR) + 2.f * mMSangles[iLayer] + delta, o2::constants::math::PI * 0.5f);
    }
  }

  for (int iLayer{0}; iLayer < std::min((int)mTracklets.size(), maxLayers); ++iLayer) {
    deepVectorClear(mTracklets[iLayer]);
    deepVectorClear(mTrackletLabels[iLayer]);
    if (iLayer < (int)mCells.size()) {
      deepVectorClear(mCells[iLayer]);
      deepVectorClear(mTrackletsLookupTable[iLayer]);
      mTrackletsLookupTable[iLayer].resize(mClusters[iLayer + 1].size() + 1, 0);
      deepVectorClear(mCellLabels[iLayer]);
    }

    if (iLayer < (int)mCells.size() - 1) {
      deepVectorClear(mCellsLookupTable[iLayer]);
      deepVectorClear(mCellsNeighbours[iLayer]);
      deepVectorClear(mCellsNeighboursLUT[iLayer]);
    }
  }
}

template <int NLayers>
unsigned long TimeFrame<NLayers>::getArtefactsMemory() const
{
  unsigned long size{0};
  for (const auto& trkl : mTracklets) {
    size += sizeof(Tracklet) * trkl.size();
  }
  for (const auto& cells : mCells) {
    size += sizeof(CellSeed) * cells.size();
  }
  for (const auto& cellsN : mCellsNeighbours) {
    size += sizeof(int) * cellsN.size();
  }
  return size;
}

template <int NLayers>
void TimeFrame<NLayers>::printArtefactsMemory() const
{
  LOGP(info, "TimeFrame: Artefacts occupy {:.2f} MB", getArtefactsMemory() / constants::MB);
}

template <int NLayers>
void TimeFrame<NLayers>::computeTrackletsPerROFScans()
{
  for (ushort iLayer = 0; iLayer < 2; ++iLayer) {
    for (unsigned int iRof{0}; iRof < getNrof(1); ++iRof) {
      if (mROFMaskView.isROFEnabled(1, iRof)) {
        mTotalTracklets[iLayer] += mNTrackletsPerROF[iLayer][iRof];
      }
    }
    std::exclusive_scan(mNTrackletsPerROF[iLayer].begin(), mNTrackletsPerROF[iLayer].end(), mNTrackletsPerROF[iLayer].begin(), 0);
    std::exclusive_scan(mNTrackletsPerCluster[iLayer].begin(), mNTrackletsPerCluster[iLayer].end(), mNTrackletsPerClusterSum[iLayer].begin(), 0);
  }
}

template <int NLayers>
void TimeFrame<NLayers>::setMemoryPool(std::shared_ptr<BoundedMemoryResource> pool)
{
  mMemoryPool = pool;

  auto initVector = [&]<typename T>(bounded_vector<T>& vec, bool useExternal = false) {
    std::pmr::memory_resource* mr = (useExternal) ? mExtMemoryPool.get() : mMemoryPool.get();
    deepVectorClear(vec, mr);
  };

  auto initContainers = [&]<typename Container>(Container& container, bool useExternal = false) {
    for (auto& v : container) {
      initVector(v, useExternal);
    }
  };

  // these will only reside on the host for the cpu part
  initContainers(mClusterExternalIndices);
  initContainers(mNTrackletsPerCluster);
  initContainers(mNTrackletsPerClusterSum);
  initContainers(mNClustersPerROF);
  initVector(mPrimaryVertices);
  initVector(mMSangles);
  initVector(mPhiCuts);
  initVector(mPositionResolution);
  initContainers(mClusterSize);
  initVector(mPValphaX);
  initVector(mBogusClusters);
  initContainers(mTrackletsIndexROF);
  initVector(mTracks);
  initContainers(mTracklets);
  initContainers(mCells);
  initContainers(mCellsNeighbours);
  initContainers(mCellsLookupTable);
  // MC info (we don't know if we have MC)
  initVector(mPrimaryVerticesLabels);
  initContainers(mLinesLabels);
  initContainers(mTrackletLabels);
  initContainers(mCellLabels);
  initVector(mTracksLabel);
  // these will use possibly an externally provided allocator
  initContainers(mClusters, hasFrameworkAllocator());
  initContainers(mUsedClusters, hasFrameworkAllocator());
  initContainers(mUnsortedClusters, hasFrameworkAllocator());
  initContainers(mIndexTables, hasFrameworkAllocator());
  initContainers(mTrackingFrameInfo, hasFrameworkAllocator());
  initContainers(mROFramesClusters, hasFrameworkAllocator());
}

template <int NLayers>
void TimeFrame<NLayers>::setFrameworkAllocator(ExternalAllocator* ext)
{
  mExternalAllocator = ext;
  mExtMemoryPool = std::make_shared<BoundedMemoryResource>(mExternalAllocator);
}

template <int NLayers>
void TimeFrame<NLayers>::wipe()
{
  deepVectorClear(mTracks);
  deepVectorClear(mTracklets);
  deepVectorClear(mCells);
  deepVectorClear(mCellsNeighbours);
  deepVectorClear(mCellsLookupTable);
  deepVectorClear(mPrimaryVertices);
  deepVectorClear(mTrackletsLookupTable);
  deepVectorClear(mClusterExternalIndices);
  deepVectorClear(mNTrackletsPerCluster);
  deepVectorClear(mNTrackletsPerClusterSum);
  deepVectorClear(mNClustersPerROF);
  deepVectorClear(mMSangles);
  deepVectorClear(mPhiCuts);
  deepVectorClear(mPositionResolution);
  deepVectorClear(mClusterSize);
  deepVectorClear(mPValphaX);
  deepVectorClear(mBogusClusters);
  deepVectorClear(mTrackletsIndexROF);
  deepVectorClear(mTrackletClusters);
  deepVectorClear(mLines);
  // if we use the external host allocator then the assumption is that we
  // don't clear the memory ourself
  if (!hasFrameworkAllocator()) {
    deepVectorClear(mClusters);
    deepVectorClear(mUsedClusters);
    deepVectorClear(mUnsortedClusters);
    deepVectorClear(mIndexTables);
    deepVectorClear(mTrackingFrameInfo);
    deepVectorClear(mROFramesClusters);
  }
  // only needed to clear if we have MC info
  if (hasMCinformation()) {
    deepVectorClear(mLinesLabels);
    deepVectorClear(mPrimaryVerticesLabels);
    deepVectorClear(mTrackletLabels);
    deepVectorClear(mCellLabels);
    deepVectorClear(mTracksLabel);
  }
}

template class TimeFrame<7>;
// ALICE3 upgrade
#ifdef ENABLE_UPGRADES
template class TimeFrame<11>;
#endif

} // namespace o2::its
