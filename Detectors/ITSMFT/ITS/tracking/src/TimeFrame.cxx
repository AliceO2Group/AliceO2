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
#include <sstream>

#include "Framework/Logger.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/MathUtils.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/TrackingConfigParam.h"

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

template <int nLayers>
void TimeFrame<nLayers>::addPrimaryVertices(const bounded_vector<Vertex>& vertices, const int iteration)
{
  for (const auto& vertex : vertices) {
    mPrimaryVertices.emplace_back(vertex); // put a copy in the present
    mTotVertPerIteration[iteration]++;
    if (!isBeamPositionOverridden) { // beam position is updated only at first occurrence of the vertex. A bit sketchy if we have past/future vertices, it should not impact too much.
      const float w = vertex.getNContributors();
      mBeamPos[0] = (mBeamPos[0] * mBeamPosWeight + vertex.getX() * w) / (mBeamPosWeight + w);
      mBeamPos[1] = (mBeamPos[1] * mBeamPosWeight + vertex.getY() * w) / (mBeamPosWeight + w);
      mBeamPosWeight += w;
    }
  }
  mROFramesPV.push_back(mPrimaryVertices.size()); // current rof must have number of vertices up to present
}

template <int nLayers>
void TimeFrame<nLayers>::addPrimaryVerticesLabels(bounded_vector<std::pair<MCCompLabel, float>>& labels)
{
  mVerticesMCRecInfo.insert(mVerticesMCRecInfo.end(), labels.begin(), labels.end());
}

template <int nLayers>
void TimeFrame<nLayers>::addPrimaryVerticesContributorLabels(bounded_vector<MCCompLabel>& labels)
{
  mVerticesContributorLabels.insert(mVerticesContributorLabels.end(), labels.begin(), labels.end());
}

template <int nLayers>
void TimeFrame<nLayers>::addPrimaryVerticesInROF(const bounded_vector<Vertex>& vertices, const int rofId, const int iteration)
{
  mPrimaryVertices.insert(mPrimaryVertices.begin() + mROFramesPV[rofId], vertices.begin(), vertices.end());
  for (int i = rofId + 1; i < mROFramesPV.size(); ++i) {
    mROFramesPV[i] += vertices.size();
  }
  mTotVertPerIteration[iteration] += vertices.size();
}

template <int nLayers>
void TimeFrame<nLayers>::addPrimaryVerticesLabelsInROF(const bounded_vector<std::pair<MCCompLabel, float>>& labels, const int rofId)
{
  mVerticesMCRecInfo.insert(mVerticesMCRecInfo.begin() + mROFramesPV[rofId], labels.begin(), labels.end());
}

template <int nLayers>
void TimeFrame<nLayers>::addPrimaryVerticesContributorLabelsInROF(const bounded_vector<MCCompLabel>& labels, const int rofId)
{
  // count the number of cont. in rofs before and including the target rof
  unsigned int n{0};
  const auto& pvs = getPrimaryVertices(0, rofId);
  for (const auto& pv : pvs) {
    n += pv.getNContributors();
  }
  mVerticesContributorLabels.insert(mVerticesContributorLabels.begin() + n, labels.begin(), labels.end());
}

template <int nLayers>
int TimeFrame<nLayers>::loadROFrameData(gsl::span<const o2::itsmft::ROFRecord> rofs,
                                        gsl::span<const itsmft::CompClusterExt> clusters,
                                        gsl::span<const unsigned char>::iterator& pattIt,
                                        const itsmft::TopologyDictionary* dict,
                                        const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

  resetROFrameData(rofs.size());
  prepareROFrameData(rofs, clusters);

  for (size_t iRof{0}; iRof < rofs.size(); ++iRof) {
    const auto& rof = rofs[iRof];
    for (int clusterId{rof.getFirstEntry()}; clusterId < rof.getFirstEntry() + rof.getNEntries(); ++clusterId) {
      const auto& c = clusters[clusterId];

      int layer = geom->getLayer(c.getSensorID());

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
      mClusterSize[clusterId] = std::clamp(clusterSize, 0u, 255u);
      auto sensorID = c.getSensorID();
      // Inverse transformation to the local --> tracking
      auto trkXYZ = geom->getMatrixT2L(sensorID) ^ locXYZ;
      // Transformation to the local --> global
      auto gloXYZ = geom->getMatrixL2G(sensorID) * locXYZ;

      addTrackingFrameInfoToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), trkXYZ.x(), geom->getSensorRefAlpha(sensorID),
                                  std::array<float, 2>{trkXYZ.y(), trkXYZ.z()},
                                  std::array<float, 3>{sigmaY2, sigmaYZ, sigmaZ2});
      /// Rotate to the global frame
      addClusterToLayer(layer, gloXYZ.x(), gloXYZ.y(), gloXYZ.z(), mUnsortedClusters[layer].size());
      addClusterExternalIndexToLayer(layer, clusterId);
    }
    for (unsigned int iL{0}; iL < mUnsortedClusters.size(); ++iL) {
      mROFramesClusters[iL][iRof + 1] = mUnsortedClusters[iL].size(); // effectively calculating and exclusive sum
    }
  }

  for (auto i = 0; i < mNTrackletsPerCluster.size(); ++i) {
    mNTrackletsPerCluster[i].resize(mUnsortedClusters[1].size());
    mNTrackletsPerClusterSum[i].resize(mUnsortedClusters[1].size() + 1); // Exc sum "prepends" a 0
  }

  if (mcLabels != nullptr) {
    mClusterLabels = mcLabels;
  }

  return mNrof;
}

template <int nLayers>
void TimeFrame<nLayers>::resetROFrameData(size_t nRofs)
{
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    deepVectorClear(mUnsortedClusters[iLayer], getMaybeFrameworkHostResource());
    deepVectorClear(mTrackingFrameInfo[iLayer], getMaybeFrameworkHostResource());
    clearResizeBoundedVector(mROFramesClusters[iLayer], nRofs + 1, getMaybeFrameworkHostResource());
    deepVectorClear(mClusterExternalIndices[iLayer], mMemoryPool.get());

    if (iLayer < 2) {
      deepVectorClear(mTrackletsIndexROF[iLayer], mMemoryPool.get());
      deepVectorClear(mNTrackletsPerCluster[iLayer], mMemoryPool.get());
      deepVectorClear(mNTrackletsPerClusterSum[iLayer], mMemoryPool.get());
    }
  }
}

template <int nLayers>
void TimeFrame<nLayers>::prepareROFrameData(gsl::span<const o2::itsmft::ROFRecord> rofs,
                                            gsl::span<const itsmft::CompClusterExt> clusters)
{
  GeometryTGeo* geom = GeometryTGeo::Instance();
  mNrof = rofs.size();
  clearResizeBoundedVector(mClusterSize, clusters.size(), mMemoryPool.get());
  std::array<int, nLayers> clusterCountPerLayer{};
  for (const auto& clus : clusters) {
    ++clusterCountPerLayer[geom->getLayer(clus.getSensorID())];
  }
  for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
    mUnsortedClusters[iLayer].reserve(clusterCountPerLayer[iLayer]);
    mTrackingFrameInfo[iLayer].reserve(clusterCountPerLayer[iLayer]);
    mClusterExternalIndices[iLayer].reserve(clusterCountPerLayer[iLayer]);
  }
}

template <int nLayers>
void TimeFrame<nLayers>::prepareClusters(const TrackingParameters& trkParam, const int maxLayers)
{
  const int numBins{trkParam.PhiBins * trkParam.ZBins};
  const int stride{numBins + 1};
  bounded_vector<ClusterHelper> cHelper(mMemoryPool.get());
  bounded_vector<int> clsPerBin(numBins, 0, mMemoryPool.get());
  bounded_vector<int> lutPerBin(numBins, 0, mMemoryPool.get());
  for (int rof{0}; rof < mNrof; ++rof) {
    if ((int)mMultiplicityCutMask.size() == mNrof && !mMultiplicityCutMask[rof]) {
      continue;
    }
    for (int iLayer{0}, stopLayer = std::min(trkParam.NLayers, maxLayers); iLayer < stopLayer; ++iLayer) {
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

template <int nLayers>
void TimeFrame<nLayers>::initialise(const int iteration, const TrackingParameters& trkParam, const int maxLayers, bool resetVertices)
{
  if (iteration == 0) {
    if (maxLayers < trkParam.NLayers && resetVertices) {
      resetRofPV();
      deepVectorClear(mTotVertPerIteration);
    }
    deepVectorClear(mTracks);
    deepVectorClear(mTracksLabel);
    deepVectorClear(mLines);
    deepVectorClear(mLinesLabels);
    if (resetVertices) {
      deepVectorClear(mVerticesMCRecInfo);
      deepVectorClear(mVerticesContributorLabels);
    }
    clearResizeBoundedVector(mTracks, mNrof, mMemoryPool.get());
    clearResizeBoundedVector(mTracksLabel, mNrof, mMemoryPool.get());
    clearResizeBoundedVector(mLinesLabels, mNrof, mMemoryPool.get());
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
      clearResizeBoundedVector(mClusters[iLayer], mUnsortedClusters[iLayer].size(), getMaybeFrameworkHostResource(maxLayers != nLayers));
      clearResizeBoundedVector(mUsedClusters[iLayer], mUnsortedClusters[iLayer].size(), getMaybeFrameworkHostResource(maxLayers != nLayers));
      mPositionResolution[iLayer] = o2::gpu::CAMath::Sqrt(0.5f * (trkParam.SystErrorZ2[iLayer] + trkParam.SystErrorY2[iLayer]) + trkParam.LayerResolution[iLayer] * trkParam.LayerResolution[iLayer]);
    }
    clearResizeBoundedArray(mIndexTables, mNrof * (trkParam.ZBins * trkParam.PhiBins + 1), getMaybeFrameworkHostResource(maxLayers != nLayers));
    clearResizeBoundedVector(mLines, mNrof, mMemoryPool.get());
    clearResizeBoundedVector(mTrackletClusters, mNrof, mMemoryPool.get());

    for (int iLayer{0}; iLayer < trkParam.NLayers; ++iLayer) {
      if (trkParam.SystErrorY2[iLayer] > 0.f || trkParam.SystErrorZ2[iLayer] > 0.f) {
        for (auto& tfInfo : mTrackingFrameInfo[iLayer]) {
          /// Account for alignment systematics in the cluster covariance matrix
          tfInfo.covarianceTrackingFrame[0] += trkParam.SystErrorY2[iLayer];
          tfInfo.covarianceTrackingFrame[2] += trkParam.SystErrorZ2[iLayer];
        }
      }
    }
    mMinR.fill(10000.);
    mMaxR.fill(-1.);
  }
  mNTrackletsPerROF.resize(2);
  for (auto& v : mNTrackletsPerROF) {
    v = bounded_vector<int>(mNrof + 1, 0, mMemoryPool.get());
  }
  if (iteration == 0 || iteration == 3) {
    prepareClusters(trkParam, maxLayers);
  }
  mTotalTracklets = {0, 0};
  if (maxLayers < trkParam.NLayers) { // Vertexer only, but in both iterations
    for (size_t iLayer{0}; iLayer < maxLayers; ++iLayer) {
      deepVectorClear(mUsedClusters[iLayer]);
      clearResizeBoundedVector(mUsedClusters[iLayer], mUnsortedClusters[iLayer].size(), mMemoryPool.get());
    }
  }

  mTotVertPerIteration.resize(1 + iteration);
  mNoVertexROF = 0;
  deepVectorClear(mRoads);
  deepVectorClear(mRoadLabels);

  mMSangles.resize(trkParam.NLayers);
  mPhiCuts.resize(mClusters.size() - 1, 0.f);

  float oneOverR{0.001f * 0.3f * std::abs(mBz) / trkParam.TrackletMinPt};
  for (unsigned int iLayer{0}; iLayer < nLayers; ++iLayer) {
    mMSangles[iLayer] = math_utils::MSangle(0.14f, trkParam.TrackletMinPt, trkParam.LayerxX0[iLayer]);
    mPositionResolution[iLayer] = o2::gpu::CAMath::Sqrt(0.5f * (trkParam.SystErrorZ2[iLayer] + trkParam.SystErrorY2[iLayer]) + trkParam.LayerResolution[iLayer] * trkParam.LayerResolution[iLayer]);
    if (iLayer < mClusters.size() - 1) {
      const float& r1 = trkParam.LayerRadii[iLayer];
      const float& r2 = trkParam.LayerRadii[iLayer + 1];
      const float res1 = o2::gpu::CAMath::Hypot(trkParam.PVres, mPositionResolution[iLayer]);
      const float res2 = o2::gpu::CAMath::Hypot(trkParam.PVres, mPositionResolution[iLayer + 1]);
      const float cosTheta1half = o2::gpu::CAMath::Sqrt(1.f - math_utils::Sq(0.5f * r1 * oneOverR));
      const float cosTheta2half = o2::gpu::CAMath::Sqrt(1.f - math_utils::Sq(0.5f * r2 * oneOverR));
      float x = r2 * cosTheta1half - r1 * cosTheta2half;
      float delta = o2::gpu::CAMath::Sqrt(1.f / (1.f - 0.25f * math_utils::Sq(x * oneOverR)) * (math_utils::Sq(0.25f * r1 * r2 * math_utils::Sq(oneOverR) / cosTheta2half + cosTheta1half) * math_utils::Sq(res1) + math_utils::Sq(0.25f * r1 * r2 * math_utils::Sq(oneOverR) / cosTheta1half + cosTheta2half) * math_utils::Sq(res2)));
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

template <int nLayers>
unsigned long TimeFrame<nLayers>::getArtefactsMemory() const
{
  unsigned long size{0};
  for (const auto& trkl : mTracklets) {
    size += sizeof(Tracklet) * trkl.size();
  }
  for (const auto& cells : mCells) {
    size += sizeof(CellSeedN) * cells.size();
  }
  for (const auto& cellsN : mCellsNeighbours) {
    size += sizeof(int) * cellsN.size();
  }
  return size + sizeof(Road<nLayers - 2>) * mRoads.size();
}

template <int nLayers>
void TimeFrame<nLayers>::printArtefactsMemory() const
{
  LOGP(info, "TimeFrame: Artefacts occupy {:.2f} MB", getArtefactsMemory() / constants::MB);
}

template <int nLayers>
void TimeFrame<nLayers>::fillPrimaryVerticesXandAlpha()
{
  deepVectorClear(mPValphaX);
  mPValphaX.reserve(mPrimaryVertices.size());
  for (auto& pv : mPrimaryVertices) {
    mPValphaX.emplace_back(std::array<float, 2>{o2::gpu::CAMath::Hypot(pv.getX(), pv.getY()), math_utils::computePhi(pv.getX(), pv.getY())});
  }
}

template <int nLayers>
void TimeFrame<nLayers>::computeTrackletsPerROFScans()
{
  for (ushort iLayer = 0; iLayer < 2; ++iLayer) {
    for (unsigned int iRof{0}; iRof < mNrof; ++iRof) {
      if (mMultiplicityCutMask[iRof]) {
        mTotalTracklets[iLayer] += mNTrackletsPerROF[iLayer][iRof];
      }
    }
    std::exclusive_scan(mNTrackletsPerROF[iLayer].begin(), mNTrackletsPerROF[iLayer].end(), mNTrackletsPerROF[iLayer].begin(), 0);
    std::exclusive_scan(mNTrackletsPerCluster[iLayer].begin(), mNTrackletsPerCluster[iLayer].end(), mNTrackletsPerClusterSum[iLayer].begin(), 0);
  }
}

template <int nLayers>
void TimeFrame<nLayers>::checkTrackletLUTs()
{
  for (uint32_t iLayer{0}; iLayer < getTracklets().size(); ++iLayer) {
    int prev{-1};
    int count{0};
    for (uint32_t iTracklet{0}; iTracklet < getTracklets()[iLayer].size(); ++iTracklet) {
      auto& trk = getTracklets()[iLayer][iTracklet];
      int currentId{trk.firstClusterIndex};
      if (currentId < prev) {
        LOG(info) << "First Cluster Index not increasing monotonically on L:T:ID:Prev " << iLayer << "\t" << iTracklet << "\t" << currentId << "\t" << prev;
      } else if (currentId == prev) {
        count++;
      } else {
        if (iLayer > 0) {
          auto& lut{getTrackletsLookupTable()[iLayer - 1]};
          if (count != lut[prev + 1] - lut[prev]) {
            LOG(info) << "LUT count broken " << iLayer - 1 << "\t" << prev << "\t" << count << "\t" << lut[prev + 1] << "\t" << lut[prev];
          }
        }
        count = 1;
      }
      prev = currentId;
      if (iLayer > 0) {
        auto& lut{getTrackletsLookupTable()[iLayer - 1]};
        if (iTracklet >= (uint32_t)(lut[currentId + 1]) || iTracklet < (uint32_t)(lut[currentId])) {
          LOG(info) << "LUT broken: " << iLayer - 1 << "\t" << currentId << "\t" << iTracklet;
        }
      }
    }
  }
}

template <int nLayers>
void TimeFrame<nLayers>::printTrackletLUTonLayer(int i)
{
  LOG(info) << "-------- Tracklet LUT " << i;
  std::stringstream s;
  for (int j : mTrackletsLookupTable[i]) {
    s << j << "\t";
  }
  LOG(info) << s.str();
  LOG(info) << "--------";
}

template <int nLayers>
void TimeFrame<nLayers>::printCellLUTonLayer(int i)
{
  LOG(info) << "-------- Cell LUT " << i;
  std::stringstream s;
  for (int j : mCellsLookupTable[i]) {
    s << j << "\t";
  }
  LOG(info) << s.str();
  LOG(info) << "--------";
}

template <int nLayers>
void TimeFrame<nLayers>::printTrackletLUTs()
{
  for (unsigned int i{0}; i < mTrackletsLookupTable.size(); ++i) {
    printTrackletLUTonLayer(i);
  }
}

template <int nLayers>
void TimeFrame<nLayers>::printCellLUTs()
{
  for (unsigned int i{0}; i < mCellsLookupTable.size(); ++i) {
    printCellLUTonLayer(i);
  }
}

template <int nLayers>
void TimeFrame<nLayers>::printVertices()
{
  LOG(info) << "Vertices in ROF (nROF = " << mNrof << ", lut size = " << mROFramesPV.size() << ")";
  for (unsigned int iR{0}; iR < mROFramesPV.size(); ++iR) {
    LOG(info) << mROFramesPV[iR] << "\t";
  }
  LOG(info) << "\n\n Vertices:";
  for (unsigned int iV{0}; iV < mPrimaryVertices.size(); ++iV) {
    LOG(info) << mPrimaryVertices[iV].getX() << "\t" << mPrimaryVertices[iV].getY() << "\t" << mPrimaryVertices[iV].getZ();
  }
  LOG(info) << "--------";
}

template <int nLayers>
void TimeFrame<nLayers>::printROFoffsets()
{
  LOG(info) << "--------";
  for (unsigned int iLayer{0}; iLayer < mROFramesClusters.size(); ++iLayer) {
    LOG(info) << "Layer " << iLayer;
    std::stringstream s;
    for (auto value : mROFramesClusters[iLayer]) {
      s << value << "\t";
    }
    LOG(info) << s.str();
  }
}

template <int nLayers>
void TimeFrame<nLayers>::printNClsPerROF()
{
  LOG(info) << "--------";
  for (unsigned int iLayer{0}; iLayer < mNClustersPerROF.size(); ++iLayer) {
    LOG(info) << "Layer " << iLayer;
    std::stringstream s;
    for (auto& value : mNClustersPerROF[iLayer]) {
      s << value << "\t";
    }
    LOG(info) << s.str();
  }
}

template <int nLayers>
void TimeFrame<nLayers>::printSliceInfo(const int startROF, const int sliceSize)
{
  LOG(info) << "Dumping slice of " << sliceSize << " rofs:";
  for (int iROF{startROF}; iROF < startROF + sliceSize; ++iROF) {
    LOG(info) << "ROF " << iROF << " dump:";
    for (unsigned int iLayer{0}; iLayer < mClusters.size(); ++iLayer) {
      LOG(info) << "Layer " << iLayer << " has: " << getClustersOnLayer(iROF, iLayer).size() << " clusters.";
    }
    LOG(info) << "Number of seeding vertices: " << getPrimaryVertices(iROF).size();
    int iVertex{0};
    for (auto& v : getPrimaryVertices(iROF)) {
      LOG(info) << "\t vertex " << iVertex++ << ": x=" << v.getX() << " "
                << " y=" << v.getY() << " z=" << v.getZ() << " has " << v.getNContributors() << " contributors.";
    }
  }
}

template <int nLayers>
void TimeFrame<nLayers>::setMemoryPool(std::shared_ptr<BoundedMemoryResource> pool)
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
  initVector(mTotVertPerIteration);
  initContainers(mClusterExternalIndices);
  initContainers(mNTrackletsPerCluster);
  initContainers(mNTrackletsPerClusterSum);
  initContainers(mNClustersPerROF);
  initVector(mROFramesPV);
  initVector(mPrimaryVertices);
  initVector(mRoads);
  initVector(mMSangles);
  initVector(mPhiCuts);
  initVector(mPositionResolution);
  initVector(mClusterSize);
  initVector(mPValphaX);
  initVector(mBogusClusters);
  initContainers(mTrackletsIndexROF);
  initContainers(mTracks);
  initContainers(mTracklets);
  initContainers(mCells);
  initContainers(mCellsNeighbours);
  initContainers(mCellsLookupTable);
  // MC info (we don't know if we have MC)
  initVector(mVerticesContributorLabels);
  initContainers(mLinesLabels);
  initContainers(mTrackletLabels);
  initContainers(mCellLabels);
  initVector(mRoadLabels);
  initContainers(mTracksLabel);
  // these will use possibly an externally provided allocator
  initContainers(mClusters, hasFrameworkAllocator());
  initContainers(mUsedClusters, hasFrameworkAllocator());
  initContainers(mUnsortedClusters, hasFrameworkAllocator());
  initContainers(mIndexTables, hasFrameworkAllocator());
  initContainers(mTrackingFrameInfo, hasFrameworkAllocator());
  initContainers(mROFramesClusters, hasFrameworkAllocator());
}

template <int nLayers>
void TimeFrame<nLayers>::setFrameworkAllocator(ExternalAllocator* ext)
{
  mExternalAllocator = ext;
  mExtMemoryPool = std::make_shared<BoundedMemoryResource>(mExternalAllocator);
}

template <int nLayers>
void TimeFrame<nLayers>::wipe()
{
  deepVectorClear(mTracks);
  deepVectorClear(mTracklets);
  deepVectorClear(mCells);
  deepVectorClear(mRoads);
  deepVectorClear(mCellsNeighbours);
  deepVectorClear(mCellsLookupTable);
  deepVectorClear(mTotVertPerIteration);
  deepVectorClear(mPrimaryVertices);
  deepVectorClear(mTrackletsLookupTable);
  deepVectorClear(mClusterExternalIndices);
  deepVectorClear(mNTrackletsPerCluster);
  deepVectorClear(mNTrackletsPerClusterSum);
  deepVectorClear(mNClustersPerROF);
  deepVectorClear(mROFramesPV);
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
    deepVectorClear(mVerticesContributorLabels);
    deepVectorClear(mTrackletLabels);
    deepVectorClear(mCellLabels);
    deepVectorClear(mRoadLabels);
    deepVectorClear(mTracksLabel);
  }
}

template class TimeFrame<7>;

} // namespace o2::its
