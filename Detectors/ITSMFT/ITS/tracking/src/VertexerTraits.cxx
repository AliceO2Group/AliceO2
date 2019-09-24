// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file VertexerTraits.cxx
/// \brief
/// \author matteo.concas@cern.ch

#include <cassert>

#include "ITStracking/VertexerTraits.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"

// debug
#ifdef _ALLOW_DEBUG_TREES_ITS_
#include "CommonUtils/TreeStreamRedirector.h"
#endif
// \debug

#define LAYER0_TO_LAYER1 0
#define LAYER1_TO_LAYER2 1

namespace o2
{
namespace its
{

using constants::index_table::PhiBins;
using constants::index_table::ZBins;
using constants::its::LayersRCoordinate;
using constants::its::LayersZCoordinate;
using constants::math::TwoPi;
using index_table_utils::getZBinIndex;

void trackleterKernelSerial(
  const std::vector<Cluster>& clustersNextLayer,    // 0 2
  const std::vector<Cluster>& clustersCurrentLayer, // 1 1
  const std::array<int, ZBins * PhiBins + 1>& indexTableNext,
  const unsigned char pairOfLayers,
  const float phiCut,
  std::vector<Tracklet>& Tracklets,
  std::vector<int>& foundTracklets,
  const ROframe* evt = nullptr,
  const int maxTrackletsPerCluster = static_cast<int>(2e3))
{
  foundTracklets.resize(clustersCurrentLayer.size(), 0);
  // loop on layer1 clusters
  for (unsigned int iCurrentLayerClusterIndex{0}; iCurrentLayerClusterIndex < clustersCurrentLayer.size(); ++iCurrentLayerClusterIndex) {
    int storedTracklets{0};
    const Cluster currentCluster{clustersCurrentLayer[iCurrentLayerClusterIndex]};
    const int layerIndex{pairOfLayers == LAYER0_TO_LAYER1 ? 0 : 2};
    const int4 selectedBinsRect{VertexerTraits::getBinsRect(currentCluster, layerIndex, 0.f, 50.f, phiCut / 2)};
    if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
      int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
      if (phiBinsNum < 0) {
        phiBinsNum += PhiBins;
      }
      // loop on phi bins next layer
      for (int iPhiBin{selectedBinsRect.y}, iPhiCount{0}; iPhiCount < phiBinsNum; iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++) {
        const int firstBinIndex{index_table_utils::getBinIndex(selectedBinsRect.x, iPhiBin)};
        const int firstRowClusterIndex{indexTableNext[firstBinIndex]};
        const int maxRowClusterIndex{indexTableNext[firstBinIndex + ZBins]};
        // loop on clusters next layer
        for (int iNextLayerClusterIndex{firstRowClusterIndex}; iNextLayerClusterIndex < maxRowClusterIndex && iNextLayerClusterIndex < static_cast<int>(clustersNextLayer.size()); ++iNextLayerClusterIndex) {
          const Cluster& nextCluster{clustersNextLayer[iNextLayerClusterIndex]};
          if (gpu::GPUCommonMath::Abs(currentCluster.phiCoordinate - nextCluster.phiCoordinate) < phiCut) {
            if (storedTracklets < maxTrackletsPerCluster) {
              if (pairOfLayers == LAYER0_TO_LAYER1) {
                Tracklets.emplace_back(iNextLayerClusterIndex, iCurrentLayerClusterIndex, nextCluster, currentCluster);
              } else {
                Tracklets.emplace_back(iCurrentLayerClusterIndex, iNextLayerClusterIndex, currentCluster, nextCluster);
              }
              ++storedTracklets;
            }
          }
        }
      }
    }
    foundTracklets[iCurrentLayerClusterIndex] = storedTracklets;
  }
}

void trackletSelectionKernelSerial(
  const std::vector<Cluster>& clustersNextLayer,    //0
  const std::vector<Cluster>& clustersCurrentLayer, //1
  const std::vector<Tracklet>& tracklets01,
  const std::vector<Tracklet>& tracklets12,
  const std::vector<int>& foundTracklets01,
  const std::vector<int>& foundTracklets12,
  std::vector<Line>& destTracklets,
#ifdef _ALLOW_DEBUG_TREES_ITS_
  std::vector<std::array<int, 2>>& allowedTrackletPairs,
#endif
  const float tanLambdaCut = 0.025f,
  const float phiCut = 0.005f,
  const int maxTracklets = static_cast<int>(2e3))
{
  int offset01{0};
  int offset12{0};
  for (unsigned int iCurrentLayerClusterIndex{0}; iCurrentLayerClusterIndex < clustersCurrentLayer.size(); ++iCurrentLayerClusterIndex) {
    int validTracklets{0};
    for (int iTracklet12{offset12}; iTracklet12 < offset12 + foundTracklets12[iCurrentLayerClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{offset01}; iTracklet01 < offset01 + foundTracklets01[iCurrentLayerClusterIndex]; ++iTracklet01) {
        const float deltaTanLambda{gpu::GPUCommonMath::Abs(tracklets01[iTracklet01].tanLambda - tracklets12[iTracklet12].tanLambda)};
        const float deltaPhi{gpu::GPUCommonMath::Abs(tracklets01[iTracklet01].phiCoordinate - tracklets12[iTracklet12].phiCoordinate)};
        if (deltaTanLambda < tanLambdaCut && deltaPhi < phiCut && validTracklets != maxTracklets) {
          assert(tracklets01[iTracklet01].secondClusterIndex == tracklets12[iTracklet12].firstClusterIndex);
          destTracklets.emplace_back(tracklets01[iTracklet01], clustersNextLayer.data(), clustersCurrentLayer.data());
#ifdef _ALLOW_DEBUG_TREES_ITS_
          allowedTrackletPairs.push_back(std::array<int, 2>{iTracklet01, iTracklet12});
#endif
          ++validTracklets;
        }
      }
    }
    offset01 += foundTracklets01[iCurrentLayerClusterIndex];
    offset12 += foundTracklets12[iCurrentLayerClusterIndex];
  }
}
#ifdef _ALLOW_DEBUG_TREES_ITS_
VertexerTraits::VertexerTraits(const std::string dbgFilename) : mAverageClustersRadii{std::array<float, 3>{0.f, 0.f, 0.f}},
                                                                mMaxDirectorCosine3{0.f}
{
  mVrtParams.phiSpan = static_cast<int>(std::ceil(constants::index_table::PhiBins * mVrtParams.phiCut /
                                                  constants::math::TwoPi));
  mVrtParams.zSpan = static_cast<int>(std::ceil(mVrtParams.zCut * constants::index_table::InverseZBinSize()[0]));
  setIsGPU(false);

  std::cout << "[DEBUG] Creating file: " << dbgFilename.data() << std::endl;
  mDebugger = new StandaloneDebugger(dbgFilename.data());
}
#endif

VertexerTraits::VertexerTraits() : mAverageClustersRadii{std::array<float, 3>{0.f, 0.f, 0.f}},
                                   mMaxDirectorCosine3{0.f}
{
  mVrtParams.phiSpan = static_cast<int>(std::ceil(constants::index_table::PhiBins * mVrtParams.phiCut /
                                                  constants::math::TwoPi));
  mVrtParams.zSpan = static_cast<int>(std::ceil(mVrtParams.zCut * constants::index_table::InverseZBinSize()[0]));
}

#ifdef _ALLOW_DEBUG_TREES_ITS_
VertexerTraits::~VertexerTraits()
{
  if (!mIsGPU) {
    delete mDebugger;
  }
}
#endif

void VertexerTraits::reset()
{
  for (int iLayer{0}; iLayer < constants::its::LayersNumberVertexer; ++iLayer) {
    mClusters[iLayer].clear();
    mIndexTables[iLayer].fill(0);
  }

  mTracklets.clear();
  mTrackletClusters.clear();
  mVertices.clear();
  mComb01.clear();
  mComb12.clear();
  mFoundTracklets01.clear();
  mFoundTracklets12.clear();
#ifdef _ALLOW_DEBUG_TREES_ITS_
  mAllowedTrackletPairs.clear();
#endif
  mAverageClustersRadii = {0.f, 0.f, 0.f};
  mMaxDirectorCosine3 = 0.f;
}

std::vector<int> VertexerTraits::getMClabelsLayer(const int layer) const
{
  return mEvent->getTracksId(layer, mClusters[layer]);
}

void VertexerTraits::arrangeClusters(ROframe* event)
{
  mEvent = event;
  for (int iLayer{0}; iLayer < constants::its::LayersNumberVertexer; ++iLayer) {
    const auto& currentLayer{event->getClustersOnLayer(iLayer)};
    const size_t clustersNum{currentLayer.size()};
    if (clustersNum > 0) {
      if (clustersNum > mClusters[iLayer].capacity()) {
        mClusters[iLayer].reserve(clustersNum);
      }
      for (unsigned int iCluster{0}; iCluster < clustersNum; ++iCluster) {
        mClusters[iLayer].emplace_back(iLayer, currentLayer.at(iCluster));
        mAverageClustersRadii[iLayer] += mClusters[iLayer].back().rCoordinate;
      }
      mAverageClustersRadii[iLayer] *= 1.f / clustersNum;

      std::sort(mClusters[iLayer].begin(), mClusters[iLayer].end(), [](Cluster& cluster1, Cluster& cluster2) {
        return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
      });
      int previousBinIndex{0};
      mIndexTables[iLayer][0] = 0;
      for (unsigned int iCluster{0}; iCluster < clustersNum; ++iCluster) {
        const int currentBinIndex{mClusters[iLayer][iCluster].indexTableBinIndex};
        if (currentBinIndex > previousBinIndex) {
          for (int iBin{previousBinIndex + 1}; iBin <= currentBinIndex; ++iBin) {
            mIndexTables[iLayer][iBin] = iCluster;
          }
          previousBinIndex = currentBinIndex;
        }
      }
      for (int iBin{previousBinIndex + 1}; iBin <= ZBins * PhiBins; iBin++) {
        mIndexTables[iLayer][iBin] = static_cast<int>(clustersNum);
      }
    }
  }
  mDeltaRadii10 = mAverageClustersRadii[1] - mAverageClustersRadii[0];
  mDeltaRadii21 = mAverageClustersRadii[2] - mAverageClustersRadii[1];
  mMaxDirectorCosine3 =
    LayersZCoordinate()[2] / std::sqrt(LayersZCoordinate()[2] * LayersZCoordinate()[2] +
                                       (mDeltaRadii10 + mDeltaRadii21) * (mDeltaRadii10 + mDeltaRadii21));
}

const std::vector<std::pair<int, int>> VertexerTraits::selectClusters(const std::array<int, ZBins * PhiBins + 1>& indexTable,
                                                                      const std::array<int, 4>& selectedBinsRect)
{
  std::vector<std::pair<int, int>> filteredBins{};
  int phiBinsNum{selectedBinsRect[3] - selectedBinsRect[1] + 1};
  if (phiBinsNum < 0)
    phiBinsNum += PhiBins;
  filteredBins.reserve(phiBinsNum);
  for (int iPhiBin{selectedBinsRect[1]}, iPhiCount{0}; iPhiCount < phiBinsNum;
       iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++) {
    const int firstBinIndex{index_table_utils::getBinIndex(selectedBinsRect[0], iPhiBin)};
    filteredBins.emplace_back(
      indexTable[firstBinIndex],
      index_table_utils::countRowSelectedBins(indexTable, iPhiBin, selectedBinsRect[0], selectedBinsRect[2]));
  }
  return filteredBins;
}

void VertexerTraits::computeTrackletsPureMontecarlo()
{
  assert(mEvent != nullptr);

  std::vector<int> foundTracklets01;
  std::vector<int> foundTracklets12;

  for (unsigned int iCurrentLayerClusterIndex{0}; iCurrentLayerClusterIndex < mClusters[0].size(); ++iCurrentLayerClusterIndex) {
    auto& currentCluster{mClusters[0][iCurrentLayerClusterIndex]};
    for (unsigned int iNextLayerClusterIndex = 0; iNextLayerClusterIndex < mClusters[1].size(); iNextLayerClusterIndex++) {
      const Cluster& nextCluster{mClusters[1][iNextLayerClusterIndex]};
      const auto& lblNext = mEvent->getClusterLabels(1, nextCluster.clusterId);
      const auto& lblCurr = mEvent->getClusterLabels(0, currentCluster.clusterId);
      if (lblNext.compare(lblCurr) == 1 && lblCurr.getSourceID() == 0) {
        mComb01.emplace_back(iCurrentLayerClusterIndex, iNextLayerClusterIndex, currentCluster, nextCluster);
      }
    }
  }

  for (unsigned int iCurrentLayerClusterIndex{0}; iCurrentLayerClusterIndex < mClusters[2].size(); ++iCurrentLayerClusterIndex) {
    auto& currentCluster{mClusters[2][iCurrentLayerClusterIndex]};
    for (unsigned int iNextLayerClusterIndex = 0; iNextLayerClusterIndex < mClusters[1].size(); iNextLayerClusterIndex++) {
      const Cluster& nextCluster{mClusters[1][iNextLayerClusterIndex]};
      const auto& lblNext = mEvent->getClusterLabels(1, nextCluster.clusterId);
      const auto& lblCurr = mEvent->getClusterLabels(2, currentCluster.clusterId);
      if (lblNext.compare(lblCurr) == 1 && lblCurr.getSourceID() == 0) {
        mComb12.emplace_back(iNextLayerClusterIndex, iCurrentLayerClusterIndex, nextCluster, currentCluster);
      }
    }
  }

  for (auto& trk : mComb01) {
    mTracklets.emplace_back(trk, mClusters[0].data(), mClusters[1].data());
  }

#ifdef _ALLOW_DEBUG_TREES_ITS_
  for (int iTracklet01{0}; iTracklet01 < static_cast<int>(mComb01.size()); ++iTracklet01) {
    auto& trklet01 = mComb01[iTracklet01];
    for (int iTracklet12{0}; iTracklet12 < static_cast<int>(mComb12.size()); ++iTracklet12) {
      auto& trklet12 = mComb12[iTracklet12];
      if (trklet01.secondClusterIndex == trklet12.firstClusterIndex) {
        mAllowedTrackletPairs.push_back(std::array<int, 2>{iTracklet01, iTracklet12});
      }
    }
  }
  if (isDebugFlag(VertexerDebug::TrackletTreeAll)) {
    mDebugger->fillTrackletSelectionTree(mClusters, mComb01, mComb12, mAllowedTrackletPairs, mEvent);
  }
#endif
}

void VertexerTraits::computeTracklets()
{
  trackleterKernelSerial(
    mClusters[0],
    mClusters[1],
    mIndexTables[0],
    LAYER0_TO_LAYER1,
    mVrtParams.phiCut,
    mComb01,
    mFoundTracklets01,
    mEvent);

  trackleterKernelSerial(
    mClusters[2],
    mClusters[1],
    mIndexTables[2],
    LAYER1_TO_LAYER2,
    mVrtParams.phiCut,
    mComb12,
    mFoundTracklets12,
    mEvent);

#ifdef _ALLOW_DEBUG_TREES_ITS_
  if (isDebugFlag(VertexerDebug::CombinatoricsTreeAll)) {
    // fillCombinatoricsTree();
    mDebugger->fillCombinatoricsTree(mComb01, mComb12);
  }
#endif
}

void VertexerTraits::computeTrackletMatching()
{
  trackletSelectionKernelSerial(
    mClusters[0],
    mClusters[1],
    mComb01,
    mComb12,
    mFoundTracklets01,
    mFoundTracklets12,
    mTracklets,
#ifdef _ALLOW_DEBUG_TREES_ITS_
    mAllowedTrackletPairs,
#endif
    mVrtParams.phiCut,
    mVrtParams.tanLambdaCut);
#ifdef _ALLOW_DEBUG_TREES_ITS_
  if (isDebugFlag(VertexerDebug::TrackletTreeAll)) {
    mDebugger->fillTrackletSelectionTree(mClusters, mComb01, mComb12, mAllowedTrackletPairs, mEvent);
  }
  if (isDebugFlag(VertexerDebug::LineTreeAll)) {
    mDebugger->fillPairsInfoTree(mTracklets, mEvent);
  }
  if (isDebugFlag(VertexerDebug::LineSummaryAll)) {
    mDebugger->fillLinesSummaryTree(mTracklets, mEvent);
  }
#endif
}

#ifdef _ALLOW_DEBUG_TREES_ITS_
void VertexerTraits::computeMCFiltering()
{
  assert(mEvent != nullptr);
  for (size_t iTracklet{0}; iTracklet < mComb01.size(); ++iTracklet) {
    const auto& lbl0 = mEvent->getClusterLabels(0, mClusters[0][mComb01[iTracklet].firstClusterIndex].clusterId);
    const auto& lbl1 = mEvent->getClusterLabels(1, mClusters[1][mComb01[iTracklet].secondClusterIndex].clusterId);
    if (!(lbl0.compare(lbl1) == 1 && lbl0.getSourceID() == 0)) { // evtId && trackId && isValid
      mComb01.erase(mComb01.begin() + iTracklet);
      --iTracklet; // vector size has been decreased
    }
  }

  for (size_t iTracklet{0}; iTracklet < mComb12.size(); ++iTracklet) {
    const auto& lbl1 = mEvent->getClusterLabels(1, mClusters[1][mComb12[iTracklet].firstClusterIndex].clusterId);
    const auto& lbl2 = mEvent->getClusterLabels(2, mClusters[2][mComb12[iTracklet].secondClusterIndex].clusterId);
    if (!(lbl1.compare(lbl2) == 1 && lbl1.getSourceID() == 0)) { // evtId && trackId && isValid
      mComb12.erase(mComb12.begin() + iTracklet);
      --iTracklet; // vector size has been decreased
    }
  }
  if (isDebugFlag(VertexerDebug::CombinatoricsTreeAll)) {
    mDebugger->fillCombinatoricsMCTree(mComb01, mComb12);
  }
}
#endif

void VertexerTraits::computeVertices()
{
  const int numTracklets{static_cast<int>(mTracklets.size())};
  std::vector<bool> usedTracklets{};
  usedTracklets.resize(mTracklets.size(), false);
  for (int tracklet1{0}; tracklet1 < numTracklets; ++tracklet1) {
    if (usedTracklets[tracklet1])
      continue;
    for (int tracklet2{tracklet1 + 1}; tracklet2 < numTracklets; ++tracklet2) {
      if (usedTracklets[tracklet2])
        continue;
      if (Line::getDCA(mTracklets[tracklet1], mTracklets[tracklet2]) <= mVrtParams.pairCut) {
        mTrackletClusters.emplace_back(tracklet1, mTracklets[tracklet1], tracklet2, mTracklets[tracklet2]);
        std::array<float, 3> tmpVertex{mTrackletClusters.back().getVertex()};
        if (tmpVertex[0] * tmpVertex[0] + tmpVertex[1] * tmpVertex[1] > 4.f) {
          mTrackletClusters.pop_back();
          break;
        }
        usedTracklets[tracklet1] = true;
        usedTracklets[tracklet2] = true;
        for (int tracklet3{0}; tracklet3 < numTracklets; ++tracklet3) {
          if (usedTracklets[tracklet3])
            continue;
          if (Line::getDistanceFromPoint(mTracklets[tracklet3], tmpVertex) < mVrtParams.pairCut) {
            mTrackletClusters.back().add(tracklet3, mTracklets[tracklet3]);
            usedTracklets[tracklet3] = true;
            tmpVertex = mTrackletClusters.back().getVertex();
          }
        }
        break;
      }
    }
  }

  std::sort(mTrackletClusters.begin(), mTrackletClusters.end(),
            [](ClusterLines& cluster1, ClusterLines& cluster2) { return cluster1.getSize() > cluster2.getSize(); });
  int noClusters{static_cast<int>(mTrackletClusters.size())};
  for (int iCluster1{0}; iCluster1 < noClusters; ++iCluster1) {
    std::array<float, 3> vertex1{mTrackletClusters[iCluster1].getVertex()};
    std::array<float, 3> vertex2{};
    for (int iCluster2{iCluster1 + 1}; iCluster2 < noClusters; ++iCluster2) {
      vertex2 = mTrackletClusters[iCluster2].getVertex();
      if (std::abs(vertex1[2] - vertex2[2]) < mVrtParams.clusterCut) {

        float distance{(vertex1[0] - vertex2[0]) * (vertex1[0] - vertex2[0]) +
                       (vertex1[1] - vertex2[1]) * (vertex1[1] - vertex2[1]) +
                       (vertex1[2] - vertex2[2]) * (vertex1[2] - vertex2[2])};
        if (distance <= mVrtParams.pairCut * mVrtParams.pairCut) {
          for (auto label : mTrackletClusters[iCluster2].getLabels()) {
            mTrackletClusters[iCluster1].add(label, mTracklets[label]);
            vertex1 = mTrackletClusters[iCluster1].getVertex();
          }
        }
        mTrackletClusters.erase(mTrackletClusters.begin() + iCluster2);
        --iCluster2;
        --noClusters;
      }
    }
  }
  for (int iCluster{0}; iCluster < noClusters; ++iCluster) {
    if (mTrackletClusters[iCluster].getSize() < mVrtParams.clusterContributorsCut && noClusters > 1) {
      mTrackletClusters.erase(mTrackletClusters.begin() + iCluster);
      noClusters--;
      continue;
    }
    float dist{0.};
    for (auto& line : mTrackletClusters[iCluster].getLines()) {
      dist += Line::getDistanceFromPoint(line, mTrackletClusters[iCluster].getVertex()) /
              mTrackletClusters[iCluster].getSize();
    }
    if (mTrackletClusters[iCluster].getVertex()[0] * mTrackletClusters[iCluster].getVertex()[0] +
          mTrackletClusters[iCluster].getVertex()[1] * mTrackletClusters[iCluster].getVertex()[1] <
        1.98 * 1.98) {
      mVertices.emplace_back(mTrackletClusters[iCluster].getVertex()[0],
                             mTrackletClusters[iCluster].getVertex()[1],
                             mTrackletClusters[iCluster].getVertex()[2],
                             mTrackletClusters[iCluster].getRMS2(),         // Symm matrix. Diagonal: RMS2 components,
                                                                            // off-diagonal: square mean of projections on planes.
                             mTrackletClusters[iCluster].getSize(),         // Contributors
                             mTrackletClusters[iCluster].getAvgDistance2(), // In place of chi2
                             mEvent->getROFrameId());
      mEvent->addPrimaryVertex(mVertices.back().mX, mVertices.back().mY, mVertices.back().mZ);
    }
  }
}

#ifdef _ALLOW_DEBUG_TREES_ITS_

// Montecarlo validation for externally-provided tracklets (GPU-like cases)
void VertexerTraits::filterTrackletsWithMC(std::vector<Tracklet>& tracklets01,
                                           std::vector<Tracklet>& tracklets12,
                                           std::vector<int>& indices01,
                                           std::vector<int>& indices12,
                                           const int stride)
{
  // Tracklets 01
  assert(mEvent != nullptr);
  for (size_t iFoundTrackletIndex{0}; iFoundTrackletIndex < indices01.size(); ++iFoundTrackletIndex) // access indices vector to get numfoundtracklets
  {
    const size_t offset = iFoundTrackletIndex * stride;
    int removed{0};
    for (size_t iTrackletIndex{0}; iTrackletIndex < indices01[iFoundTrackletIndex]; ++iTrackletIndex) {
      const size_t iTracklet{offset + iTrackletIndex};
      const auto& lbl0 = mEvent->getClusterLabels(0, mClusters[0][tracklets01[iTracklet].firstClusterIndex].clusterId);
      const auto& lbl1 = mEvent->getClusterLabels(1, mClusters[1][tracklets01[iTracklet].secondClusterIndex].clusterId);
      if (!(lbl0.compare(lbl1) == 1 && lbl0.getSourceID() == 0)) {
        tracklets01[iTracklet] = Tracklet();
        ++removed;
      }
    }
    std::sort(tracklets01.begin() + offset, tracklets01.begin() + offset + indices01[iFoundTrackletIndex]);
    indices01[iFoundTrackletIndex] -= removed; // decrease number of tracklets by the number of removed ones
  }

  // Tracklets 12
  for (size_t iFoundTrackletIndex{0}; iFoundTrackletIndex < indices12.size(); ++iFoundTrackletIndex) // access indices vector to get numfoundtracklets
  {
    const size_t offset = iFoundTrackletIndex * stride;
    int removed{0};
    for (size_t iTrackletIndex{0}; iTrackletIndex < indices12[iFoundTrackletIndex]; ++iTrackletIndex) {
      const size_t iTracklet{offset + iTrackletIndex};
      const auto& lbl1 = mEvent->getClusterLabels(1, mClusters[1][tracklets12[iTracklet].firstClusterIndex].clusterId);
      const auto& lbl2 = mEvent->getClusterLabels(2, mClusters[2][tracklets12[iTracklet].secondClusterIndex].clusterId);
      if (!(lbl1.compare(lbl2) == 1 && lbl1.getSourceID() == 0)) {
        tracklets12[iTracklet] = Tracklet();
        ++removed;
      }
    }
    std::sort(tracklets12.begin() + offset, tracklets12.begin() + offset + indices12[iFoundTrackletIndex]);
    indices12[iFoundTrackletIndex] -= removed; // decrease number of tracklets by the number of removed ones
  }
}

#endif

void VertexerTraits::dumpVertexerTraits()
{
  std::cout << "\tDump traits:" << std::endl;
  std::cout << "\tTracklets found: " << mTracklets.size() << std::endl;
  std::cout << "\tClusters of tracklets: " << mTrackletClusters.size() << std::endl;
  std::cout << "\tmVrtParams.pairCut: " << mVrtParams.pairCut << std::endl;
  std::cout << "\tVertices found: " << mVertices.size() << std::endl;
}

VertexerTraits* createVertexerTraits()
{
  return new VertexerTraits;
}

} // namespace its
} // namespace o2
