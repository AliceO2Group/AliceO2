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
/// \file VertexerTraits.cxx
/// \brief
/// \author matteo.concas@cern.ch

#include <cassert>
#include <ostream>

#include <boost/histogram.hpp>
#include <boost/format.hpp>

#include "ITStracking/VertexerTraits.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"

#ifdef _ALLOW_DEBUG_TREES_ITS_
#include "ITStracking/StandaloneDebugger.h"
#include <unordered_map>
#endif

#define LAYER0_TO_LAYER1 0
#define LAYER1_TO_LAYER2 1

namespace o2
{
namespace its
{
using boost::histogram::indexed;
using constants::math::TwoPi;

enum class TrackletMode {
  Layer0Layer1,
  Layer1Layer2
};

template <TrackletMode Mode>
void trackleterKernelSerial(
  const gsl::span<const Cluster>& clustersNextLayer,    // 0 2
  const gsl::span<const Cluster>& clustersCurrentLayer, // 1 1
  int* indexTableNext,
  const unsigned char pairOfLayers,
  const float phiCut,
  std::vector<Tracklet>& Tracklets,
  std::vector<int>& foundTracklets,
  const IndexTableUtils& utils,
  const int maxTrackletsPerCluster = static_cast<int>(2e3))
{
  const int PhiBins{utils.getNphiBins()};
  const int ZBins{utils.getNzBins()};

  // loop on layer1 clusters
  for (unsigned int iCurrentLayerClusterIndex{0}; iCurrentLayerClusterIndex < clustersCurrentLayer.size(); ++iCurrentLayerClusterIndex) {
    int storedTracklets{0};
    const Cluster& currentCluster{clustersCurrentLayer[iCurrentLayerClusterIndex]};
    const int layerIndex{pairOfLayers == LAYER0_TO_LAYER1 ? 0 : 2};

    const int4 selectedBinsRect{VertexerTraits::getBinsRect(currentCluster, layerIndex, 0.f, 50.f, phiCut / 2, utils)};

    if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
      int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
      if (phiBinsNum < 0) {
        phiBinsNum += PhiBins;
      }
      // loop on phi bins next layer
      for (int iPhiBin{selectedBinsRect.y}, iPhiCount{0}; iPhiCount < phiBinsNum; iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++) {
        const int firstBinIndex{utils.getBinIndex(selectedBinsRect.x, iPhiBin)};
        const int firstRowClusterIndex{indexTableNext[firstBinIndex]};
        const int maxRowClusterIndex{indexTableNext[firstBinIndex + ZBins]};
        // loop on clusters next layer
        for (int iNextLayerClusterIndex{firstRowClusterIndex}; iNextLayerClusterIndex < maxRowClusterIndex && iNextLayerClusterIndex < static_cast<int>(clustersNextLayer.size()); ++iNextLayerClusterIndex) {
          const Cluster& nextCluster{clustersNextLayer[iNextLayerClusterIndex]};
          if (o2::gpu::GPUCommonMath::Abs(currentCluster.phi - nextCluster.phi) < phiCut) {
            if (storedTracklets < maxTrackletsPerCluster) {
              if constexpr (Mode == TrackletMode::Layer0Layer1) {
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
  const gsl::span<const Cluster> clustersNextLayer,    //0
  const gsl::span<const Cluster> clustersCurrentLayer, //1
  const gsl::span<const Tracklet>& tracklets01,
  const gsl::span<const Tracklet>& tracklets12,
  const std::vector<int>& foundTracklets01,
  const std::vector<int>& foundTracklets12,
  std::vector<Line>& destTracklets,
#ifdef _ALLOW_DEBUG_TREES_ITS_
  std::vector<std::array<int, 2>>& allowedTrackletPairs,
  StandaloneDebugger* debugger,
  ROframe* event,
#endif
  const float tanLambdaCut = 0.025f,
  const float phiCut = 0.005f,
  const int maxTracklets = static_cast<int>(1e2))
{
  int offset01{0};
  int offset12{0};
  for (unsigned int iCurrentLayerClusterIndex{0}; iCurrentLayerClusterIndex < clustersCurrentLayer.size(); ++iCurrentLayerClusterIndex) {
    int validTracklets{0};
    for (int iTracklet12{offset12}; iTracklet12 < offset12 + foundTracklets12[iCurrentLayerClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{offset01}; iTracklet01 < offset01 + foundTracklets01[iCurrentLayerClusterIndex]; ++iTracklet01) {
        const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(tracklets01[iTracklet01].tanLambda - tracklets12[iTracklet12].tanLambda)};
        const float deltaPhi{o2::gpu::GPUCommonMath::Abs(tracklets01[iTracklet01].phi - tracklets12[iTracklet12].phi)};
        if (deltaTanLambda < tanLambdaCut && deltaPhi < phiCut && validTracklets != maxTracklets) {
          if (tracklets01[iTracklet01].secondClusterIndex != tracklets12[iTracklet12].firstClusterIndex) {
            LOG(fatal) << "Second clIdx in tracklet 01: " << tracklets01[iTracklet01].secondClusterIndex << "first clIdx in tracklet 12: " << tracklets12[iTracklet12].firstClusterIndex;
          }
#ifdef _ALLOW_DEBUG_TREES_ITS_
          allowedTrackletPairs.push_back(std::array<int, 2>{iTracklet01, iTracklet12});
          destTracklets.emplace_back(tracklets01[iTracklet01], clustersNextLayer.data(), clustersCurrentLayer.data(), debugger->getEventId(clustersNextLayer[tracklets01[iTracklet01].firstClusterIndex].clusterId, clustersCurrentLayer[tracklets01[iTracklet01].secondClusterIndex].clusterId, event));
#else
          destTracklets.emplace_back(tracklets01[iTracklet01], clustersNextLayer.data(), clustersCurrentLayer.data());
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
VertexerTraits::VertexerTraits()
{
  mVrtParams.phiSpan = static_cast<int>(std::ceil(constants::its2::PhiBins * mVrtParams.phiCut /
                                                  constants::math::TwoPi));
  mVrtParams.zSpan = static_cast<int>(std::ceil(mVrtParams.zCut * constants::its2::InverseZBinSize()[0]));
  setIsGPU(false);

  LOG(info) << "[DEBUG] Creating file: dbg_ITSVertexerCPU.root" << std::endl;
  mDebugger = new StandaloneDebugger("dbg_ITSVertexerCPU.root");
}
#else
VertexerTraits::VertexerTraits()
{
}
#endif

#ifdef _ALLOW_DEBUG_TREES_ITS_
VertexerTraits::~VertexerTraits()
{
  if (!mIsGPU) {
    delete mDebugger;
  }
}
#endif

std::vector<int> VertexerTraits::getMClabelsLayer(const int layer) const
{
  return mEvent->getTracksId(layer, mClusters[layer]);
}

const std::vector<std::pair<int, int>> VertexerTraits::selectClusters(const int* indexTable,
                                                                      const std::array<int, 4>& selectedBinsRect,
                                                                      const IndexTableUtils& utils)
{
  std::vector<std::pair<int, int>> filteredBins{};
  int phiBinsNum{selectedBinsRect[3] - selectedBinsRect[1] + 1};
  if (phiBinsNum < 0) {
    phiBinsNum += utils.getNphiBins();
  }
  filteredBins.reserve(phiBinsNum);
  for (int iPhiBin{selectedBinsRect[1]}, iPhiCount{0}; iPhiCount < phiBinsNum;
       iPhiBin = ++iPhiBin == utils.getNphiBins() ? 0 : iPhiBin, iPhiCount++) {
    const int firstBinIndex{utils.getBinIndex(selectedBinsRect[0], iPhiBin)};
    filteredBins.emplace_back(
      indexTable[firstBinIndex],
      utils.countRowSelectedBins(indexTable, iPhiBin, selectedBinsRect[0], selectedBinsRect[2]));
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
      const auto& lblNext = *(mEvent->getClusterLabels(1, nextCluster.clusterId).begin());
      const auto& lblCurr = *(mEvent->getClusterLabels(0, currentCluster.clusterId).begin());
      if (lblNext.compare(lblCurr) == 1 && lblCurr.getSourceID() == 0) {
        mComb01.emplace_back(iCurrentLayerClusterIndex, iNextLayerClusterIndex, currentCluster, nextCluster);
      }
    }
  }

  for (unsigned int iCurrentLayerClusterIndex{0}; iCurrentLayerClusterIndex < mClusters[2].size(); ++iCurrentLayerClusterIndex) {
    auto& currentCluster{mClusters[2][iCurrentLayerClusterIndex]};
    for (unsigned int iNextLayerClusterIndex = 0; iNextLayerClusterIndex < mClusters[1].size(); iNextLayerClusterIndex++) {
      const Cluster& nextCluster{mClusters[1][iNextLayerClusterIndex]};
      const auto& lblNext = *(mEvent->getClusterLabels(1, nextCluster.clusterId).begin());
      const auto& lblCurr = *(mEvent->getClusterLabels(2, currentCluster.clusterId).begin());
      if (lblNext.compare(lblCurr) == 1 && lblCurr.getSourceID() == 0) {
        mComb12.emplace_back(iNextLayerClusterIndex, iCurrentLayerClusterIndex, nextCluster, currentCluster);
      }
    }
  }

  for (auto& trk : mComb01) {
    mTracklets.emplace_back(trk, mClusters[0].data(), mClusters[1].data()); // any check on the propagation to the third layer?
  }

#ifdef _ALLOW_DEBUG_TREES_ITS_
  for (int iTracklet01{0}; iTracklet01 < static_cast<int>(mComb01.size()); ++iTracklet01) {
    auto& trklet01 = mComb01[iTracklet01];
    for (int iTracklet12{0}; iTracklet12 < static_cast<int>(mComb12.size()); ++iTracklet12) {
      auto& trklet12 = mComb12[iTracklet12];
      if (trklet01.secondClusterIndex == trklet12.firstClusterIndex) {
        mAllowedTrackletPairs.push_back(std::array<int, 2>{iTracklet01, iTracklet12});
        break;
      }
    }
  }
  if (isDebugFlag(VertexerDebug::CombinatoricsTreeAll)) {
    mDebugger->fillCombinatoricsTree(mClusters, mComb01, mComb12, mEvent);
  }
  if (isDebugFlag(VertexerDebug::TrackletTreeAll)) {
    mDebugger->fillTrackletSelectionTree(mClusters, mComb01, mComb12, mAllowedTrackletPairs, mEvent);
  }
#endif
}

void VertexerTraits::computeTracklets()
{
  for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
    mTimeFrame->getNTrackletsCluster(rofId)[0].resize(mTimeFrame->getClustersOnLayer(rofId, 1).size());
    mTimeFrame->getNTrackletsCluster(rofId)[1].resize(mTimeFrame->getClustersOnLayer(rofId, 1).size());

    trackleterKernelSerial<TrackletMode::Layer0Layer1>(
      mTimeFrame->getClustersOnLayer(rofId, 0),  // mClusters[0]
      mTimeFrame->getClustersOnLayer(rofId, 1),  // mClusters[1]
      mTimeFrame->getIndexTableL0(rofId).data(), // atm dedicated table for L0
      LAYER0_TO_LAYER1,
      mVrtParams.phiCut,
      mTimeFrame->getTracklets()[0],
      mTimeFrame->getNTrackletsCluster(rofId)[0],
      mIndexTableUtils);

    trackleterKernelSerial<TrackletMode::Layer1Layer2>(
      mTimeFrame->getClustersOnLayer(rofId, 2),    // mClusters[2]
      mTimeFrame->getClustersOnLayer(rofId, 1),    // mClusters[1]
      mTimeFrame->getIndexTables(rofId)[1].data(), // index tables start from layer 1
      LAYER1_TO_LAYER2,
      mVrtParams.phiCut,
      mTimeFrame->getTracklets()[1],
      mTimeFrame->getNTrackletsCluster(rofId)[1],
      mIndexTableUtils);

    mTimeFrame->getNTrackletsROf(0, rofId) = std::accumulate(mTimeFrame->getNTrackletsCluster(rofId)[0].begin(), mTimeFrame->getNTrackletsCluster(rofId)[0].end(), 0);
    mTimeFrame->getNTrackletsROf(1, rofId) = std::accumulate(mTimeFrame->getNTrackletsCluster(rofId)[1].begin(), mTimeFrame->getNTrackletsCluster(rofId)[1].end(), 0);
  }
  mTimeFrame->computeTrackletsScans();
}

void VertexerTraits::computeTrackletMatching()
{
  TimeFrame* tf = mTimeFrame;
  for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
    trackletSelectionKernelSerial(
      mTimeFrame->getClustersOnLayer(rofId, 0), // mClusters[0],
      mTimeFrame->getClustersOnLayer(rofId, 1), // mClusters[1],
      mTimeFrame->getFoundTracklets(rofId, 0),
      mTimeFrame->getFoundTracklets(rofId, 1),
      mTimeFrame->getNTrackletsCluster(rofId)[0], // mFoundTracklets01,
      mTimeFrame->getNTrackletsCluster(rofId)[1], // mFoundTracklets12,
      mTimeFrame->getLines(rofId),
#ifdef _ALLOW_DEBUG_TREES_ITS_
      mAllowedTrackletPairs,
      mDebugger,
      mEvent,
#endif
      mVrtParams.tanLambdaCut,
      mVrtParams.phiCut);
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
}

void VertexerTraits::computeVertices()
{
  TimeFrame* tf = mTimeFrame;
  for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
    const int numTracklets{static_cast<int>(mTimeFrame->getLines(rofId).size())};
    std::vector<bool> usedTracklets{};
    usedTracklets.resize(mTimeFrame->getLines(rofId).size(), false);
    for (int tracklet1{0}; tracklet1 < numTracklets; ++tracklet1) {
      if (usedTracklets[tracklet1]) {
        continue;
      }
      for (int tracklet2{tracklet1 + 1}; tracklet2 < numTracklets; ++tracklet2) {
        if (usedTracklets[tracklet2]) {
          continue;
        }
        if (Line::getDCA(mTimeFrame->getLines(rofId)[tracklet1], mTimeFrame->getLines(rofId)[tracklet2]) <= mVrtParams.pairCut) {
          mTimeFrame->getTrackletClusters(rofId).emplace_back(tracklet1, mTimeFrame->getLines(rofId)[tracklet1], tracklet2, mTimeFrame->getLines(rofId)[tracklet2]);
          std::array<float, 3> tmpVertex{mTimeFrame->getTrackletClusters(rofId).back().getVertex()};
          if (tmpVertex[0] * tmpVertex[0] + tmpVertex[1] * tmpVertex[1] > 4.f) {
            mTimeFrame->getTrackletClusters(rofId).pop_back();
            break;
          }
          usedTracklets[tracklet1] = true;
          usedTracklets[tracklet2] = true;
          for (int tracklet3{0}; tracklet3 < numTracklets; ++tracklet3) {
            if (usedTracklets[tracklet3]) {
              continue;
            }
            if (Line::getDistanceFromPoint(mTimeFrame->getLines(rofId)[tracklet3], tmpVertex) < mVrtParams.pairCut) {
              mTimeFrame->getTrackletClusters(rofId).back().add(tracklet3, mTimeFrame->getLines(rofId)[tracklet3]);
              usedTracklets[tracklet3] = true;
              tmpVertex = mTimeFrame->getTrackletClusters(rofId).back().getVertex();
            }
          }
          break;
        }
      }
    }
#ifdef _ALLOW_DEBUG_TREES_ITS_
    if (isDebugFlag(VertexerDebug::LineSummaryAll)) {
      mDebugger->fillLineClustersTree(mTimeFrame->getTrackletClusters(rofId), mEvent);
    }
#endif
    std::sort(mTimeFrame->getTrackletClusters(rofId).begin(), mTimeFrame->getTrackletClusters(rofId).end(),
              [](ClusterLines& cluster1, ClusterLines& cluster2) { return cluster1.getSize() > cluster2.getSize(); });
    int noClusters{static_cast<int>(mTimeFrame->getTrackletClusters(rofId).size())};
    for (int iCluster1{0}; iCluster1 < noClusters; ++iCluster1) {
      std::array<float, 3> vertex1{mTimeFrame->getTrackletClusters(rofId)[iCluster1].getVertex()};
      std::array<float, 3> vertex2{};
      for (int iCluster2{iCluster1 + 1}; iCluster2 < noClusters; ++iCluster2) {
        vertex2 = mTimeFrame->getTrackletClusters(rofId)[iCluster2].getVertex();
        if (std::abs(vertex1[2] - vertex2[2]) < mVrtParams.clusterCut) {
          float distance{(vertex1[0] - vertex2[0]) * (vertex1[0] - vertex2[0]) +
                         (vertex1[1] - vertex2[1]) * (vertex1[1] - vertex2[1]) +
                         (vertex1[2] - vertex2[2]) * (vertex1[2] - vertex2[2])};
          if (distance <= mVrtParams.pairCut * mVrtParams.pairCut) {
            for (auto label : mTimeFrame->getTrackletClusters(rofId)[iCluster2].getLabels()) {
              mTimeFrame->getTrackletClusters(rofId)[iCluster1].add(label, mTimeFrame->getLines(rofId)[label]);
              vertex1 = mTimeFrame->getTrackletClusters(rofId)[iCluster1].getVertex();
            }
          }
          mTimeFrame->getTrackletClusters(rofId).erase(mTimeFrame->getTrackletClusters(rofId).begin() + iCluster2);
          --iCluster2;
          --noClusters;
        }
      }
    }
    for (int iCluster{0}; iCluster < noClusters; ++iCluster) {
      if (mTimeFrame->getTrackletClusters(rofId)[iCluster].getSize() < mVrtParams.clusterContributorsCut && noClusters > 1) {
        mTimeFrame->getTrackletClusters(rofId).erase(mTimeFrame->getTrackletClusters(rofId).begin() + iCluster);
        noClusters--;
        continue;
      }
      if (mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[0] * mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[0] +
            mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[1] * mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[1] <
          1.98 * 1.98) {
        mVertices.emplace_back(mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[0],
                               mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[1],
                               mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[2],
                               mTimeFrame->getTrackletClusters(rofId)[iCluster].getRMS2(),         // Symm matrix. Diagonal: RMS2 components,
                                                                                                   // off-diagonal: square mean of projections on planes.
                               mTimeFrame->getTrackletClusters(rofId)[iCluster].getSize(),         // Contributors
                               mTimeFrame->getTrackletClusters(rofId)[iCluster].getAvgDistance2(), // In place of chi2
                               rofId
#ifdef _ALLOW_DEBUG_TREES_ITS_
                               ,
                               mTimeFrame->getTrackletClusters(rofId)[iCluster].getEventId(),
                               mTimeFrame->getTrackletClusters(rofId)[iCluster].getPurity()
#endif
        );
      }
    }
#ifdef _ALLOW_DEBUG_TREES_ITS_
    for (auto& vertex : mVertices) {
      mDebugger->fillVerticesInfoTree(vertex.mX, vertex.mY, vertex.mZ, vertex.mContributors, mEvent->getROFrameId(), vertex.mEventId, vertex.mPurity);
    }
#endif
    std::vector<Vertex> vertices;
    for (auto& vertex : mVertices) {
      vertices.emplace_back(o2::math_utils::Point3D<float>(vertex.mX, vertex.mY, vertex.mZ), vertex.mRMS2, vertex.mContributors, vertex.mAvgDistance2);
      vertices.back().setTimeStamp(vertex.mTimeStamp);
    }
    mTimeFrame->addPrimaryVertices(vertices);
    mVertices.clear();
  }
}

void VertexerTraits::computeHistVertices()
{
  o2::its::VertexerHistogramsConfiguration histConf;
#ifdef _ALLOW_DEBUG_TREES_ITS_
  std::array<std::vector<int>, 4001> indicesBookKeeper; // HARD_TO_CODE
  for (auto v : indicesBookKeeper) {
    v.clear();
  }
  const float binSize{histConf.highHistBoundariesXYZ[2] - histConf.lowHistBoundariesXYZ[2]};
#endif
  std::vector<boost::histogram::axis::regular<float>> axes;
  axes.reserve(3);
  for (size_t iAxis{0}; iAxis < 3; ++iAxis) {
    axes.emplace_back(histConf.nBinsXYZ[iAxis] - 1, histConf.lowHistBoundariesXYZ[iAxis], histConf.highHistBoundariesXYZ[iAxis]);
  }

  auto histX = boost::histogram::make_histogram(axes[0]);
  auto histY = boost::histogram::make_histogram(axes[1]);
  auto histZ = boost::histogram::make_histogram(axes[2]);

  // Loop over lines, calculate transverse vertices within beampipe and fill XY histogram to find pseudobeam projection
  for (size_t iTracklet1{0}; iTracklet1 < mTracklets.size(); ++iTracklet1) {
    for (size_t iTracklet2{iTracklet1 + 1}; iTracklet2 < mTracklets.size(); ++iTracklet2) {
      if (Line::getDCA(mTracklets[iTracklet1], mTracklets[iTracklet2]) < mVrtParams.histPairCut) {
        ClusterLines cluster{mTracklets[iTracklet1], mTracklets[iTracklet2]};
        if (cluster.getVertex()[0] * cluster.getVertex()[0] + cluster.getVertex()[1] * cluster.getVertex()[1] < 1.98f * 1.98f) {
          histX(cluster.getVertex()[0]);
          histY(cluster.getVertex()[1]);
        }
      }
    }
  }

  // Try again to use std::max_element as soon as boost is upgraded to 1.71...
  // atm you can iterate over histograms, not really possible to get bin index. Need to use iterate(histogram)
  int maxXBinContent{0};
  int maxYBinContent{0};
  int maxXIndex{0};
  int maxYIndex{0};
  for (auto x : indexed(histX)) {
    if (x.get() > maxXBinContent) {
      maxXBinContent = x.get();
      maxXIndex = x.index();
    }
  }

  for (auto y : indexed(histY)) {
    if (y.get() > maxYBinContent) {
      maxYBinContent = y.get();
      maxYIndex = y.index();
    }
  }

  // Compute weighted average around XY to smooth the position
  if (maxXBinContent || maxYBinContent) {
    float tmpX{histConf.lowHistBoundariesXYZ[0] + histConf.binSizeHistX * maxXIndex + histConf.binSizeHistX / 2};
    float tmpY{histConf.lowHistBoundariesXYZ[1] + histConf.binSizeHistY * maxYIndex + histConf.binSizeHistY / 2};
    int sumX{maxXBinContent};
    int sumY{maxYBinContent};
    float wX{tmpX * static_cast<float>(maxXBinContent)};
    float wY{tmpY * static_cast<float>(maxYBinContent)};
    for (int iBinX{std::max(0, maxXIndex - histConf.binSpanXYZ[0])}; iBinX < std::min(maxXIndex + histConf.binSpanXYZ[0] + 1, histConf.nBinsXYZ[0] - 1); ++iBinX) {
      if (iBinX != maxXIndex) {
        wX += (histConf.lowHistBoundariesXYZ[0] + histConf.binSizeHistX * iBinX + histConf.binSizeHistX / 2) * histX.at(iBinX);
        sumX += histX.at(iBinX);
      }
    }
    for (int iBinY{std::max(0, maxYIndex - histConf.binSpanXYZ[1])}; iBinY < std::min(maxYIndex + histConf.binSpanXYZ[1] + 1, histConf.nBinsXYZ[1] - 1); ++iBinY) {
      if (iBinY != maxYIndex) {
        wY += (histConf.lowHistBoundariesXYZ[1] + histConf.binSizeHistY * iBinY + histConf.binSizeHistY / 2) * histY.at(iBinY);
        sumY += histY.at(iBinY);
      }
    }

    const float beamCoordinateX{wX / sumX};
    const float beamCoordinateY{wY / sumY};

    // create actual pseudobeam line
    Line pseudoBeam{std::array<float, 3>{beamCoordinateX, beamCoordinateY, 1}, std::array<float, 3>{beamCoordinateX, beamCoordinateY, -1}};

    // Fill z coordinate histogram
    for (auto& line : mTracklets) {
      if (Line::getDCA(line, pseudoBeam) < mVrtParams.histPairCut) {
        ClusterLines cluster{line, pseudoBeam};
#ifdef _ALLOW_DEBUG_TREES_ITS_
        // here needs to store ID and bin
        int b_index = StandaloneDebugger::getBinIndex(cluster.getVertex()[2],
                                                      histConf.nBinsXYZ[2] - 1,
                                                      histConf.lowHistBoundariesXYZ[2],
                                                      histConf.highHistBoundariesXYZ[2]);
        const int evId = cluster.getEventId();
        if (!(evId < 0)) {
          indicesBookKeeper[b_index].push_back(evId);
        }
#endif
        histZ(cluster.getVertex()[2]);
      }
    }
    for (int iVertex{0};; ++iVertex) {
#ifdef _ALLOW_DEBUG_TREES_ITS_
      int mVotes{0};
      int mNVotes{0};
      int mPoll{-1};
      int mTotVotes{0};
      int mSwitches{0};
      std::unordered_map<int, int> mMap;
#endif
      int maxZBinContent{0};
      int maxZIndex{0};
      // find maximum
      for (auto z : indexed(histZ)) {
        if (z.get() > maxZBinContent) {
          maxZBinContent = z.get();
          maxZIndex = z.index();
        }
      }
      float tmpZ{histConf.lowHistBoundariesXYZ[2] + histConf.binSizeHistZ * maxZIndex + histConf.binSizeHistZ / 2};
      int sumZ{maxZBinContent};
      float wZ{tmpZ * static_cast<float>(maxZBinContent)};
      for (int iBinZ{std::max(0, maxZIndex - histConf.binSpanXYZ[2])}; iBinZ < std::min(maxZIndex + histConf.binSpanXYZ[2] + 1, histConf.nBinsXYZ[2] - 1); ++iBinZ) {
        if (iBinZ != maxZIndex) {
          wZ += (histConf.lowHistBoundariesXYZ[2] + histConf.binSizeHistZ * iBinZ + histConf.binSizeHistZ / 2) * histZ.at(iBinZ);
          sumZ += histZ.at(iBinZ);
          histZ.at(iBinZ) = 0;
#ifdef _ALLOW_DEBUG_TREES_ITS_
          indicesBookKeeper[iBinZ].clear();
#endif
        }
      }
      if ((sumZ < mVrtParams.clusterContributorsCut) && (iVertex != 0)) {
        break;
      }
#ifdef _ALLOW_DEBUG_TREES_ITS_
      for (int iBinZ{std::max(0, maxZIndex - histConf.binSpanXYZ[2])};
           iBinZ < std::min(maxZIndex + histConf.binSpanXYZ[2] + 1, histConf.nBinsXYZ[2] - 1);
           ++iBinZ) {
        // Boyer-Moore-ish
        for (auto& evId : indicesBookKeeper[iBinZ]) {
          if (mNVotes == 0) {
            mPoll = evId;
            ++mNVotes;
          } else {
            if (evId == mPoll) {
              ++mNVotes;
            } else {
              ++mSwitches;
              --mNVotes;
            }
          }
          auto it = mMap.find(evId);
          if (it == mMap.end()) {
            mMap.emplace(evId, 1);
          } else {
            it->second += 1;
          }
          ++mTotVotes;
        }
        // std::cout<<"\n";
      }
      // Compute purity
      auto it = mMap.find(mPoll);
      assert(it != mMap.end());
      float purity{(float)it->second / (float)mTotVotes};
#endif
      histZ.at(maxZIndex) = 0;
#ifdef _ALLOW_DEBUG_TREES_ITS_
      indicesBookKeeper[maxZIndex].clear();
#endif
      const float vertexZCoordinate{wZ / sumZ};
      mVertices.emplace_back(beamCoordinateX,
                             beamCoordinateY,
                             vertexZCoordinate,
                             std::array<float, 6>{0., 0., 0., 0., 0., 0.},
                             sumZ,
                             0.,
                             mEvent->getROFrameId()
#ifdef _ALLOW_DEBUG_TREES_ITS_
                               ,
                             mPoll,
                             purity
#endif
      );
    }
  }
#ifdef _ALLOW_DEBUG_TREES_ITS_
  for (auto& vertex : mVertices) {
    mDebugger->fillVerticesInfoTree(vertex.mX, vertex.mY, vertex.mZ, vertex.mContributors, mEvent->getROFrameId(), vertex.mEventId, vertex.mPurity);
  }
#endif
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
      const auto& lbl0 = *(mEvent->getClusterLabels(0, mClusters[0][tracklets01[iTracklet].firstClusterIndex].clusterId).begin());
      const auto& lbl1 = *(mEvent->getClusterLabels(1, mClusters[1][tracklets01[iTracklet].secondClusterIndex].clusterId).begin());
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
      const auto& lbl1 = *(mEvent->getClusterLabels(1, mClusters[1][tracklets12[iTracklet].firstClusterIndex].clusterId).begin());
      const auto& lbl2 = *(mEvent->getClusterLabels(2, mClusters[2][tracklets12[iTracklet].secondClusterIndex].clusterId).begin());
      if (!(lbl1.compare(lbl2) == 1 && lbl1.getSourceID() == 0)) {
        tracklets12[iTracklet] = Tracklet();
        ++removed;
      }
    }
    std::sort(tracklets12.begin() + offset, tracklets12.begin() + offset + indices12[iFoundTrackletIndex]);
    indices12[iFoundTrackletIndex] -= removed; // decrease number of tracklets by the number of removed ones
  }
}

void VertexerTraits::computeMCFiltering()
{
  assert(mEvent != nullptr);
  for (size_t iTracklet{0}; iTracklet < mComb01.size(); ++iTracklet) {
    const auto& lbl0 = *(mEvent->getClusterLabels(0, mClusters[0][mComb01[iTracklet].firstClusterIndex].clusterId).begin());
    const auto& lbl1 = *(mEvent->getClusterLabels(1, mClusters[1][mComb01[iTracklet].secondClusterIndex].clusterId).begin());
    if (!(lbl0.compare(lbl1) == 1 && lbl0.getSourceID() == 0)) { // evtId && trackId && isValid
      mComb01.erase(mComb01.begin() + iTracklet);
      --iTracklet; // vector size has been decreased
    }
  }

  for (size_t iTracklet{0}; iTracklet < mComb12.size(); ++iTracklet) {
    const auto& lbl1 = *(mEvent->getClusterLabels(1, mClusters[1][mComb12[iTracklet].firstClusterIndex].clusterId).begin());
    const auto& lbl2 = *(mEvent->getClusterLabels(2, mClusters[2][mComb12[iTracklet].secondClusterIndex].clusterId).begin());
    if (!(lbl1.compare(lbl2) == 1 && lbl1.getSourceID() == 0)) { // evtId && trackId && isValid
      mComb12.erase(mComb12.begin() + iTracklet);
      --iTracklet; // vector size has been decreased
    }
  }
  if (isDebugFlag(VertexerDebug::CombinatoricsTreeAll)) {
    mDebugger->fillCombinatoricsTree(mClusters, mComb01, mComb12, mEvent);
  }
}
#endif

void VertexerTraits::dumpVertexerTraits()
{
  TimeFrame* tf = mTimeFrame;
  for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
    std::cout << "\tDump traits:" << std::endl;
    // std::cout << "\tTracklets found: " << mTracklets.size() << std::endl;
    // std::cout << "\tClusters of tracklets: " << mTrackletClusters.size() << std::endl;
    // std::cout << "\tmVrtParams.pairCut: " << mVrtParams.pairCut << std::endl;
    // std::cout << "\tmVrtParams.phiCut: " << mVrtParams.phiCut << std::endl;
    std::cout << "\tVertices found: " << mVertices.size() << std::endl;
  }
}

VertexerTraits* createVertexerTraits()
{
  return new VertexerTraits;
}

} // namespace its
} // namespace o2
