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
/// \file TrackerTraitsCPU.cxx
/// \brief
///

#include "ITStracking/TrackerTraitsCPU.h"

#include "CommonConstants/MathConstants.h"
#include "ITStracking/Cell.h"
#include "ITStracking/Constants.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/Tracklet.h"
#include <fmt/format.h>
#include "ReconstructionDataFormats/Track.h"
#include <cassert>
#include <iostream>

#include "GPUCommonMath.h"

namespace o2
{
namespace its
{

void TrackerTraitsCPU::computeLayerTracklets()
{
  TimeFrame* tf = mTimeFrame;

  for (int rof0{0}; rof0 < tf->getNrof(); ++rof0) {
    gsl::span<const float3> primaryVertices = tf->getPrimaryVertices(rof0);
    for (int iLayer{0}; iLayer < mTrkParams.TrackletsPerRoad(); ++iLayer) {
      gsl::span<const Cluster> layer0 = tf->getClustersOnLayer(rof0, iLayer);
      if (layer0.empty()) {
        return;
      }

      const int currentLayerClustersNum{static_cast<int>(layer0.size())};

      for (int iCluster{0}; iCluster < currentLayerClustersNum; ++iCluster) {
        const Cluster& currentCluster{layer0[iCluster]};

        if (tf->isClusterUsed(iLayer, currentCluster.clusterId)) {
          continue;
        }

        for (auto& primaryVertex : primaryVertices) {
          const float tanLambda{(currentCluster.zCoordinate - primaryVertex.z) / currentCluster.radius};

          const float zAtRmin{tanLambda * (tf->getMinR(iLayer + 1) - currentCluster.radius) + currentCluster.zCoordinate};
          const float zAtRmax{tanLambda * (tf->getMaxR(iLayer + 1) - currentCluster.radius) + currentCluster.zCoordinate};

          const int4 selectedBinsRect{getBinsRect(currentCluster, iLayer, zAtRmin, zAtRmax,
                                                  mTrkParams.TrackletMaxDeltaZ[iLayer], mTrkParams.TrackletMaxDeltaPhi)};

          if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
            continue;
          }

          int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};

          if (phiBinsNum < 0) {
            phiBinsNum += mTrkParams.PhiBins;
          }

          int minRof = (rof0 > 0) ? rof0 - 1 : 0;
          int maxRof = (rof0 == tf->getNrof() - 1) ? rof0 : rof0 + 1;
          for (int rof1{minRof}; rof1 < maxRof; ++rof1) {
            gsl::span<const Cluster> layer1 = tf->getClustersOnLayer(rof1, iLayer + 1);
            if (layer1.empty()) {
              continue;
            }

            for (int iPhiBin{selectedBinsRect.y}, iPhiCount{0}; iPhiCount < phiBinsNum;
                 iPhiBin = (++iPhiBin == tf->mIndexTableUtils.getNphiBins()) ? 0 : iPhiBin, iPhiCount++) {
              const int firstBinIndex{tf->mIndexTableUtils.getBinIndex(selectedBinsRect.x, iPhiBin)};
              const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
              if (firstBinIndex < 0 || firstBinIndex > tf->getIndexTables(rof1)[iLayer].size() ||
                  maxBinIndex < 0 || maxBinIndex > tf->getIndexTables(rof1)[iLayer].size()) {
                std::cout << iLayer << "\t" << iCluster << "\t" << zAtRmin << "\t" << zAtRmax << "\t" << mTrkParams.TrackletMaxDeltaZ[iLayer] << "\t" << mTrkParams.TrackletMaxDeltaPhi << std::endl;
                std::cout << currentCluster.zCoordinate << "\t" << primaryVertex.z << "\t" << currentCluster.radius << std::endl;
                std::cout << tf->getMinR(iLayer + 1) << "\t" << currentCluster.radius << "\t" << currentCluster.zCoordinate << std::endl;
                std::cout << "Illegal access to IndexTable " << firstBinIndex << "\t" << maxBinIndex << "\t" << selectedBinsRect.z << "\t" << selectedBinsRect.x << std::endl;
                exit(1);
              }
              const int firstRowClusterIndex = tf->getIndexTables(rof1)[iLayer][firstBinIndex];
              const int maxRowClusterIndex = tf->getIndexTables(rof1)[iLayer][maxBinIndex];

              for (int iNextCluster{firstRowClusterIndex}; iNextCluster < maxRowClusterIndex; ++iNextCluster) {
                if (iNextCluster >= (int)tf->getClusters()[iLayer + 1].size()) {
                  break;
                }
                const Cluster& nextCluster{layer1[iNextCluster]};

                if (tf->isClusterUsed(iLayer + 1, nextCluster.clusterId)) {
                  continue;
                }

                const float deltaZ{gpu::GPUCommonMath::Abs(tanLambda * (nextCluster.radius - currentCluster.radius) +
                                                           currentCluster.zCoordinate - nextCluster.zCoordinate)};
                const float deltaPhi{gpu::GPUCommonMath::Abs(currentCluster.phi - nextCluster.phi)};

                if (deltaZ < mTrkParams.TrackletMaxDeltaZ[iLayer] &&
                    (deltaPhi < mTrkParams.TrackletMaxDeltaPhi ||
                     gpu::GPUCommonMath::Abs(deltaPhi - constants::math::TwoPi) < mTrkParams.TrackletMaxDeltaPhi)) {
                  const int currentSortedIndex{tf->getSortedIndex(rof0, iLayer, iCluster)};
                  if (iLayer > 0 && tf->getTrackletsLookupTable()[iLayer - 1].size() <= currentSortedIndex) {
                    tf->getTrackletsLookupTable()[iLayer - 1].resize(currentSortedIndex + 1, tf->getTracklets()[iLayer].size());
                  }

                  tf->getTracklets()[iLayer].emplace_back(currentSortedIndex, tf->getSortedIndex(rof1, iLayer + 1, iNextCluster), currentCluster,
                                                          nextCluster, rof0, rof1);
                  if (tf->hasMCinformation()) {
                    MCCompLabel currentLab{tf->getClusterLabels(iLayer, currentCluster.clusterId)};
                    MCCompLabel nextLab{tf->getClusterLabels(iLayer + 1, nextCluster.clusterId)};
                    tf->getTrackletsLabel(iLayer).emplace_back(currentLab == nextLab ? currentLab : MCCompLabel());
                  }
                }
              }
            }
          }
        }
      }
      if (iLayer > 0) {
        auto currentSize{tf->getTrackletsLookupTable()[iLayer - 1].size()};
        tf->getTrackletsLookupTable()[iLayer - 1].resize(currentSize + currentLayerClustersNum + 1, tf->getTracklets()[iLayer].size());
      }
    }
  }
#ifdef CA_DEBUG
  std::cout << "+++ Number of tracklets per layer: ";
  for (int iLayer{0}; iLayer < mTrkParams.TrackletsPerRoad(); ++iLayer) {
    std::cout << primaryVertexContext->getTracklets()[iLayer].size() << "\t";
  }
#endif
}

void TrackerTraitsCPU::computeLayerCells()
{
  TimeFrame* tf = mTimeFrame;
  for (int iLayer{0}; iLayer < mTrkParams.CellsPerRoad(); ++iLayer) {

    if (tf->getTracklets()[iLayer + 1].empty() ||
        tf->getTracklets()[iLayer].empty()) {
      continue;
    }

    const int currentLayerTrackletsNum{static_cast<int>(tf->getTracklets()[iLayer].size())};

    for (int iTracklet{0}; iTracklet < currentLayerTrackletsNum; ++iTracklet) {

      const Tracklet& currentTracklet{tf->getTracklets()[iLayer][iTracklet]};
      const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
      const int nextLayerFirstTrackletIndex{
        tf->getTrackletsLookupTable()[iLayer][nextLayerClusterIndex]};
      const int nextLayerLastTrackletIndex{
        tf->getTrackletsLookupTable()[iLayer][nextLayerClusterIndex + 1]};

      if (nextLayerFirstTrackletIndex == nextLayerLastTrackletIndex) {
        continue;
      }

      const Cluster& cellClus0{tf->getClusters()[iLayer][currentTracklet.firstClusterIndex]};
      const Cluster& cellClus1{
        tf->getClusters()[iLayer + 1][currentTracklet.secondClusterIndex]};
      const float cellClus0R2{cellClus0.radius * cellClus0.radius};
      const float cellClus1R2{cellClus1.radius * cellClus1.radius};
      const float3 firstDeltaVector{cellClus1.xCoordinate - cellClus0.xCoordinate,
                                    cellClus1.yCoordinate - cellClus0.yCoordinate,
                                    cellClus1R2 - cellClus0R2};

      for (int iNextTracklet{nextLayerFirstTrackletIndex}; iNextTracklet < nextLayerLastTrackletIndex; ++iNextTracklet) {
        if (tf->getTracklets()[iLayer + 1][iNextTracklet].firstClusterIndex != nextLayerClusterIndex) {
          break;
        }
        const Tracklet& nextTracklet{tf->getTracklets()[iLayer + 1][iNextTracklet]};
        const float deltaTanLambda{std::abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};
        const float deltaPhi{std::abs(currentTracklet.phi - nextTracklet.phi)};

        if (deltaTanLambda < mTrkParams.CellMaxDeltaTanLambda &&
            (deltaPhi < mTrkParams.CellMaxDeltaPhi ||
             std::abs(deltaPhi - constants::math::TwoPi) < mTrkParams.CellMaxDeltaPhi)) {

          const float averageTanLambda{0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda)};
          const float directionZIntersection{-averageTanLambda * cellClus0.radius +
                                             cellClus0.zCoordinate};

          unsigned short romin = std::min(std::min(currentTracklet.rof[0], currentTracklet.rof[1]), nextTracklet.rof[1]);
          unsigned short romax = std::max(std::max(currentTracklet.rof[0], currentTracklet.rof[1]), nextTracklet.rof[1]);
          bool deltaZflag{false};
          gsl::span<const float3> primaryVertices{tf->getPrimaryVertices(romin, romax)};
          for (const auto& primaryVertex : primaryVertices)
            deltaZflag = std::abs(directionZIntersection - primaryVertex.z) < mTrkParams.CellMaxDeltaZ[iLayer];

          if (deltaZflag) {

            const Cluster& thirdCellCluster{
              tf->getClusters()[iLayer + 2][nextTracklet.secondClusterIndex]};

            const float thirdCellClusterR2{thirdCellCluster.radius *
                                           thirdCellCluster.radius};

            const float3 secondDeltaVector{thirdCellCluster.xCoordinate - cellClus0.xCoordinate,
                                           thirdCellCluster.yCoordinate - cellClus0.yCoordinate,
                                           thirdCellClusterR2 -
                                             cellClus0R2};

            float3 cellPlaneNormalVector{math_utils::crossProduct(firstDeltaVector, secondDeltaVector)};

            const float vectorNorm{std::hypot(cellPlaneNormalVector.x, cellPlaneNormalVector.y, cellPlaneNormalVector.z)};

            if (vectorNorm < constants::math::FloatMinThreshold ||
                std::abs(cellPlaneNormalVector.z) < constants::math::FloatMinThreshold) {
              continue;
            }

            const float inverseVectorNorm{1.0f / vectorNorm};
            const float3 normVect{cellPlaneNormalVector.x * inverseVectorNorm,
                                  cellPlaneNormalVector.y * inverseVectorNorm,
                                  cellPlaneNormalVector.z * inverseVectorNorm};
            const float planeDistance{-normVect.x * (cellClus1.xCoordinate - tf->getBeamX()) -
                                      (normVect.y * cellClus1.yCoordinate - tf->getBeamY()) -
                                      normVect.z * cellClus1R2};
            const float normVectZsquare{normVect.z * normVect.z};
            const float cellRadius{std::sqrt(
              (1.0f - normVectZsquare - 4.0f * planeDistance * normVect.z) /
              (4.0f * normVectZsquare))};
            const float2 circleCenter{-0.5f * normVect.x / normVect.z,
                                      -0.5f * normVect.y / normVect.z};
            const float dca{std::abs(cellRadius - std::hypot(circleCenter.x, circleCenter.y))};

            if (dca > mTrkParams.CellMaxDCA[iLayer]) {
              continue;
            }

            const float cellTrajectoryCurvature{1.0f / cellRadius};
            if (iLayer > 0 && tf->getCellsLookupTable()[iLayer - 1].size() <= iTracklet) {
              tf->getCellsLookupTable()[iLayer - 1].resize(iTracklet + 1, tf->getCells()[iLayer].size());
            }

            tf->getCells()[iLayer].emplace_back(
              currentTracklet.firstClusterIndex, nextTracklet.firstClusterIndex, nextTracklet.secondClusterIndex,
              iTracklet, iNextTracklet, normVect, cellTrajectoryCurvature);
          }
        }
      }
    }
    if (iLayer > 0) {
      tf->getCellsLookupTable()[iLayer - 1].resize(currentLayerTrackletsNum + 1, currentLayerTrackletsNum);
    }
  }
}

void TrackerTraitsCPU::refitTracks(const std::vector<std::vector<TrackingFrameInfo>>& tf, std::vector<TrackITSExt>& tracks)
{
  std::vector<const Cell*> cells;
  for (int iLayer = 0; iLayer < mTrkParams.CellsPerRoad(); iLayer++) {
    cells.push_back(mTimeFrame->getCells()[iLayer].data());
  }
  std::vector<const Cluster*> clusters;
  for (int iLayer = 0; iLayer < mTrkParams.NLayers; iLayer++) {
    clusters.push_back(mTimeFrame->getClusters()[iLayer].data());
  }
  mChainRunITSTrackFit(*mChain, mTimeFrame->getRoads(), clusters, cells, tf, tracks);
}

} // namespace its
} // namespace o2
