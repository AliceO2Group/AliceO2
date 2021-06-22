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
  PrimaryVertexContext* primaryVertexContext = mPrimaryVertexContext;
  for (int iLayer{0}; iLayer < mTrkParams.TrackletsPerRoad(); ++iLayer) {
    if (primaryVertexContext->getClusters()[iLayer].empty() || primaryVertexContext->getClusters()[iLayer + 1].empty()) {
      continue;
    }

    const float3& primaryVertex = primaryVertexContext->getPrimaryVertex();
    const int currentLayerClustersNum{static_cast<int>(primaryVertexContext->getClusters()[iLayer].size())};

    for (int iCluster{0}; iCluster < currentLayerClustersNum; ++iCluster) {
      const Cluster& currentCluster{primaryVertexContext->getClusters()[iLayer][iCluster]};

      if (primaryVertexContext->isClusterUsed(iLayer, currentCluster.clusterId)) {
        continue;
      }

      const float tanLambda{(currentCluster.zCoordinate - primaryVertex.z) / currentCluster.rCoordinate};
      const float zAtRmin{tanLambda * (mPrimaryVertexContext->getMinR(iLayer + 1) -
                                       currentCluster.rCoordinate) +
                          currentCluster.zCoordinate};
      const float zAtRmax{tanLambda * (mPrimaryVertexContext->getMaxR(iLayer + 1) -
                                       currentCluster.rCoordinate) +
                          currentCluster.zCoordinate};

      const int4 selectedBinsRect{getBinsRect(currentCluster, iLayer, zAtRmin, zAtRmax,
                                              mTrkParams.TrackletMaxDeltaZ[iLayer], mTrkParams.TrackletMaxDeltaPhi)};

      if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
        continue;
      }

      int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};

      if (phiBinsNum < 0) {
        phiBinsNum += mTrkParams.PhiBins;
      }

      for (int iPhiBin{selectedBinsRect.y}, iPhiCount{0}; iPhiCount < phiBinsNum;
           iPhiBin = ++iPhiBin == mTrkParams.PhiBins ? 0 : iPhiBin, iPhiCount++) {
        const int firstBinIndex{primaryVertexContext->mIndexTableUtils.getBinIndex(selectedBinsRect.x, iPhiBin)};
        const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
        const int firstRowClusterIndex = primaryVertexContext->getIndexTables()[iLayer][firstBinIndex];
        const int maxRowClusterIndex = primaryVertexContext->getIndexTables()[iLayer][maxBinIndex];

        for (int iNextLayerCluster{firstRowClusterIndex}; iNextLayerCluster < maxRowClusterIndex;
             ++iNextLayerCluster) {

          if (iNextLayerCluster >= (int)primaryVertexContext->getClusters()[iLayer + 1].size()) {
            break;
          }

          const Cluster& nextCluster{primaryVertexContext->getClusters()[iLayer + 1][iNextLayerCluster]};

          if (primaryVertexContext->isClusterUsed(iLayer + 1, nextCluster.clusterId)) {
            continue;
          }

          const float deltaZ{o2::gpu::GPUCommonMath::Abs(tanLambda * (nextCluster.rCoordinate - currentCluster.rCoordinate) +
                                                         currentCluster.zCoordinate - nextCluster.zCoordinate)};
          const float deltaPhi{o2::gpu::GPUCommonMath::Abs(currentCluster.phiCoordinate - nextCluster.phiCoordinate)};

          if (deltaZ < mTrkParams.TrackletMaxDeltaZ[iLayer] &&
              (deltaPhi < mTrkParams.TrackletMaxDeltaPhi ||
               o2::gpu::GPUCommonMath::Abs(deltaPhi - constants::math::TwoPi) < mTrkParams.TrackletMaxDeltaPhi)) {

            if (iLayer > 0 &&
                primaryVertexContext->getTrackletsLookupTable()[iLayer - 1][iCluster] == constants::its::UnusedIndex) {

              primaryVertexContext->getTrackletsLookupTable()[iLayer - 1][iCluster] =
                primaryVertexContext->getTracklets()[iLayer].size();
            }

            primaryVertexContext->getTracklets()[iLayer].emplace_back(iCluster, iNextLayerCluster, currentCluster,
                                                                      nextCluster);
          }
        }
      }
    }
    if (iLayer > 0 && iLayer < mTrkParams.TrackletsPerRoad() - 1 &&
        primaryVertexContext->getTracklets()[iLayer].size() > primaryVertexContext->getCellsLookupTable()[iLayer - 1].size()) {
      throw std::runtime_error(fmt::format("not enough memory in the CellsLookupTable, increase the tracklet memory coefficients: {} tracklets on L{}, lookup table size {} on L{}",
                                           primaryVertexContext->getTracklets()[iLayer].size(), iLayer, primaryVertexContext->getCellsLookupTable()[iLayer - 1].size(), iLayer - 1));
    }
  }
#ifdef CA_DEBUG
  std::cout << "+++ Number of tracklets per layer: ";
  for (int iLayer{0}; iLayer < mTrkParams.TrackletsPerRoad(); ++iLayer) {
    std::cout << primaryVertexContext->getTracklets()[iLayer].size() << "\t";
  }
  std::cout << std::endl;
#endif
}

void TrackerTraitsCPU::computeLayerCells()
{
  PrimaryVertexContext* primaryVertexContext = mPrimaryVertexContext;
  for (int iLayer{0}; iLayer < mTrkParams.CellsPerRoad(); ++iLayer) {

    if (primaryVertexContext->getTracklets()[iLayer + 1].empty() ||
        primaryVertexContext->getTracklets()[iLayer].empty()) {

      return;
    }

    const float3& primaryVertex = primaryVertexContext->getPrimaryVertex();
    const int currentLayerTrackletsNum{static_cast<int>(primaryVertexContext->getTracklets()[iLayer].size())};

    for (int iTracklet{0}; iTracklet < currentLayerTrackletsNum; ++iTracklet) {

      const Tracklet& currentTracklet{primaryVertexContext->getTracklets()[iLayer][iTracklet]};
      const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
      const int nextLayerFirstTrackletIndex{
        primaryVertexContext->getTrackletsLookupTable()[iLayer][nextLayerClusterIndex]};

      if (nextLayerFirstTrackletIndex == constants::its::UnusedIndex) {

        continue;
      }

      const Cluster& firstCellCluster{primaryVertexContext->getClusters()[iLayer][currentTracklet.firstClusterIndex]};
      const Cluster& secondCellCluster{
        primaryVertexContext->getClusters()[iLayer + 1][currentTracklet.secondClusterIndex]};
      const float firstCellClusterQuadraticRCoordinate{firstCellCluster.rCoordinate * firstCellCluster.rCoordinate};
      const float secondCellClusterQuadraticRCoordinate{secondCellCluster.rCoordinate *
                                                        secondCellCluster.rCoordinate};
      const float3 firstDeltaVector{secondCellCluster.xCoordinate - firstCellCluster.xCoordinate,
                                    secondCellCluster.yCoordinate - firstCellCluster.yCoordinate,
                                    secondCellClusterQuadraticRCoordinate - firstCellClusterQuadraticRCoordinate};
      const int nextLayerTrackletsNum{static_cast<int>(primaryVertexContext->getTracklets()[iLayer + 1].size())};

      for (int iNextLayerTracklet{nextLayerFirstTrackletIndex};
           iNextLayerTracklet < nextLayerTrackletsNum &&
           primaryVertexContext->getTracklets()[iLayer + 1][iNextLayerTracklet].firstClusterIndex ==
             nextLayerClusterIndex;
           ++iNextLayerTracklet) {

        const Tracklet& nextTracklet{primaryVertexContext->getTracklets()[iLayer + 1][iNextLayerTracklet]};
        const float deltaTanLambda{std::abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};
        const float deltaPhi{std::abs(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate)};

        if (deltaTanLambda < mTrkParams.CellMaxDeltaTanLambda &&
            (deltaPhi < mTrkParams.CellMaxDeltaPhi ||
             std::abs(deltaPhi - constants::math::TwoPi) < mTrkParams.CellMaxDeltaPhi)) {

          const float averageTanLambda{0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda)};
          const float directionZIntersection{-averageTanLambda * firstCellCluster.rCoordinate +
                                             firstCellCluster.zCoordinate};
          const float deltaZ{std::abs(directionZIntersection - primaryVertex.z)};

          if (deltaZ < mTrkParams.CellMaxDeltaZ[iLayer]) {

            const Cluster& thirdCellCluster{
              primaryVertexContext->getClusters()[iLayer + 2][nextTracklet.secondClusterIndex]};

            const float thirdCellClusterQuadraticRCoordinate{thirdCellCluster.rCoordinate *
                                                             thirdCellCluster.rCoordinate};

            const float3 secondDeltaVector{thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate,
                                           thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate,
                                           thirdCellClusterQuadraticRCoordinate -
                                             firstCellClusterQuadraticRCoordinate};

            float3 cellPlaneNormalVector{math_utils::crossProduct(firstDeltaVector, secondDeltaVector)};

            const float vectorNorm{std::sqrt(cellPlaneNormalVector.x * cellPlaneNormalVector.x +
                                             cellPlaneNormalVector.y * cellPlaneNormalVector.y +
                                             cellPlaneNormalVector.z * cellPlaneNormalVector.z)};

            if (vectorNorm < constants::math::FloatMinThreshold ||
                std::abs(cellPlaneNormalVector.z) < constants::math::FloatMinThreshold) {

              continue;
            }

            const float inverseVectorNorm{1.0f / vectorNorm};
            const float3 normalizedPlaneVector{cellPlaneNormalVector.x * inverseVectorNorm,
                                               cellPlaneNormalVector.y * inverseVectorNorm,
                                               cellPlaneNormalVector.z * inverseVectorNorm};
            const float planeDistance{-normalizedPlaneVector.x * (secondCellCluster.xCoordinate - primaryVertex.x) -
                                      (normalizedPlaneVector.y * secondCellCluster.yCoordinate - primaryVertex.y) -
                                      normalizedPlaneVector.z * secondCellClusterQuadraticRCoordinate};
            const float normalizedPlaneVectorQuadraticZCoordinate{normalizedPlaneVector.z * normalizedPlaneVector.z};
            const float cellTrajectoryRadius{std::sqrt(
              (1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector.z) /
              (4.0f * normalizedPlaneVectorQuadraticZCoordinate))};
            const float2 circleCenter{-0.5f * normalizedPlaneVector.x / normalizedPlaneVector.z,
                                      -0.5f * normalizedPlaneVector.y / normalizedPlaneVector.z};
            const float distanceOfClosestApproach{std::abs(
              cellTrajectoryRadius - std::sqrt(circleCenter.x * circleCenter.x + circleCenter.y * circleCenter.y))};

            if (distanceOfClosestApproach >
                mTrkParams.CellMaxDCA[iLayer]) {

              continue;
            }

            const float cellTrajectoryCurvature{1.0f / cellTrajectoryRadius};
            if (iLayer > 0 &&
                primaryVertexContext->getCellsLookupTable()[iLayer - 1][iTracklet] == constants::its::UnusedIndex) {

              primaryVertexContext->getCellsLookupTable()[iLayer - 1][iTracklet] =
                primaryVertexContext->getCells()[iLayer].size();
            }

            primaryVertexContext->getCells()[iLayer].emplace_back(
              currentTracklet.firstClusterIndex, nextTracklet.firstClusterIndex, nextTracklet.secondClusterIndex,
              iTracklet, iNextLayerTracklet, normalizedPlaneVector, cellTrajectoryCurvature);
          }
        }
      }
    }
  }
#ifdef CA_DEBUG
  std::cout << "+++ Number of cells per layer: ";
  for (int iLayer{0}; iLayer < mTrkParams.CellsPerRoad(); ++iLayer) {
    std::cout << primaryVertexContext->getCells()[iLayer].size() << "\t";
  }
  std::cout << std::endl;
#endif
}

void TrackerTraitsCPU::refitTracks(const std::vector<std::vector<TrackingFrameInfo>>& tf, std::vector<TrackITSExt>& tracks)
{
  std::vector<const Cell*> cells;
  for (int iLayer = 0; iLayer < mTrkParams.CellsPerRoad(); iLayer++) {
    cells.push_back(mPrimaryVertexContext->getCells()[iLayer].data());
  }
  std::vector<const Cluster*> clusters;
  for (int iLayer = 0; iLayer < mTrkParams.NLayers; iLayer++) {
    clusters.push_back(mPrimaryVertexContext->getClusters()[iLayer].data());
  }
  mChainRunITSTrackFit(*mChain, mPrimaryVertexContext->getRoads(), clusters, cells, tf, tracks);
}

} // namespace its
} // namespace o2
