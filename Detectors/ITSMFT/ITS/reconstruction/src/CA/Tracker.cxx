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
/// \file Tracker.cxx
/// \brief
///

#include "ITSReconstruction/CA/Tracker.h"

#include "CommonConstants/MathConstants.h"
#include "ITSReconstruction/CA/Cell.h"
#include "ITSReconstruction/CA/Constants.h"
#include "ITSReconstruction/CA/IndexTableUtils.h"
#include "ITSReconstruction/CA/Layer.h"
#include "ITSReconstruction/CA/Tracklet.h"

#include "ReconstructionDataFormats/Track.h"
#include <cassert>
#include <iostream>

namespace o2
{
namespace ITS
{
namespace CA
{

#if !TRACKINGITSU_GPU_MODE
template <>
void TrackerTraits<false>::computeLayerTracklets(PrimaryVertexContext& primaryVertexContext, const TrackingParameters& trkPars, int iteration)
{
  for (int iLayer{ 0 }; iLayer < Constants::ITS::TrackletsPerRoad; ++iLayer) {
    if (primaryVertexContext.getClusters()[iLayer].empty() || primaryVertexContext.getClusters()[iLayer + 1].empty()) {
      return;
    }

    const float3& primaryVertex = primaryVertexContext.getPrimaryVertex();
    const int currentLayerClustersNum{ static_cast<int>(primaryVertexContext.getClusters()[iLayer].size()) };

    for (int iCluster{ 0 }; iCluster < currentLayerClustersNum; ++iCluster) {
      const Cluster& currentCluster{ primaryVertexContext.getClusters()[iLayer][iCluster] };

      const float tanLambda{ (currentCluster.zCoordinate - primaryVertex.z) / currentCluster.rCoordinate };
      const float directionZIntersection{ tanLambda * (Constants::ITS::LayersRCoordinate()[iLayer + 1] -
                                                       currentCluster.rCoordinate) +
                                          currentCluster.zCoordinate };

      const int4 selectedBinsRect{ getBinsRect(currentCluster, iLayer, directionZIntersection,
                                               trkPars.TrackletMaxDeltaZ[iteration][iLayer], trkPars.TrackletMaxDeltaPhi[iteration]) };

      if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
        continue;
      }

      int phiBinsNum{ selectedBinsRect.w - selectedBinsRect.y + 1 };

      if (phiBinsNum < 0) {
        phiBinsNum += Constants::IndexTable::PhiBins;
      }

      for (int iPhiBin{ selectedBinsRect.y }, iPhiCount{ 0 }; iPhiCount < phiBinsNum;
           iPhiBin = ++iPhiBin == Constants::IndexTable::PhiBins ? 0 : iPhiBin, iPhiCount++) {

        const int firstBinIndex{ IndexTableUtils::getBinIndex(selectedBinsRect.x, iPhiBin) };
        const int maxBinIndex{ firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1 };
        const int firstRowClusterIndex = primaryVertexContext.getIndexTables()[iLayer][firstBinIndex];
        const int maxRowClusterIndex = primaryVertexContext.getIndexTables()[iLayer][maxBinIndex];

        for (int iNextLayerCluster{ firstRowClusterIndex }; iNextLayerCluster < maxRowClusterIndex;
             ++iNextLayerCluster) {

          const Cluster& nextCluster{ primaryVertexContext.getClusters()[iLayer + 1][iNextLayerCluster] };

          if (primaryVertexContext.isClusterUsed(iLayer + 1, nextCluster.clusterId))
            continue;

          const float deltaZ{ MATH_ABS(tanLambda * (nextCluster.rCoordinate - currentCluster.rCoordinate) +
                                       currentCluster.zCoordinate - nextCluster.zCoordinate) };
          const float deltaPhi{ MATH_ABS(currentCluster.phiCoordinate - nextCluster.phiCoordinate) };

          if (deltaZ < trkPars.TrackletMaxDeltaZ[iteration][iLayer] &&
              (deltaPhi < trkPars.TrackletMaxDeltaPhi[iteration] ||
               MATH_ABS(deltaPhi - Constants::Math::TwoPi) < trkPars.TrackletMaxDeltaPhi[iteration])) {

            if (iLayer > 0 &&
                primaryVertexContext.getTrackletsLookupTable()[iLayer - 1][iCluster] == Constants::ITS::UnusedIndex) {

              primaryVertexContext.getTrackletsLookupTable()[iLayer - 1][iCluster] =
                primaryVertexContext.getTracklets()[iLayer].size();
            }

            primaryVertexContext.getTracklets()[iLayer].emplace_back(iCluster, iNextLayerCluster, currentCluster,
                                                                     nextCluster);
          }
        }
      }
    }
  }
}

template <>
void TrackerTraits<false>::computeLayerCells(PrimaryVertexContext& primaryVertexContext, const TrackingParameters& trkPars, int iteration)
{
  for (int iLayer{ 0 }; iLayer < Constants::ITS::CellsPerRoad; ++iLayer) {

    if (primaryVertexContext.getTracklets()[iLayer + 1].empty() ||
        primaryVertexContext.getTracklets()[iLayer].empty()) {

      return;
    }

    const float3& primaryVertex = primaryVertexContext.getPrimaryVertex();
    const int currentLayerTrackletsNum{ static_cast<int>(primaryVertexContext.getTracklets()[iLayer].size()) };

    for (int iTracklet{ 0 }; iTracklet < currentLayerTrackletsNum; ++iTracklet) {

      const Tracklet& currentTracklet{ primaryVertexContext.getTracklets()[iLayer][iTracklet] };
      const int nextLayerClusterIndex{ currentTracklet.secondClusterIndex };
      const int nextLayerFirstTrackletIndex{
        primaryVertexContext.getTrackletsLookupTable()[iLayer][nextLayerClusterIndex]
      };

      if (nextLayerFirstTrackletIndex == Constants::ITS::UnusedIndex) {

        continue;
      }

      const Cluster& firstCellCluster{ primaryVertexContext.getClusters()[iLayer][currentTracklet.firstClusterIndex] };
      const Cluster& secondCellCluster{
        primaryVertexContext.getClusters()[iLayer + 1][currentTracklet.secondClusterIndex]
      };
      const float firstCellClusterQuadraticRCoordinate{ firstCellCluster.rCoordinate * firstCellCluster.rCoordinate };
      const float secondCellClusterQuadraticRCoordinate{ secondCellCluster.rCoordinate *
                                                         secondCellCluster.rCoordinate };
      const float3 firstDeltaVector{ secondCellCluster.xCoordinate - firstCellCluster.xCoordinate,
                                     secondCellCluster.yCoordinate - firstCellCluster.yCoordinate,
                                     secondCellClusterQuadraticRCoordinate - firstCellClusterQuadraticRCoordinate };
      const int nextLayerTrackletsNum{ static_cast<int>(primaryVertexContext.getTracklets()[iLayer + 1].size()) };

      for (int iNextLayerTracklet{ nextLayerFirstTrackletIndex };
           iNextLayerTracklet < nextLayerTrackletsNum &&
           primaryVertexContext.getTracklets()[iLayer + 1][iNextLayerTracklet].firstClusterIndex ==
             nextLayerClusterIndex;
           ++iNextLayerTracklet) {

        const Tracklet& nextTracklet{ primaryVertexContext.getTracklets()[iLayer + 1][iNextLayerTracklet] };
        const float deltaTanLambda{ std::abs(currentTracklet.tanLambda - nextTracklet.tanLambda) };
        const float deltaPhi{ std::abs(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate) };

        if (deltaTanLambda < trkPars.CellMaxDeltaTanLambda[iteration] &&
            (deltaPhi < trkPars.CellMaxDeltaPhi[iteration] ||
             std::abs(deltaPhi - Constants::Math::TwoPi) < trkPars.CellMaxDeltaPhi[iteration])) {

          const float averageTanLambda{ 0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda) };
          const float directionZIntersection{ -averageTanLambda * firstCellCluster.rCoordinate +
                                              firstCellCluster.zCoordinate };
          const float deltaZ{ std::abs(directionZIntersection - primaryVertex.z) };

          if (deltaZ < trkPars.CellMaxDeltaZ[iteration][iLayer]) {

            const Cluster& thirdCellCluster{
              primaryVertexContext.getClusters()[iLayer + 2][nextTracklet.secondClusterIndex]
            };

            const float thirdCellClusterQuadraticRCoordinate{ thirdCellCluster.rCoordinate *
                                                              thirdCellCluster.rCoordinate };

            const float3 secondDeltaVector{ thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate,
                                            thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate,
                                            thirdCellClusterQuadraticRCoordinate -
                                              firstCellClusterQuadraticRCoordinate };

            float3 cellPlaneNormalVector{ MathUtils::crossProduct(firstDeltaVector, secondDeltaVector) };

            const float vectorNorm{ std::sqrt(cellPlaneNormalVector.x * cellPlaneNormalVector.x +
                                              cellPlaneNormalVector.y * cellPlaneNormalVector.y +
                                              cellPlaneNormalVector.z * cellPlaneNormalVector.z) };

            if (vectorNorm < Constants::Math::FloatMinThreshold ||
                std::abs(cellPlaneNormalVector.z) < Constants::Math::FloatMinThreshold) {

              continue;
            }

            const float inverseVectorNorm{ 1.0f / vectorNorm };
            const float3 normalizedPlaneVector{ cellPlaneNormalVector.x * inverseVectorNorm,
                                                cellPlaneNormalVector.y * inverseVectorNorm,
                                                cellPlaneNormalVector.z * inverseVectorNorm };
            const float planeDistance{ -normalizedPlaneVector.x * (secondCellCluster.xCoordinate - primaryVertex.x) -
                                       (normalizedPlaneVector.y * secondCellCluster.yCoordinate - primaryVertex.y) -
                                       normalizedPlaneVector.z * secondCellClusterQuadraticRCoordinate };
            const float normalizedPlaneVectorQuadraticZCoordinate{ normalizedPlaneVector.z * normalizedPlaneVector.z };
            const float cellTrajectoryRadius{ std::sqrt(
              (1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector.z) /
              (4.0f * normalizedPlaneVectorQuadraticZCoordinate)) };
            const float2 circleCenter{ -0.5f * normalizedPlaneVector.x / normalizedPlaneVector.z,
                                       -0.5f * normalizedPlaneVector.y / normalizedPlaneVector.z };
            const float distanceOfClosestApproach{ std::abs(
              cellTrajectoryRadius - std::sqrt(circleCenter.x * circleCenter.x + circleCenter.y * circleCenter.y)) };

            if (distanceOfClosestApproach >
                trkPars.CellMaxDCA[iteration][iLayer]) {

              continue;
            }

            const float cellTrajectoryCurvature{ 1.0f / cellTrajectoryRadius };
            if (iLayer > 0 &&
                primaryVertexContext.getCellsLookupTable()[iLayer - 1][iTracklet] == Constants::ITS::UnusedIndex) {

              primaryVertexContext.getCellsLookupTable()[iLayer - 1][iTracklet] =
                primaryVertexContext.getCells()[iLayer].size();
            }

            primaryVertexContext.getCells()[iLayer].emplace_back(
              currentTracklet.firstClusterIndex, nextTracklet.firstClusterIndex, nextTracklet.secondClusterIndex,
              iTracklet, iNextLayerTracklet, normalizedPlaneVector, cellTrajectoryCurvature);
          }
        }
      }
    }
  }
}
#endif

template <bool IsGPU>
Tracker<IsGPU>::Tracker()
{
  // Nothing to do
}

template <bool IsGPU>
void Tracker<IsGPU>::clustersToTracks(const Event& event, std::ostream& timeBenchmarkOutputStream)
{
  const int verticesNum = event.getPrimaryVerticesNum();
  mTracks.clear();
  mTrackLabels.clear();

  for (int iVertex = 0; iVertex < verticesNum; ++iVertex) {

    float total{ 0.f };

    for (int iteration = 0; iteration < mTrkParams.NumberOfIterations(); ++iteration) {
      total += evaluateTask(&Tracker<IsGPU>::initialisePrimaryVertexContext, "Context initialisation",
                            timeBenchmarkOutputStream, event, iVertex, iteration);
      total += evaluateTask(&Tracker<IsGPU>::computeTracklets, "Tracklet finding", timeBenchmarkOutputStream, iteration);
      total += evaluateTask(&Tracker<IsGPU>::computeCells, "Cell finding", timeBenchmarkOutputStream, iteration);
      total += evaluateTask(&Tracker<IsGPU>::findCellsNeighbours, "Neighbour finding", timeBenchmarkOutputStream, iteration);
      total += evaluateTask(&Tracker<IsGPU>::findRoads, "Road finding", timeBenchmarkOutputStream, iteration);
      total += evaluateTask(&Tracker<IsGPU>::findTracks, "Track finding", timeBenchmarkOutputStream, event);
    }
    total += evaluateTask(&Tracker<IsGPU>::computeTracksMClabels, "Tracks Monte Carlo labels computation",
                          timeBenchmarkOutputStream, event);

    if (Constants::DoTimeBenchmarks)
      timeBenchmarkOutputStream << std::setw(2) << " - "
                                << "Vertex processing completed in: " << total << "ms" << std::endl;

    mTracks.insert(mTracks.end(), mPrimaryVertexContext.getTracks().begin(), mPrimaryVertexContext.getTracks().end());
    for (auto& label : mPrimaryVertexContext.getTrackLabels()) {
      mTrackLabels.addElement(mTrackLabels.getIndexedSize(), label);
    }
  }
}

template <bool IsGPU>
void Tracker<IsGPU>::computeTracklets(int& iteration)
{
  Trait::computeLayerTracklets(mPrimaryVertexContext, mTrkParams, iteration);
}

template <bool IsGPU>
void Tracker<IsGPU>::computeCells(int& iteration)
{
  Trait::computeLayerCells(mPrimaryVertexContext, mTrkParams, iteration);
}

template <bool IsGPU>
void Tracker<IsGPU>::findCellsNeighbours(int& iteration)
{
  for (int iLayer{ 0 }; iLayer < Constants::ITS::CellsPerRoad - 1; ++iLayer) {

    if (mPrimaryVertexContext.getCells()[iLayer + 1].empty() ||
        mPrimaryVertexContext.getCellsLookupTable()[iLayer].empty()) {
      continue;
    }

    int layerCellsNum{ static_cast<int>(mPrimaryVertexContext.getCells()[iLayer].size()) };

    for (int iCell{ 0 }; iCell < layerCellsNum; ++iCell) {

      const Cell& currentCell{ mPrimaryVertexContext.getCells()[iLayer][iCell] };
      const int nextLayerTrackletIndex{ currentCell.getSecondTrackletIndex() };
      const int nextLayerFirstCellIndex{ mPrimaryVertexContext.getCellsLookupTable()[iLayer][nextLayerTrackletIndex] };
      if (nextLayerFirstCellIndex != Constants::ITS::UnusedIndex &&
          mPrimaryVertexContext.getCells()[iLayer + 1][nextLayerFirstCellIndex].getFirstTrackletIndex() ==
            nextLayerTrackletIndex) {

        const int nextLayerCellsNum{ static_cast<int>(mPrimaryVertexContext.getCells()[iLayer + 1].size()) };
        mPrimaryVertexContext.getCellsNeighbours()[iLayer].resize(nextLayerCellsNum);

        for (int iNextLayerCell{ nextLayerFirstCellIndex };
             iNextLayerCell < nextLayerCellsNum &&
             mPrimaryVertexContext.getCells()[iLayer + 1][iNextLayerCell].getFirstTrackletIndex() ==
               nextLayerTrackletIndex;
             ++iNextLayerCell) {

          Cell& nextCell{ mPrimaryVertexContext.getCells()[iLayer + 1][iNextLayerCell] };
          const float3 currentCellNormalVector{ currentCell.getNormalVectorCoordinates() };
          const float3 nextCellNormalVector{ nextCell.getNormalVectorCoordinates() };
          const float3 normalVectorsDeltaVector{ currentCellNormalVector.x - nextCellNormalVector.x,
                                                 currentCellNormalVector.y - nextCellNormalVector.y,
                                                 currentCellNormalVector.z - nextCellNormalVector.z };

          const float deltaNormalVectorsModulus{ (normalVectorsDeltaVector.x * normalVectorsDeltaVector.x) +
                                                 (normalVectorsDeltaVector.y * normalVectorsDeltaVector.y) +
                                                 (normalVectorsDeltaVector.z * normalVectorsDeltaVector.z) };
          const float deltaCurvature{ std::abs(currentCell.getCurvature() - nextCell.getCurvature()) };

          if (deltaNormalVectorsModulus < mTrkParams.NeighbourMaxDeltaN[iteration][iLayer] &&
              deltaCurvature < mTrkParams.NeighbourMaxDeltaCurvature[iteration][iLayer]) {

            mPrimaryVertexContext.getCellsNeighbours()[iLayer][iNextLayerCell].push_back(iCell);

            const int currentCellLevel{ currentCell.getLevel() };

            if (currentCellLevel >= nextCell.getLevel()) {

              nextCell.setLevel(currentCellLevel + 1);
            }
          }
        }
      }
    }
  }
}

template <bool IsGPU>
void Tracker<IsGPU>::findRoads(int& iteration)
{
  for (int iLevel{ Constants::ITS::CellsPerRoad }; iLevel >= mTrkParams.CellMinimumLevel(iteration); --iLevel) {
    int nRoads = -mPrimaryVertexContext.getRoads().size();
    const int minimumLevel{ iLevel - 1 };

    for (int iLayer{ Constants::ITS::CellsPerRoad - 1 }; iLayer >= minimumLevel; --iLayer) {

      const int levelCellsNum{ static_cast<int>(mPrimaryVertexContext.getCells()[iLayer].size()) };

      for (int iCell{ 0 }; iCell < levelCellsNum; ++iCell) {

        Cell& currentCell{ mPrimaryVertexContext.getCells()[iLayer][iCell] };

        if (currentCell.getLevel() != iLevel) {

          continue;
        }

        mPrimaryVertexContext.getRoads().emplace_back(iLayer, iCell);

        const int cellNeighboursNum{ static_cast<int>(
          mPrimaryVertexContext.getCellsNeighbours()[iLayer - 1][iCell].size()) };
        bool isFirstValidNeighbour = true;

        for (int iNeighbourCell{ 0 }; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

          const int neighbourCellId = mPrimaryVertexContext.getCellsNeighbours()[iLayer - 1][iCell][iNeighbourCell];
          const Cell& neighbourCell = mPrimaryVertexContext.getCells()[iLayer - 1][neighbourCellId];

          if (iLevel - 1 != neighbourCell.getLevel()) {
            continue;
          }

          if (isFirstValidNeighbour) {

            isFirstValidNeighbour = false;

          } else {

            mPrimaryVertexContext.getRoads().emplace_back(iLayer, iCell);
          }

          traverseCellsTree(neighbourCellId, iLayer - 1);
        }

        // TODO: crosscheck for short track iterations
        // currentCell.setLevel(0);
      }
    }
    nRoads += mPrimaryVertexContext.getRoads().size();
  }
}

template <bool IsGPU>
void Tracker<IsGPU>::findTracks(const Event& event)
{
  mPrimaryVertexContext.getTracks().reserve(mPrimaryVertexContext.getTracks().capacity() + mPrimaryVertexContext.getRoads().size());
  std::vector<TrackITS> tracks;
  tracks.reserve(mPrimaryVertexContext.getRoads().size());
  for (auto& road : mPrimaryVertexContext.getRoads()) {
    std::array<int, 7> clusters{ Constants::ITS::UnusedIndex };
    int lastCellLevel = Constants::ITS::UnusedIndex;
    for (int iCell{ 0 }; iCell < Constants::ITS::CellsPerRoad; ++iCell) {
      const int cellIndex = road[iCell];
      if (cellIndex == Constants::ITS::UnusedIndex) {
        continue;
      } else {
        clusters[iCell] = mPrimaryVertexContext.getCells()[iCell][cellIndex].getFirstClusterIndex();
        clusters[iCell + 1] = mPrimaryVertexContext.getCells()[iCell][cellIndex].getSecondClusterIndex();
        clusters[iCell + 2] = mPrimaryVertexContext.getCells()[iCell][cellIndex].getThirdClusterIndex();
        lastCellLevel = iCell;
      }
    }
    if (lastCellLevel == Constants::ITS::UnusedIndex)
      continue;

    /// From primary vertex context index to event index (== the one used as input of the tracking code)
    for (int iC{ 0 }; iC < clusters.size(); iC++) {
      if (clusters[iC] != Constants::ITS::UnusedIndex) {
        clusters[iC] = mPrimaryVertexContext.getClusters()[iC][clusters[iC]].clusterId;
      }
    }
    /// Track seed preparation. Clusters are numbered progressively from the outermost to the innermost.
    const auto& cluster1_glo = event.getLayer(lastCellLevel + 2).getCluster(clusters[lastCellLevel + 2]);
    const auto& cluster2_glo = event.getLayer(lastCellLevel + 1).getCluster(clusters[lastCellLevel + 1]);
    const auto& cluster3_glo = event.getLayer(lastCellLevel).getCluster(clusters[lastCellLevel]);

    const auto& cluster3_tf = event.getLayer(lastCellLevel).getTrackingFrameInfo(clusters[lastCellLevel]);

    /// FIXME!
    TrackITS temporaryTrack{ buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_glo, cluster3_tf) };
    for (size_t iC = 0; iC < clusters.size(); ++iC) {
      temporaryTrack.setExternalClusterIndex(iC, clusters[iC], clusters[iC] != Constants::ITS::UnusedIndex);
    }
    bool fitSuccess = fitTrack(event, temporaryTrack, Constants::ITS::LayersNumber - 4, -1, -1);
    if (!fitSuccess)
      continue;
    temporaryTrack.resetCovariance();
    fitSuccess = fitTrack(event, temporaryTrack, 0, Constants::ITS::LayersNumber, 1);
    if (!fitSuccess)
      continue;
    temporaryTrack.getParamOut() = temporaryTrack;
    temporaryTrack.resetCovariance();
    fitSuccess = fitTrack(event, temporaryTrack, Constants::ITS::LayersNumber - 1, -1, -1);
    if (!fitSuccess)
      continue;
    temporaryTrack.setROFrame(mROFrame);
    tracks.emplace_back(temporaryTrack);
  }

  std::sort(tracks.begin(), tracks.end(),
            [](TrackITS& track1, TrackITS& track2) { return track1.isBetter(track2, 1.e6f); });

  for (auto& track : tracks) {
    bool sharingCluster = false;
    for (int iLayer{ 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {
      if (track.getClusterIndex(iLayer) == Constants::ITS::UnusedIndex) {
        continue;
      }
      sharingCluster |= mPrimaryVertexContext.isClusterUsed(iLayer, track.getClusterIndex(iLayer));
    }
    if (sharingCluster) {
      continue;
    }
    for (int iLayer{ 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {
      if (track.getClusterIndex(iLayer) == Constants::ITS::UnusedIndex) {
        continue;
      }
      mPrimaryVertexContext.markUsedCluster(iLayer, track.getClusterIndex(iLayer));
    }
    mPrimaryVertexContext.getTracks().emplace_back(track);
  }
}

template <bool IsGPU>
bool Tracker<IsGPU>::fitTrack(const Event& event, TrackITS& track, int start, int end, int step)
{
  track.setChi2(0);
  for (int iLayer{ start }; iLayer != end; iLayer += step) {
    if (track.getClusterIndex(iLayer) == Constants::ITS::UnusedIndex) {
      continue;
    }
    const TrackingFrameInfo& trackingHit = event.getLayer(iLayer).getTrackingFrameInfo(track.getClusterIndex(iLayer));

    if (!track.rotate(trackingHit.alphaTrackingFrame))
      return false;

    if (!track.propagateTo(trackingHit.xTrackingFrame, getBz()))
      return false;

    track.setChi2(track.getChi2() +
                  track.getPredictedChi2(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame));
    if (!track.TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame))
      return false;

    const float xx0 = (iLayer > 2) ? 0.008f : 0.003f; // Rough layer thickness
    constexpr float radiationLength = 9.36f;          // Radiation length of Si [cm]
    constexpr float density = 2.33f;                  // Density of Si [g/cm^3]
    if (!track.correctForMaterial(xx0, xx0 * radiationLength * density, true))
      return false;
  }
  return true;
}

template <bool IsGPU>
void Tracker<IsGPU>::traverseCellsTree(const int currentCellId, const int currentLayerId)
{
  Cell& currentCell{ mPrimaryVertexContext.getCells()[currentLayerId][currentCellId] };
  const int currentCellLevel = currentCell.getLevel();

  mPrimaryVertexContext.getRoads().back().addCell(currentLayerId, currentCellId);

  if (currentLayerId > 0) {

    const int cellNeighboursNum{ static_cast<int>(
      mPrimaryVertexContext.getCellsNeighbours()[currentLayerId - 1][currentCellId].size()) };
    bool isFirstValidNeighbour = true;

    for (int iNeighbourCell{ 0 }; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

      const int neighbourCellId =
        mPrimaryVertexContext.getCellsNeighbours()[currentLayerId - 1][currentCellId][iNeighbourCell];
      const Cell& neighbourCell = mPrimaryVertexContext.getCells()[currentLayerId - 1][neighbourCellId];

      if (currentCellLevel - 1 != neighbourCell.getLevel()) {
        continue;
      }

      if (isFirstValidNeighbour) {
        isFirstValidNeighbour = false;
      } else {
        mPrimaryVertexContext.getRoads().push_back(mPrimaryVertexContext.getRoads().back());
      }

      traverseCellsTree(neighbourCellId, currentLayerId - 1);
    }
  }

  // TODO: crosscheck for short track iterations
  // currentCell.setLevel(0);
}

template <bool IsGPU>
void Tracker<IsGPU>::computeRoadsMClabels(const Event& event)
{
  /// Moore's Voting Algorithm

  int roadsNum{ static_cast<int>(mPrimaryVertexContext.getRoads().size()) };

  for (int iRoad{ 0 }; iRoad < roadsNum; ++iRoad) {

    Road& currentRoad{ mPrimaryVertexContext.getRoads()[iRoad] };
    int maxOccurrencesValue{ Constants::ITS::UnusedIndex };
    int count{ 0 };
    bool isFakeRoad{ false };
    bool isFirstRoadCell{ true };

    for (int iCell{ 0 }; iCell < Constants::ITS::CellsPerRoad; ++iCell) {
      const int currentCellIndex{ currentRoad[iCell] };

      if (currentCellIndex == Constants::ITS::UnusedIndex) {
        if (isFirstRoadCell) {
          continue;
        } else {
          break;
        }
      }

      const Cell& currentCell{ mPrimaryVertexContext.getCells()[iCell][currentCellIndex] };

      if (isFirstRoadCell) {

        const int cl0index{ mPrimaryVertexContext.getClusters()[iCell][currentCell.getFirstClusterIndex()].clusterId };
        auto& cl0labs{ event.getLayer(iCell).getClusterLabel(cl0index) };
        maxOccurrencesValue = cl0labs.getTrackID();
        count = 1;

        const int cl1index{
          mPrimaryVertexContext.getClusters()[iCell + 1][currentCell.getSecondClusterIndex()].clusterId
        };
        auto& cl1labs{ event.getLayer(iCell + 1).getClusterLabel(cl1index) };
        const int secondMonteCarlo{ cl1labs.getTrackID() };

        if (secondMonteCarlo == maxOccurrencesValue) {
          ++count;
        } else {
          maxOccurrencesValue = secondMonteCarlo;
          count = 1;
          isFakeRoad = true;
        }

        isFirstRoadCell = false;
      }

      const int cl2index{
        mPrimaryVertexContext.getClusters()[iCell + 2][currentCell.getThirdClusterIndex()].clusterId
      };
      auto& cl2labs{ event.getLayer(iCell + 2).getClusterLabel(cl2index) };
      const int currentMonteCarlo = { cl2labs.getTrackID() };

      if (currentMonteCarlo == maxOccurrencesValue) {
        ++count;
      } else {
        --count;
        isFakeRoad = true;
      }

      if (count == 0) {
        maxOccurrencesValue = currentMonteCarlo;
        count = 1;
      }
    }

    currentRoad.setLabel(maxOccurrencesValue);
    currentRoad.setFakeRoad(isFakeRoad);
  }
}

template <bool IsGPU>
void Tracker<IsGPU>::computeTracksMClabels(const Event& event)
{
  /// Moore's Voting Algorithm

  int tracksNum{ static_cast<int>(mPrimaryVertexContext.getTracks().size()) };

  for (TrackITS& track : mPrimaryVertexContext.getTracks()) {

    MCCompLabel maxOccurrencesValue{ Constants::ITS::UnusedIndex, Constants::ITS::UnusedIndex,
                                     Constants::ITS::UnusedIndex };
    int count{ 0 };
    bool isFakeTrack{ false };

    for (int iCluster = 0; iCluster < TrackITS::MaxClusters; ++iCluster) {
      const int index = track.getClusterIndex(iCluster);
      if (index == Constants::ITS::UnusedIndex) {
        continue;
      }

      const MCCompLabel& currentLabel = event.getLayer(iCluster).getClusterLabel(index);
      if (currentLabel == maxOccurrencesValue) {
        ++count;
      } else {
        if (count != 0) { // only in the first iteration count can be 0 at this point
          isFakeTrack = true;
          --count;
        }
        if (count == 0) {
          maxOccurrencesValue = currentLabel;
          count = 1;
        }
      }
    }

    if (isFakeTrack)
      maxOccurrencesValue.set(-maxOccurrencesValue.getTrackID(), maxOccurrencesValue.getEventID(),
                              maxOccurrencesValue.getSourceID());
    mPrimaryVertexContext.getTrackLabels().push_back(maxOccurrencesValue);
  }
}

/// Clusters are given from outside inward (cluster1 is the outermost). The innermost cluster is given in the tracking
/// frame coordinates
/// whereas the others are referred to the global frame. This function is almost a clone of CookSeed, adapted to return
/// a TrackParCov
template <bool IsGPU>
track::TrackParCov Tracker<IsGPU>::buildTrackSeed(const Cluster& cluster1, const Cluster& cluster2,
                                                  const Cluster& cluster3, const TrackingFrameInfo& tf3)
{
  const float ca = std::cos(tf3.alphaTrackingFrame), sa = std::sin(tf3.alphaTrackingFrame);
  const float x1 = cluster1.xCoordinate * ca + cluster1.yCoordinate * sa;
  const float y1 = -cluster1.xCoordinate * sa + cluster1.yCoordinate * ca;
  const float z1 = cluster1.zCoordinate;
  const float x2 = cluster2.xCoordinate * ca + cluster2.yCoordinate * sa;
  const float y2 = -cluster2.xCoordinate * sa + cluster2.yCoordinate * ca;
  const float z2 = cluster2.zCoordinate;
  const float x3 = tf3.xTrackingFrame;
  const float y3 = tf3.positionTrackingFrame[0];
  const float z3 = tf3.positionTrackingFrame[1];

  const float crv = MathUtils::computeCurvature(x1, y1, x2, y2, x3, y3);
  const float x0 = MathUtils::computeCurvatureCentreX(x1, y1, x2, y2, x3, y3);
  const float tgl12 = MathUtils::computeTanDipAngle(x1, y1, x2, y2, z1, z2);
  const float tgl23 = MathUtils::computeTanDipAngle(x2, y2, x3, y3, z2, z3);

  const float fy = 1. / (cluster2.rCoordinate - cluster3.rCoordinate);
  const float& tz = fy;
  const float cy = (MathUtils::computeCurvature(x1, y1, x2, y2 + Constants::ITS::Resolution, x3, y3) - crv) /
                   (Constants::ITS::Resolution * getBz() * constants::math::B2C) *
                   20.f; // FIXME: MS contribution to the cov[14] (*20 added)
  constexpr float s2 = Constants::ITS::Resolution * Constants::ITS::Resolution;

  return track::TrackParCov(tf3.xTrackingFrame, tf3.alphaTrackingFrame,
                            { y3, z3, crv * (x3 - x0), 0.5f * (tgl12 + tgl23),
                              std::abs(getBz()) < constants::math::Almost0 ? constants::math::Almost0
                                                                           : crv / (getBz() * constants::math::B2C) },
                            { s2, 0.f, s2, s2 * fy, 0.f, s2 * fy * fy, 0.f, s2 * tz, 0.f, s2 * tz * tz, s2 * cy, 0.f,
                              s2 * fy * cy, 0.f, s2 * cy * cy });
}

template class Tracker<TRACKINGITSU_GPU_MODE>;
} // namespace CA
} // namespace ITS
} // namespace o2
