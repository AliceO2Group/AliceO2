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

#include "ITStracking/Tracker.h"

#include "CommonConstants/MathConstants.h"
#include "ITStracking/Cell.h"
#include "ITStracking/Constants.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/TrackerTraitsCPU.h"

#include "ReconstructionDataFormats/Track.h"
#include <cassert>
#include <iostream>
#include <dlfcn.h>
#include <cstdlib>
#include <string>

namespace o2
{
namespace its
{

Tracker::Tracker(o2::its::TrackerTraits* traits)
{
  /// Initialise standard configuration with 1 iteration
  mTrkParams.resize(1);
  mMemParams.resize(1);
  assert(mTracks != nullptr);
  mTraits = traits;
  mPrimaryVertexContext = mTraits->getPrimaryVertexContext();
}

Tracker::~Tracker() = default;

void Tracker::clustersToTracks(const ROframe& event, std::ostream& timeBenchmarkOutputStream)
{
  const int verticesNum = event.getPrimaryVerticesNum();
  mTracks.clear();
  mTrackLabels.clear();

  for (int iVertex = 0; iVertex < verticesNum; ++iVertex) {

    float total{ 0.f };

    for (int iteration = 0; iteration < mTrkParams.size(); ++iteration) {
      mTraits->UpdateTrackingParameters(mTrkParams[iteration]);
      /// Ugly hack -> Unifiy float3 definition in CPU and CUDA/HIP code
      std::array<float, 3> pV = { event.getPrimaryVertex(iVertex).x, event.getPrimaryVertex(iVertex).y, event.getPrimaryVertex(iVertex).z };
      total += evaluateTask(&Tracker::initialisePrimaryVertexContext, "Context initialisation",
                            timeBenchmarkOutputStream, mMemParams[iteration], event.getClusters(), pV, iteration);
      total += evaluateTask(&Tracker::computeTracklets, "Tracklet finding", timeBenchmarkOutputStream);
      total += evaluateTask(&Tracker::computeCells, "Cell finding", timeBenchmarkOutputStream);
      total += evaluateTask(&Tracker::findCellsNeighbours, "Neighbour finding", timeBenchmarkOutputStream, iteration);
      total += evaluateTask(&Tracker::findRoads, "Road finding", timeBenchmarkOutputStream, iteration);
      total += evaluateTask(&Tracker::findTracks, "Track finding", timeBenchmarkOutputStream, event);
    }

    if (constants::DoTimeBenchmarks)
      timeBenchmarkOutputStream << std::setw(2) << " - "
                                << "Vertex processing completed in: " << total << "ms" << std::endl;
  }
  computeTracksMClabels(event);
}

void Tracker::computeTracklets()
{
  mTraits->computeLayerTracklets();
}

void Tracker::computeCells()
{
  mTraits->computeLayerCells();
}

void Tracker::findCellsNeighbours(int& iteration)
{
  for (int iLayer{ 0 }; iLayer < constants::its::CellsPerRoad - 1; ++iLayer) {

    if (mPrimaryVertexContext->getCells()[iLayer + 1].empty() ||
        mPrimaryVertexContext->getCellsLookupTable()[iLayer].empty()) {
      continue;
    }

    int layerCellsNum{ static_cast<int>(mPrimaryVertexContext->getCells()[iLayer].size()) };

    for (int iCell{ 0 }; iCell < layerCellsNum; ++iCell) {

      const Cell& currentCell{ mPrimaryVertexContext->getCells()[iLayer][iCell] };
      const int nextLayerTrackletIndex{ currentCell.getSecondTrackletIndex() };
      const int nextLayerFirstCellIndex{ mPrimaryVertexContext->getCellsLookupTable()[iLayer][nextLayerTrackletIndex] };
      if (nextLayerFirstCellIndex != constants::its::UnusedIndex &&
          mPrimaryVertexContext->getCells()[iLayer + 1][nextLayerFirstCellIndex].getFirstTrackletIndex() ==
            nextLayerTrackletIndex) {

        const int nextLayerCellsNum{ static_cast<int>(mPrimaryVertexContext->getCells()[iLayer + 1].size()) };
        mPrimaryVertexContext->getCellsNeighbours()[iLayer].resize(nextLayerCellsNum);

        for (int iNextLayerCell{ nextLayerFirstCellIndex };
             iNextLayerCell < nextLayerCellsNum &&
             mPrimaryVertexContext->getCells()[iLayer + 1][iNextLayerCell].getFirstTrackletIndex() ==
               nextLayerTrackletIndex;
             ++iNextLayerCell) {

          Cell& nextCell{ mPrimaryVertexContext->getCells()[iLayer + 1][iNextLayerCell] };
          const float3 currentCellNormalVector{ currentCell.getNormalVectorCoordinates() };
          const float3 nextCellNormalVector{ nextCell.getNormalVectorCoordinates() };
          const float3 normalVectorsDeltaVector{ currentCellNormalVector.x - nextCellNormalVector.x,
                                                 currentCellNormalVector.y - nextCellNormalVector.y,
                                                 currentCellNormalVector.z - nextCellNormalVector.z };

          const float deltaNormalVectorsModulus{ (normalVectorsDeltaVector.x * normalVectorsDeltaVector.x) +
                                                 (normalVectorsDeltaVector.y * normalVectorsDeltaVector.y) +
                                                 (normalVectorsDeltaVector.z * normalVectorsDeltaVector.z) };
          const float deltaCurvature{ std::abs(currentCell.getCurvature() - nextCell.getCurvature()) };

          if (deltaNormalVectorsModulus < mTrkParams[iteration].NeighbourMaxDeltaN[iLayer] &&
              deltaCurvature < mTrkParams[iteration].NeighbourMaxDeltaCurvature[iLayer]) {

            mPrimaryVertexContext->getCellsNeighbours()[iLayer][iNextLayerCell].push_back(iCell);

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

void Tracker::findRoads(int& iteration)
{
  for (int iLevel{ constants::its::CellsPerRoad }; iLevel >= mTrkParams[iteration].CellMinimumLevel(); --iLevel) {
    CA_DEBUGGER(int nRoads = -mPrimaryVertexContext->getRoads().size());
    const int minimumLevel{ iLevel - 1 };

    for (int iLayer{ constants::its::CellsPerRoad - 1 }; iLayer >= minimumLevel; --iLayer) {

      const int levelCellsNum{ static_cast<int>(mPrimaryVertexContext->getCells()[iLayer].size()) };

      for (int iCell{ 0 }; iCell < levelCellsNum; ++iCell) {

        Cell& currentCell{ mPrimaryVertexContext->getCells()[iLayer][iCell] };

        if (currentCell.getLevel() != iLevel) {

          continue;
        }

        mPrimaryVertexContext->getRoads().emplace_back(iLayer, iCell);

        const int cellNeighboursNum{ static_cast<int>(
          mPrimaryVertexContext->getCellsNeighbours()[iLayer - 1][iCell].size()) };
        bool isFirstValidNeighbour = true;

        for (int iNeighbourCell{ 0 }; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

          const int neighbourCellId = mPrimaryVertexContext->getCellsNeighbours()[iLayer - 1][iCell][iNeighbourCell];
          const Cell& neighbourCell = mPrimaryVertexContext->getCells()[iLayer - 1][neighbourCellId];

          if (iLevel - 1 != neighbourCell.getLevel()) {
            continue;
          }

          if (isFirstValidNeighbour) {

            isFirstValidNeighbour = false;

          } else {

            mPrimaryVertexContext->getRoads().emplace_back(iLayer, iCell);
          }

          traverseCellsTree(neighbourCellId, iLayer - 1);
        }

        // TODO: crosscheck for short track iterations
        // currentCell.setLevel(0);
      }
    }
#ifdef CA_DEBUG
    nRoads += mPrimaryVertexContext->getRoads().size();
    std::cout << "+++ Roads with " << iLevel + 2 << " clusters: " << nRoads << " / " << mPrimaryVertexContext->getRoads().size() << std::endl;
#endif
  }
}

void Tracker::findTracks(const ROframe& event)
{
  mTracks.reserve(mTracks.capacity() + mPrimaryVertexContext->getRoads().size());
  std::vector<TrackITSExt> tracks;
  tracks.reserve(mPrimaryVertexContext->getRoads().size());
#ifdef CA_DEBUG
  std::array<int, 4> roadCounters{ 0, 0, 0, 0 };
  std::array<int, 4> fitCounters{ 0, 0, 0, 0 };
  std::array<int, 4> backpropagatedCounters{ 0, 0, 0, 0 };
  std::array<int, 4> refitCounters{ 0, 0, 0, 0 };
  std::array<int, 4> nonsharingCounters{ 0, 0, 0, 0 };
#endif

  for (auto& road : mPrimaryVertexContext->getRoads()) {
    std::array<int, 7> clusters{ constants::its::UnusedIndex, constants::its::UnusedIndex, constants::its::UnusedIndex, constants::its::UnusedIndex, constants::its::UnusedIndex, constants::its::UnusedIndex, constants::its::UnusedIndex };
    int lastCellLevel = constants::its::UnusedIndex;
    CA_DEBUGGER(int nClusters = 2);

    for (int iCell{ 0 }; iCell < constants::its::CellsPerRoad; ++iCell) {
      const int cellIndex = road[iCell];
      if (cellIndex == constants::its::UnusedIndex) {
        continue;
      } else {
        clusters[iCell] = mPrimaryVertexContext->getCells()[iCell][cellIndex].getFirstClusterIndex();
        clusters[iCell + 1] = mPrimaryVertexContext->getCells()[iCell][cellIndex].getSecondClusterIndex();
        clusters[iCell + 2] = mPrimaryVertexContext->getCells()[iCell][cellIndex].getThirdClusterIndex();
        assert(clusters[iCell] != constants::its::UnusedIndex &&
               clusters[iCell + 1] != constants::its::UnusedIndex &&
               clusters[iCell + 2] != constants::its::UnusedIndex);
        lastCellLevel = iCell;
        CA_DEBUGGER(nClusters++);
      }
    }

    assert(nClusters >= mTrkParams[0].MinTrackLength);
    CA_DEBUGGER(roadCounters[nClusters - 4]++);

    if (lastCellLevel == constants::its::UnusedIndex)
      continue;

    /// From primary vertex context index to event index (== the one used as input of the tracking code)
    for (int iC{ 0 }; iC < clusters.size(); iC++) {
      if (clusters[iC] != constants::its::UnusedIndex) {
        clusters[iC] = mPrimaryVertexContext->getClusters()[iC][clusters[iC]].clusterId;
      }
    }
    /// Track seed preparation. Clusters are numbered progressively from the outermost to the innermost.
    const auto& cluster1_glo = event.getClustersOnLayer(lastCellLevel + 2).at(clusters[lastCellLevel + 2]);
    const auto& cluster2_glo = event.getClustersOnLayer(lastCellLevel + 1).at(clusters[lastCellLevel + 1]);
    const auto& cluster3_glo = event.getClustersOnLayer(lastCellLevel).at(clusters[lastCellLevel]);

    const auto& cluster3_tf = event.getTrackingFrameInfoOnLayer(lastCellLevel).at(clusters[lastCellLevel]);

    /// FIXME!
    TrackITSExt temporaryTrack{ buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_glo, cluster3_tf) };
    for (size_t iC = 0; iC < clusters.size(); ++iC) {
      temporaryTrack.setExternalClusterIndex(iC, clusters[iC], clusters[iC] != constants::its::UnusedIndex);
    }
    bool fitSuccess = fitTrack(event, temporaryTrack, constants::its::LayersNumber - 4, -1, -1);
    if (!fitSuccess)
      continue;
    CA_DEBUGGER(fitCounters[nClusters - 4]++);
    temporaryTrack.resetCovariance();
    fitSuccess = fitTrack(event, temporaryTrack, 0, constants::its::LayersNumber, 1);
    if (!fitSuccess)
      continue;
    CA_DEBUGGER(backpropagatedCounters[nClusters - 4]++);
    temporaryTrack.getParamOut() = temporaryTrack;
    temporaryTrack.resetCovariance();
    fitSuccess = fitTrack(event, temporaryTrack, constants::its::LayersNumber - 1, -1, -1);
    if (!fitSuccess)
      continue;
    CA_DEBUGGER(refitCounters[nClusters - 4]++);
    temporaryTrack.setROFrame(mROFrame);
    tracks.emplace_back(temporaryTrack);
    assert(nClusters == temporaryTrack.getNumberOfClusters());
  }
  //mTraits->refitTracks(event.getTrackingFrameInfo(), tracks);

  std::sort(tracks.begin(), tracks.end(),
            [](TrackITSExt& track1, TrackITSExt& track2) { return track1.isBetter(track2, 1.e6f); });

#ifdef CA_DEBUG
  std::array<int, 26> sharingMatrix{ 0 };
  int prevNclusters = 7;
  auto cumulativeIndex = [](int ncl) -> int {
    constexpr int idx[5] = { 0, 5, 11, 18, 26 };
    return idx[ncl - 4];
  };
  std::array<int, 4> xcheckCounters{ 0 };
#endif

  for (auto& track : tracks) {
    CA_DEBUGGER(int nClusters = 0);
    int nShared = 0;
    for (int iLayer{ 0 }; iLayer < constants::its::LayersNumber; ++iLayer) {
      if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
        continue;
      }
      nShared += int(mPrimaryVertexContext->isClusterUsed(iLayer, track.getClusterIndex(iLayer)));
      CA_DEBUGGER(nClusters++);
    }

#ifdef CA_DEBUG
    assert(nClusters == track.getNumberOfClusters());
    xcheckCounters[nClusters - 4]++;
    assert(nShared <= nClusters);
    sharingMatrix[cumulativeIndex(nClusters) + nShared]++;
#endif

    if (nShared > mTrkParams[0].ClusterSharing) {
      continue;
    }

#ifdef CA_DEBUG
    nonsharingCounters[nClusters - 4]++;
    assert(nClusters <= prevNclusters);
    prevNclusters = nClusters;
#endif

    for (int iLayer{ 0 }; iLayer < constants::its::LayersNumber; ++iLayer) {
      if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
        continue;
      }
      mPrimaryVertexContext->markUsedCluster(iLayer, track.getClusterIndex(iLayer));
    }
    mTracks.emplace_back(track);
  }

#ifdef CA_DEBUG
  std::cout << "+++ Found candidates with 4, 5, 6 and 7 clusters:\t";
  for (int count : roadCounters)
    std::cout << count << "\t";
  std::cout << std::endl;

  std::cout << "+++ Fitted candidates with 4, 5, 6 and 7 clusters:\t";
  for (int count : fitCounters)
    std::cout << count << "\t";
  std::cout << std::endl;

  std::cout << "+++ Backprop candidates with 4, 5, 6 and 7 clusters:\t";
  for (int count : backpropagatedCounters)
    std::cout << count << "\t";
  std::cout << std::endl;

  std::cout << "+++ Refitted candidates with 4, 5, 6 and 7 clusters:\t";
  for (int count : refitCounters)
    std::cout << count << "\t";
  std::cout << std::endl;

  std::cout << "+++ Cross check counters for 4, 5, 6 and 7 clusters:\t";
  for (size_t iCount = 0; iCount < refitCounters.size(); ++iCount) {
    std::cout << xcheckCounters[iCount] << "\t";
    //assert(refitCounters[iCount] == xcheckCounters[iCount]);
  }
  std::cout << std::endl;

  std::cout << "+++ Nonsharing candidates with 4, 5, 6 and 7 clusters:\t";
  for (int count : nonsharingCounters)
    std::cout << count << "\t";
  std::cout << std::endl;

  std::cout << "+++ Sharing matrix:\n";
  for (int iCl = 4; iCl <= 7; ++iCl) {
    std::cout << "+++ ";
    for (int iSh = cumulativeIndex(iCl); iSh < cumulativeIndex(iCl + 1); ++iSh) {
      std::cout << sharingMatrix[iSh] << "\t";
    }
    std::cout << std::endl;
  }
#endif
}

bool Tracker::fitTrack(const ROframe& event, TrackITSExt& track, int start, int end, int step)
{
  track.setChi2(0);
  for (int iLayer{ start }; iLayer != end; iLayer += step) {
    if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
      continue;
    }
    const TrackingFrameInfo& trackingHit = event.getTrackingFrameInfoOnLayer(iLayer).at(track.getClusterIndex(iLayer));

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

void Tracker::traverseCellsTree(const int currentCellId, const int currentLayerId)
{
  Cell& currentCell{ mPrimaryVertexContext->getCells()[currentLayerId][currentCellId] };
  const int currentCellLevel = currentCell.getLevel();

  mPrimaryVertexContext->getRoads().back().addCell(currentLayerId, currentCellId);

  if (currentLayerId > 0) {

    const int cellNeighboursNum{ static_cast<int>(
      mPrimaryVertexContext->getCellsNeighbours()[currentLayerId - 1][currentCellId].size()) };
    bool isFirstValidNeighbour = true;

    for (int iNeighbourCell{ 0 }; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

      const int neighbourCellId =
        mPrimaryVertexContext->getCellsNeighbours()[currentLayerId - 1][currentCellId][iNeighbourCell];
      const Cell& neighbourCell = mPrimaryVertexContext->getCells()[currentLayerId - 1][neighbourCellId];

      if (currentCellLevel - 1 != neighbourCell.getLevel()) {
        continue;
      }

      if (isFirstValidNeighbour) {
        isFirstValidNeighbour = false;
      } else {
        mPrimaryVertexContext->getRoads().push_back(mPrimaryVertexContext->getRoads().back());
      }

      traverseCellsTree(neighbourCellId, currentLayerId - 1);
    }
  }

  // TODO: crosscheck for short track iterations
  // currentCell.setLevel(0);
}

void Tracker::computeRoadsMClabels(const ROframe& event)
{
  /// Moore's Voting Algorithm
  if (!event.hasMCinformation()) {
    return;
  }

  int roadsNum{ static_cast<int>(mPrimaryVertexContext->getRoads().size()) };

  for (int iRoad{ 0 }; iRoad < roadsNum; ++iRoad) {

    Road& currentRoad{ mPrimaryVertexContext->getRoads()[iRoad] };
    int maxOccurrencesValue{ constants::its::UnusedIndex };
    int count{ 0 };
    bool isFakeRoad{ false };
    bool isFirstRoadCell{ true };

    for (int iCell{ 0 }; iCell < constants::its::CellsPerRoad; ++iCell) {
      const int currentCellIndex{ currentRoad[iCell] };

      if (currentCellIndex == constants::its::UnusedIndex) {
        if (isFirstRoadCell) {
          continue;
        } else {
          break;
        }
      }

      const Cell& currentCell{ mPrimaryVertexContext->getCells()[iCell][currentCellIndex] };

      if (isFirstRoadCell) {

        const int cl0index{ mPrimaryVertexContext->getClusters()[iCell][currentCell.getFirstClusterIndex()].clusterId };
        auto& cl0labs{ event.getClusterLabels(iCell, cl0index) };
        maxOccurrencesValue = cl0labs.getTrackID();
        count = 1;

        const int cl1index{
          mPrimaryVertexContext->getClusters()[iCell + 1][currentCell.getSecondClusterIndex()].clusterId
        };
        auto& cl1labs{ event.getClusterLabels(iCell + 1, cl1index) };
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
        mPrimaryVertexContext->getClusters()[iCell + 2][currentCell.getThirdClusterIndex()].clusterId
      };
      auto& cl2labs{ event.getClusterLabels(iCell + 2, cl2index) };
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

void Tracker::computeTracksMClabels(const ROframe& event)
{
  /// Moore's Voting Algorithm
  if (!event.hasMCinformation()) {
    return;
  }

  int tracksNum{ static_cast<int>(mTracks.size()) };

  for (auto& track : mTracks) {

    MCCompLabel maxOccurrencesValue{ constants::its::UnusedIndex, constants::its::UnusedIndex,
                                     constants::its::UnusedIndex };
    int count{ 0 };
    bool isFakeTrack{ false };

    for (int iCluster = 0; iCluster < TrackITSExt::MaxClusters; ++iCluster) {
      const int index = track.getClusterIndex(iCluster);
      if (index == constants::its::UnusedIndex) {
        continue;
      }

      const MCCompLabel& currentLabel = event.getClusterLabels(iCluster, index);
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

      track.setExternalClusterIndex(iCluster, event.getClusterExternalIndex(iCluster, index));
    }

    if (isFakeTrack)
      maxOccurrencesValue.set(-maxOccurrencesValue.getTrackID(), maxOccurrencesValue.getEventID(),
                              maxOccurrencesValue.getSourceID());
    mTrackLabels.addElement(mTrackLabels.getIndexedSize(), maxOccurrencesValue);
  }
}

/// Clusters are given from outside inward (cluster1 is the outermost). The innermost cluster is given in the tracking
/// frame coordinates
/// whereas the others are referred to the global frame. This function is almost a clone of CookSeed, adapted to return
/// a TrackParCov
track::TrackParCov Tracker::buildTrackSeed(const Cluster& cluster1, const Cluster& cluster2,
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
  const float cy = (MathUtils::computeCurvature(x1, y1, x2, y2 + constants::its::Resolution, x3, y3) - crv) /
                   (constants::its::Resolution * getBz() * o2::constants::math::B2C) *
                   20.f; // FIXME: MS contribution to the cov[14] (*20 added)
  constexpr float s2 = constants::its::Resolution * constants::its::Resolution;

  return track::TrackParCov(tf3.xTrackingFrame, tf3.alphaTrackingFrame,
                            { y3, z3, crv * (x3 - x0), 0.5f * (tgl12 + tgl23),
                              std::abs(getBz()) < o2::constants::math::Almost0 ? o2::constants::math::Almost0
                                                                           : crv / (getBz() * o2::constants::math::B2C) },
                            { s2, 0.f, s2, s2 * fy, 0.f, s2 * fy * fy, 0.f, s2 * tz, 0.f, s2 * tz * tz, s2 * cy, 0.f,
                              s2 * fy * cy, 0.f, s2 * cy * cy });
}

} // namespace its
} // namespace o2
