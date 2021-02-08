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

#include "ITStracking/Cell.h"
#include "ITStracking/Constants.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/TrackerTraitsCPU.h"
#include "ITStracking/TrackingConfigParam.h"

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

constexpr std::array<double, 3> getInverseSymm2D(const std::array<double, 3>& mat)
{
  const double det = mat[0] * mat[2] - mat[1] * mat[1];
  return std::array<double, 3>{mat[2] / det, -mat[1] / det, mat[0] / det};
}

using constants::its::UnusedIndex;
const MCCompLabel unusedMCLabel = {UnusedIndex, UnusedIndex,
                                   UnusedIndex, false};

Tracker::Tracker(o2::its::TrackerTraits* traits)
{
  /// Initialise standard configuration with 1 iteration
  mTrkParams.resize(1);
  mMemParams.resize(1);
  mTraits = traits;
  mPrimaryVertexContext = mTraits->getPrimaryVertexContext();
#ifdef CA_DEBUG
  mDebugger = new StandaloneDebugger("dbg_ITSTrackerCPU.root");
#endif
}
#ifdef CA_DEBUG
Tracker::~Tracker()
{
  delete mDebugger;
}
#else
Tracker::~Tracker() = default;
#endif

void Tracker::clustersToTracks(const ROframe& event, std::ostream& timeBenchmarkOutputStream)
{
  const int verticesNum = event.getPrimaryVerticesNum();
  mTracks.clear();
  mTrackLabels.clear();

  for (int iVertex = 0; iVertex < verticesNum; ++iVertex) {

    float total{0.f};

    for (int iteration = 0; iteration < mTrkParams.size(); ++iteration) {

      int numCls = 0;
      for (unsigned int iLayer{0}; iLayer < mTrkParams[iteration].NLayers; ++iLayer) {
        numCls += event.getClusters()[iLayer].size();
      }
      if (numCls < mTrkParams[iteration].MinTrackLength) {
        continue;
      }

      mTraits->UpdateTrackingParameters(mTrkParams[iteration]);
      /// Ugly hack -> Unifiy float3 definition in CPU and CUDA/HIP code
      int pass = iteration + iVertex; /// Do not reinitialise the context if we analyse pile-up events
      std::array<float, 3> pV = {event.getPrimaryVertex(iVertex).x, event.getPrimaryVertex(iVertex).y, event.getPrimaryVertex(iVertex).z};
      total += evaluateTask(&Tracker::initialisePrimaryVertexContext, "Context initialisation",
                            timeBenchmarkOutputStream, mMemParams[iteration], mTrkParams[iteration], event.getClusters(), pV, pass);
      total += evaluateTask(&Tracker::computeTracklets, "Tracklet finding", timeBenchmarkOutputStream);
      total += evaluateTask(&Tracker::computeCells, "Cell finding", timeBenchmarkOutputStream);
      total += evaluateTask(&Tracker::findCellsNeighbours, "Neighbour finding", timeBenchmarkOutputStream, iteration);
      total += evaluateTask(&Tracker::findRoads, "Road finding", timeBenchmarkOutputStream, iteration);
      total += evaluateTask(&Tracker::findTracks, "Track finding", timeBenchmarkOutputStream, event);
    }
    if (constants::DoTimeBenchmarks && fair::Logger::Logging(fair::Severity::info)) {
      timeBenchmarkOutputStream << std::setw(2) << " - "
                                << "Vertex processing completed in: " << total << "ms" << std::endl;
    }
  }
  if (event.hasMCinformation()) {
    computeTracksMClabels(event);
  } else {
    rectifyClusterIndices(event);
  }
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
  for (int iLayer{0}; iLayer < mTrkParams[iteration].CellsPerRoad() - 1; ++iLayer) {

    if (mPrimaryVertexContext->getCells()[iLayer + 1].empty() ||
        mPrimaryVertexContext->getCellsLookupTable()[iLayer].empty()) {
      continue;
    }

    int layerCellsNum{static_cast<int>(mPrimaryVertexContext->getCells()[iLayer].size())};
    const int nextLayerCellsNum{static_cast<int>(mPrimaryVertexContext->getCells()[iLayer + 1].size())};
    mPrimaryVertexContext->getCellsNeighbours()[iLayer].resize(nextLayerCellsNum);

    for (int iCell{0}; iCell < layerCellsNum; ++iCell) {

      const Cell& currentCell{mPrimaryVertexContext->getCells()[iLayer][iCell]};
      const int nextLayerTrackletIndex{currentCell.getSecondTrackletIndex()};
      const int nextLayerFirstCellIndex{mPrimaryVertexContext->getCellsLookupTable()[iLayer][nextLayerTrackletIndex]};
      if (nextLayerFirstCellIndex != constants::its::UnusedIndex &&
          mPrimaryVertexContext->getCells()[iLayer + 1][nextLayerFirstCellIndex].getFirstTrackletIndex() ==
            nextLayerTrackletIndex) {

        for (int iNextLayerCell{nextLayerFirstCellIndex}; iNextLayerCell < nextLayerCellsNum; ++iNextLayerCell) {

          Cell& nextCell{mPrimaryVertexContext->getCells()[iLayer + 1][iNextLayerCell]};
          if (nextCell.getFirstTrackletIndex() != nextLayerTrackletIndex) {
            break;
          }

          const float3 currentCellNormalVector{currentCell.getNormalVectorCoordinates()};
          const float3 nextCellNormalVector{nextCell.getNormalVectorCoordinates()};
          const float3 normalVectorsDeltaVector{currentCellNormalVector.x - nextCellNormalVector.x,
                                                currentCellNormalVector.y - nextCellNormalVector.y,
                                                currentCellNormalVector.z - nextCellNormalVector.z};

          const float deltaNormalVectorsModulus{(normalVectorsDeltaVector.x * normalVectorsDeltaVector.x) +
                                                (normalVectorsDeltaVector.y * normalVectorsDeltaVector.y) +
                                                (normalVectorsDeltaVector.z * normalVectorsDeltaVector.z)};
          const float deltaCurvature{std::abs(currentCell.getCurvature() - nextCell.getCurvature())};

          if (deltaNormalVectorsModulus < mTrkParams[iteration].NeighbourMaxDeltaN[iLayer] &&
              deltaCurvature < mTrkParams[iteration].NeighbourMaxDeltaCurvature[iLayer]) {

            mPrimaryVertexContext->getCellsNeighbours()[iLayer][iNextLayerCell].push_back(iCell);

            const int currentCellLevel{currentCell.getLevel()};

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
  for (int iLevel{mTrkParams[iteration].CellsPerRoad()}; iLevel >= mTrkParams[iteration].CellMinimumLevel(); --iLevel) {
    CA_DEBUGGER(int nRoads = -mPrimaryVertexContext->getRoads().size());
    const int minimumLevel{iLevel - 1};

    for (int iLayer{mTrkParams[iteration].CellsPerRoad() - 1}; iLayer >= minimumLevel; --iLayer) {

      const int levelCellsNum{static_cast<int>(mPrimaryVertexContext->getCells()[iLayer].size())};

      for (int iCell{0}; iCell < levelCellsNum; ++iCell) {

        Cell& currentCell{mPrimaryVertexContext->getCells()[iLayer][iCell]};

        if (currentCell.getLevel() != iLevel) {
          continue;
        }

        mPrimaryVertexContext->getRoads().emplace_back(iLayer, iCell);

        /// For 3 clusters roads (useful for cascades and hypertriton) we just store the single cell
        /// and we do not do the candidate tree traversal
        if (iLevel == 1) {
          continue;
        }

        const int cellNeighboursNum{static_cast<int>(
          mPrimaryVertexContext->getCellsNeighbours()[iLayer - 1][iCell].size())};
        bool isFirstValidNeighbour = true;

        for (int iNeighbourCell{0}; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

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
  std::vector<int> roadCounters(mTrkParams[0].NLayers - 3, 0);
  std::vector<int> fitCounters(mTrkParams[0].NLayers - 3, 0);
  std::vector<int> backpropagatedCounters(mTrkParams[0].NLayers - 3, 0);
  std::vector<int> refitCounters(mTrkParams[0].NLayers - 3, 0);
  std::vector<int> nonsharingCounters(mTrkParams[0].NLayers - 3, 0);
#endif

  for (auto& road : mPrimaryVertexContext->getRoads()) {
    std::vector<int> clusters(mTrkParams[0].NLayers, constants::its::UnusedIndex);
    int lastCellLevel = constants::its::UnusedIndex;
    CA_DEBUGGER(int nClusters = 2);

    for (int iCell{0}; iCell < mTrkParams[0].CellsPerRoad(); ++iCell) {
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

    CA_DEBUGGER(assert(nClusters >= mTrkParams[0].MinTrackLength));
    CA_DEBUGGER(roadCounters[nClusters - 4]++);

    if (lastCellLevel == constants::its::UnusedIndex) {
      continue;
    }

    /// From primary vertex context index to event index (== the one used as input of the tracking code)
    for (int iC{0}; iC < clusters.size(); iC++) {
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
    TrackITSExt temporaryTrack{buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_glo, cluster3_tf)};
    for (size_t iC = 0; iC < clusters.size(); ++iC) {
      temporaryTrack.setExternalClusterIndex(iC, clusters[iC], clusters[iC] != constants::its::UnusedIndex);
    }
    bool fitSuccess = fitTrack(event, temporaryTrack, mTrkParams[0].NLayers - 4, -1, -1);
    if (!fitSuccess) {
      continue;
    }
    CA_DEBUGGER(fitCounters[nClusters - 4]++);
    temporaryTrack.resetCovariance();
    fitSuccess = fitTrack(event, temporaryTrack, 0, mTrkParams[0].NLayers, 1, mTrkParams[0].FitIterationMaxChi2[0]);
    if (!fitSuccess) {
      continue;
    }
    CA_DEBUGGER(backpropagatedCounters[nClusters - 4]++);
    temporaryTrack.getParamOut() = temporaryTrack;
    temporaryTrack.resetCovariance();
    fitSuccess = fitTrack(event, temporaryTrack, mTrkParams[0].NLayers - 1, -1, -1, mTrkParams[0].FitIterationMaxChi2[1]);
#ifdef CA_DEBUG
    mDebugger->dumpTrackToBranchWithInfo("testBranch", temporaryTrack, event, mPrimaryVertexContext, true);
#endif
    if (!fitSuccess) {
      continue;
    }
    CA_DEBUGGER(refitCounters[nClusters - 4]++);
    tracks.emplace_back(temporaryTrack);
    CA_DEBUGGER(assert(nClusters == temporaryTrack.getNumberOfClusters()));
  }
  //mTraits->refitTracks(event.getTrackingFrameInfo(), tracks);

  if (mUseSmoother)
    smoothTracks(event, tracks);

  std::sort(tracks.begin(), tracks.end(),
            [](TrackITSExt& track1, TrackITSExt& track2) { return track1.isBetter(track2, 1.e6f); });

#ifdef CA_DEBUG
  // std::array<int, 26> sharingMatrix{0};
  // int prevNclusters = 7;
  // auto cumulativeIndex = [](int ncl) -> int {
  //   constexpr int idx[5] = {0, 5, 11, 18, 26};
  //   return idx[ncl - 4];
  // };
  // std::array<int, 4> xcheckCounters{0};
#endif

  for (auto& track : tracks) {
    CA_DEBUGGER(int nClusters = 0);
    int nShared = 0;
    for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
      if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
        continue;
      }
      nShared += int(mPrimaryVertexContext->isClusterUsed(iLayer, track.getClusterIndex(iLayer)));
      CA_DEBUGGER(nClusters++);
    }

    // #ifdef CA_DEBUG
    //     assert(nClusters == track.getNumberOfClusters());
    //     xcheckCounters[nClusters - 4]++;
    //     assert(nShared <= nClusters);
    //     sharingMatrix[cumulativeIndex(nClusters) + nShared]++;
    // #endif

    if (nShared > mTrkParams[0].ClusterSharing) {
      continue;
    }

    // #ifdef CA_DEBUG
    //     nonsharingCounters[nClusters - 4]++;
    //     assert(nClusters <= prevNclusters);
    //     prevNclusters = nClusters;
    // #endif

    for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
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

  // std::cout << "+++ Cross check counters for 4, 5, 6 and 7 clusters:\t";
  // for (size_t iCount = 0; iCount < refitCounters.size(); ++iCount) {
  //   std::cout << xcheckCounters[iCount] << "\t";
  //   //assert(refitCounters[iCount] == xcheckCounters[iCount]);
  // }
  // std::cout << std::endl;

  // std::cout << "+++ Nonsharing candidates with 4, 5, 6 and 7 clusters:\t";
  // for (int count : nonsharingCounters)
  //   std::cout << count << "\t";
  // std::cout << std::endl;

  // std::cout << "+++ Sharing matrix:\n";
  // for (int iCl = 4; iCl <= 7; ++iCl) {
  //   std::cout << "+++ ";
  //   for (int iSh = cumulativeIndex(iCl); iSh < cumulativeIndex(iCl + 1); ++iSh) {
  //     std::cout << sharingMatrix[iSh] << "\t";
  //   }
  //   std::cout << std::endl;
  // }
#endif
}

bool Tracker::fitTrack(const ROframe& event, TrackITSExt& track, int start, int end, int step, const float chi2cut)
{
  track.setChi2(0);
  for (int iLayer{start}; iLayer != end; iLayer += step) {
    if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
      continue;
    }
    const TrackingFrameInfo& trackingHit = event.getTrackingFrameInfoOnLayer(iLayer).at(track.getClusterIndex(iLayer));

    if (!track.rotate(trackingHit.alphaTrackingFrame)) {
      return false;
    }

    if (!track.propagateTo(trackingHit.xTrackingFrame, getBz())) {
      return false;
    }
    auto predChi2{track.getPredictedChi2(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
    if (predChi2 > chi2cut) {
      return false;
    }
    track.setChi2(track.getChi2() + predChi2);
    if (!track.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
      return false;
    }

    float xx0 = ((iLayer > 2) ? 0.008f : 0.003f); // Rough layer thickness
    float radiationLength = 9.36f;                // Radiation length of Si [cm]
    float density = 2.33f;                        // Density of Si [g/cm^3]
    float distance = xx0;                         // Default thickness

    if (mMatLayerCylSet) {
      if ((iLayer + step) != end) {
        const auto cl_0 = mPrimaryVertexContext->getClusters()[iLayer][track.getClusterIndex(iLayer)];
        const auto cl_1 = mPrimaryVertexContext->getClusters()[iLayer + step][track.getClusterIndex(iLayer + step)];

        auto matbud = mMatLayerCylSet->getMatBudget(cl_0.xCoordinate, cl_0.yCoordinate, cl_0.zCoordinate, cl_1.xCoordinate, cl_1.yCoordinate, cl_1.zCoordinate);
        xx0 = matbud.meanX2X0;
        density = matbud.meanRho;
        distance = matbud.length;
      }
    }
    // The correctForMaterial should be called with anglecorr==true if the material budget is the "mean budget in vertical direction" and with false if the the estimated budget already accounts for the track inclination.
    // Here using !mMatLayerCylSet as its presence triggers update of parameters

    if (!track.correctForMaterial(xx0, ((start < end) ? -1. : 1.) * distance * density, !mMatLayerCylSet)) {
      return false;
    }
  }
  return true;
}

void Tracker::traverseCellsTree(const int currentCellId, const int currentLayerId)
{
  Cell& currentCell{mPrimaryVertexContext->getCells()[currentLayerId][currentCellId]};
  const int currentCellLevel = currentCell.getLevel();

  mPrimaryVertexContext->getRoads().back().addCell(currentLayerId, currentCellId);

  if (currentLayerId > 0 && currentCellLevel > 1) {
    const int cellNeighboursNum{static_cast<int>(
      mPrimaryVertexContext->getCellsNeighbours()[currentLayerId - 1][currentCellId].size())};
    bool isFirstValidNeighbour = true;

    for (int iNeighbourCell{0}; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

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

  mPrimaryVertexContext->initialiseRoadLabels();

  int roadsNum{static_cast<int>(mPrimaryVertexContext->getRoads().size())};

  for (int iRoad{0}; iRoad < roadsNum; ++iRoad) {

    Road& currentRoad{mPrimaryVertexContext->getRoads()[iRoad]};
    MCCompLabel maxOccurrencesValue{constants::its::UnusedIndex, constants::its::UnusedIndex,
                                    constants::its::UnusedIndex, false};
    int count{0};
    bool isFakeRoad{false};
    bool isFirstRoadCell{true};

    for (int iCell{0}; iCell < mTrkParams[0].CellsPerRoad(); ++iCell) {
      const int currentCellIndex{currentRoad[iCell]};

      if (currentCellIndex == constants::its::UnusedIndex) {
        if (isFirstRoadCell) {
          continue;
        } else {
          break;
        }
      }

      const Cell& currentCell{mPrimaryVertexContext->getCells()[iCell][currentCellIndex]};

      if (isFirstRoadCell) {

        const int cl0index{mPrimaryVertexContext->getClusters()[iCell][currentCell.getFirstClusterIndex()].clusterId};
        auto& cl0labs{event.getClusterLabels(iCell, cl0index)};
        maxOccurrencesValue = cl0labs;
        count = 1;

        const int cl1index{mPrimaryVertexContext->getClusters()[iCell + 1][currentCell.getSecondClusterIndex()].clusterId};
        auto& cl1labs{event.getClusterLabels(iCell + 1, cl1index)};
        const int secondMonteCarlo{cl1labs.getTrackID()};

        if (secondMonteCarlo == maxOccurrencesValue) {
          ++count;
        } else {
          maxOccurrencesValue = secondMonteCarlo;
          count = 1;
          isFakeRoad = true;
        }

        isFirstRoadCell = false;
      }

      const int cl2index{mPrimaryVertexContext->getClusters()[iCell + 2][currentCell.getThirdClusterIndex()].clusterId};
      auto& cl2labs{event.getClusterLabels(iCell + 2, cl2index)};
      const int currentMonteCarlo = {cl2labs.getTrackID()};

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

    mPrimaryVertexContext->setRoadLabel(iRoad, maxOccurrencesValue, isFakeRoad);
  }
}

void Tracker::computeTracksMClabels(const ROframe& event)
{
  /// Moore's Voting Algorithm
  if (!event.hasMCinformation()) {
    return;
  }

  int tracksNum{static_cast<int>(mTracks.size())};

  for (auto& track : mTracks) {

    MCCompLabel maxOccurrencesValue{constants::its::UnusedIndex, constants::its::UnusedIndex,
                                    constants::its::UnusedIndex, false};
    int count{0};
    bool isFakeTrack{false};

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

    if (isFakeTrack) {
      maxOccurrencesValue.setFakeFlag();
    }
    mTrackLabels.emplace_back(maxOccurrencesValue);
  }
}

void Tracker::rectifyClusterIndices(const ROframe& event)
{
  int tracksNum{static_cast<int>(mTracks.size())};
  for (auto& track : mTracks) {
    for (int iCluster = 0; iCluster < TrackITSExt::MaxClusters; ++iCluster) {
      const int index = track.getClusterIndex(iCluster);
      if (index != constants::its::UnusedIndex) {
        track.setExternalClusterIndex(iCluster, event.getClusterExternalIndex(iCluster, index));
      }
    }
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

  const float crv = math_utils::computeCurvature(x1, y1, x2, y2, x3, y3);
  const float x0 = math_utils::computeCurvatureCentreX(x1, y1, x2, y2, x3, y3);
  const float tgl12 = math_utils::computeTanDipAngle(x1, y1, x2, y2, z1, z2);
  const float tgl23 = math_utils::computeTanDipAngle(x2, y2, x3, y3, z2, z3);

  const float fy = 1. / (cluster2.rCoordinate - cluster3.rCoordinate);
  const float& tz = fy;
  const float cy = (math_utils::computeCurvature(x1, y1, x2, y2 + constants::its::Resolution, x3, y3) - crv) /
                   (constants::its::Resolution * getBz() * o2::constants::math::B2C) *
                   20.f; // FIXME: MS contribution to the cov[14] (*20 added)
  constexpr float s2 = constants::its::Resolution * constants::its::Resolution;

  return track::TrackParCov(tf3.xTrackingFrame, tf3.alphaTrackingFrame,
                            {y3, z3, crv * (x3 - x0), 0.5f * (tgl12 + tgl23),
                             std::abs(getBz()) < o2::constants::math::Almost0 ? o2::constants::math::Almost0
                                                                              : crv / (getBz() * o2::constants::math::B2C)},
                            {s2, 0.f, s2, s2 * fy, 0.f, s2 * fy * fy, 0.f, s2 * tz, 0.f, s2 * tz * tz, s2 * cy, 0.f,
                             s2 * fy * cy, 0.f, s2 * cy * cy});
}

void Tracker::getGlobalConfiguration()
{
  auto& tc = o2::its::TrackerParamConfig::Instance();

  if (tc.useMatBudLUT) {
    initMatBudLUTFromFile();
  }
  setUseSmoother(tc.useKalmanSmoother);
}

// Smoother
float Tracker::getSmoothedPredictedChi2(const o2::track::TrackParCov& outwT, // outwards track: from innermost cluster to outermost
                                        const o2::track::TrackParCov& inwT,  // inwards track: from outermost cluster to innermost
                                        const std::array<float, 2>& cls,
                                        const std::array<float, 3>& clCov)
{
  // Tracks need to be already propagated, compute only smoothed prediction
  // Symmetric covariances assumed

  if (outwT.getX() != inwT.getX()) {
    LOG(ERROR) << "Tracks need to be propagated to the same point! inwT.X=" << inwT.getX() << " outwT.X=" << outwT.getX();
  }

  std::array<double, 2> pp1 = {static_cast<double>(outwT.getY()), static_cast<double>(outwT.getZ())}; // predicted Y,Z points
  std::array<double, 2> pp2 = {static_cast<double>(inwT.getY()), static_cast<double>(inwT.getZ())};   // predicted Y,Z points

  std::array<double, 3> c1 = {static_cast<double>(outwT.getSigmaY2()),
                              static_cast<double>(outwT.getSigmaZY()),
                              static_cast<double>(outwT.getSigmaZ2())}; // Cov. track 1

  std::array<double, 3> c2 = {static_cast<double>(inwT.getSigmaY2()),
                              static_cast<double>(inwT.getSigmaZY()),
                              static_cast<double>(inwT.getSigmaZ2())}; // Cov. track 2

  std::array<double, 3> w1 = getInverseSymm2D(c1); // weight matrices
  std::array<double, 3> w2 = getInverseSymm2D(c2);

  std::array<double, 3> w1w2 = {w1[0] + w2[0], w1[1] + w2[1], w1[2] + w2[2]};
  std::array<double, 3> C = getInverseSymm2D(w1w2); // C = (W1+W2)^-1

  std::array<double, 2> w1pp1 = {w1[0] * pp1[0] + w1[1] * pp1[1], w1[1] * pp1[0] + w1[2] * pp1[1]};
  std::array<double, 2> w2pp2 = {w2[0] * pp2[0] + w2[1] * pp2[1], w2[1] * pp2[0] + w2[2] * pp2[1]};

  float Y = static_cast<float>(C[0] * (w1pp1[0] + w2pp2[0]) + C[1] * (w1pp1[1] + w2pp2[1]));
  float Z = static_cast<float>(C[1] * (w1pp1[0] + w2pp2[0]) + C[2] * (w1pp1[1] + w2pp2[1]));

  std::array<double, 2> delta = {Y - cls[0], Z - cls[1]};
  std::array<double, 3> CCp = {C[0] + clCov[0], C[1] + clCov[1], C[2] + clCov[2]};

  float chi2 = static_cast<float>(delta[0] * (CCp[0] * delta[0] + CCp[1] * delta[1]) + delta[1] * (CCp[1] * delta[0] + CCp[2] * delta[1]));
#ifdef CA_DEBUG
  LOG(INFO) << "Propagated t1_y: " << pp1[0] << " t1_z: " << pp1[1];
  LOG(INFO) << "Propagated t2_y: " << pp2[0] << " t2_z: " << pp2[1];
  LOG(INFO) << "Smoothed prediction Y: " << Y << " Z: " << Z;
  LOG(INFO) << "cov t1: 0: " << c1[0] << " 1: " << c1[1] << " 2: " << c1[2];
  LOG(INFO) << "cov t2: 0: " << c2[0] << " 1: " << c2[1] << " 2: " << c2[2];
  LOG(INFO) << "cov Pr: 0: " << C[0] << " 1: " << C[1] << " 2: " << C[2];
  LOG(INFO) << "chi2: " << chi2;
  LOG(INFO) << "";
#endif
  return chi2;
}

bool Tracker::initializeSmootherTracks(const ROframe& event,
                                       o2::its::TrackITSExt& outwT, // outwards track: from innermost cluster to outermost
                                       o2::its::TrackITSExt& inwT,  // inwards track: from outermost cluster to innermost
                                       const int firstLayer,
                                       const int lastLayer)
{
  float radiationLength = 9.36f; // Radiation length of Si [cm]
  float density = 2.33f;         // Density of Si [g/cm^3]
  float distance;                // Default thickness

  outwT.resetCovariance();
  outwT.setChi2(0);
  inwT.resetCovariance();
  inwT.setChi2(0);

  // Initialise tracks with their first two clusters respectively
  for (auto iLayer{0}; iLayer < 2; ++iLayer) {
    int iLayerInwardsTrack{lastLayer - iLayer};
    const TrackingFrameInfo& inwHit = event.getTrackingFrameInfoOnLayer(iLayerInwardsTrack).at(inwT.getClusterIndex(iLayerInwardsTrack));
    float xx0_inw = ((iLayerInwardsTrack > 2) ? 0.008f : 0.003f); // Rough layer thickness
    inwT.rotate(inwHit.alphaTrackingFrame);
    inwT.propagateTo(inwHit.xTrackingFrame, getBz());
    inwT.setChi2(inwT.getChi2() +
                 inwT.getPredictedChi2(inwHit.positionTrackingFrame, inwHit.covarianceTrackingFrame));
    inwT.o2::track::TrackParCov::update(inwHit.positionTrackingFrame, inwHit.covarianceTrackingFrame);

    if (mMatLayerCylSet) {
      if (iLayer) {
        const auto cl_0 = mPrimaryVertexContext->getClusters()[iLayerInwardsTrack][inwT.getClusterIndex(iLayerInwardsTrack)];
        const auto cl_1 = mPrimaryVertexContext->getClusters()[iLayerInwardsTrack - 1][inwT.getClusterIndex(iLayerInwardsTrack - 1)];

        auto matbud = mMatLayerCylSet->getMatBudget(cl_0.xCoordinate, cl_0.yCoordinate, cl_0.zCoordinate, cl_1.xCoordinate, cl_1.yCoordinate, cl_1.zCoordinate);
        xx0_inw = matbud.meanX2X0;
        density = matbud.meanRho;
        distance = matbud.length;
      }
    }

    if (!inwT.correctForMaterial(xx0_inw, distance * density, !mMatLayerCylSet)) {
      return false;
    }
    const TrackingFrameInfo& outwHit = event.getTrackingFrameInfoOnLayer(iLayer).at(outwT.getClusterIndex(iLayer));
    float xx0_out = ((iLayer > 2) ? 0.008f : 0.003f);
    outwT.rotate(outwHit.alphaTrackingFrame);
    outwT.propagateTo(outwHit.xTrackingFrame, getBz());
    outwT.setChi2(outwT.getChi2() +
                  outwT.getPredictedChi2(outwHit.positionTrackingFrame, outwHit.covarianceTrackingFrame));
    outwT.o2::track::TrackParCov::update(outwHit.positionTrackingFrame, outwHit.covarianceTrackingFrame);

    if (mMatLayerCylSet) {
      if (iLayer) {
        const auto cl_0 = mPrimaryVertexContext->getClusters()[iLayer][outwT.getClusterIndex(iLayer)];
        const auto cl_1 = mPrimaryVertexContext->getClusters()[iLayer + 1][outwT.getClusterIndex(iLayer + 1)];

        auto matbud = mMatLayerCylSet->getMatBudget(cl_0.xCoordinate, cl_0.yCoordinate, cl_0.zCoordinate, cl_1.xCoordinate, cl_1.yCoordinate, cl_1.zCoordinate);
        xx0_out = matbud.meanX2X0;
        density = matbud.meanRho;
        distance = matbud.length;
      }
    }
    if (!outwT.correctForMaterial(xx0_out, -distance * density, !mMatLayerCylSet)) {
      return false;
    }
  }

  return true;
}

bool Tracker::kalmanPropagateOutwardsTrack(const ROframe& event,
                                           o2::its::TrackITSExt& track,
                                           const int first,
                                           const int last)
{
  float radiationLength = 9.36f; // Radiation length of Si [cm]
  float density = 2.33f;         // Density of Si [g/cm^3]
  float distance;                // Default thickness
  bool status{false};

  for (auto iLevel{first}; iLevel < last; ++iLevel) {
    float xx0 = ((iLevel > 2) ? 0.008f : 0.003f); // Rough layer thickness
    distance = xx0;
    const TrackingFrameInfo& tF = event.getTrackingFrameInfoOnLayer(iLevel).at(track.getClusterIndex(iLevel));
    status = track.rotate(tF.alphaTrackingFrame);
    status &= track.propagateTo(tF.xTrackingFrame, getBz());
    track.setChi2(track.getChi2() +
                  track.getPredictedChi2(tF.positionTrackingFrame, tF.covarianceTrackingFrame));
    status &= track.o2::track::TrackParCov::update(tF.positionTrackingFrame, tF.covarianceTrackingFrame);
    if (mMatLayerCylSet) {
      if (iLevel != first || iLevel != last) {
        const auto cl_0 = mPrimaryVertexContext->getClusters()[iLevel][track.getClusterIndex(iLevel)];
        const auto cl_1 = mPrimaryVertexContext->getClusters()[iLevel + 1][track.getClusterIndex(iLevel + 1)];

        auto matbud = mMatLayerCylSet->getMatBudget(cl_0.xCoordinate, cl_0.yCoordinate, cl_0.zCoordinate, cl_1.xCoordinate, cl_1.yCoordinate, cl_1.zCoordinate);
        xx0 = matbud.meanX2X0;
        density = matbud.meanRho;
        distance = matbud.length;
      }
    }

    if (!track.correctForMaterial(xx0, -distance * density, !mMatLayerCylSet)) {
      return false;
    }
  }
  return status;
}

void Tracker::smoothTracks(const ROframe& event, std::vector<TrackITSExt>& tracks)
{
  // Operate on groups of siblings trying to shuffle their clusters to find a better chi2
  constexpr float radiationLength = 9.36f; // Radiation length of Si [cm]
  constexpr float density = 2.33f;         // Density of Si [g/cm^3]
  for (auto& indexed : mPrimaryVertexContext->getTracksIndexTable()) {
    for (auto iFirstTrack{indexed.first}; iFirstTrack < indexed.first + indexed.second; ++iFirstTrack) {
      // LOG(WARN) << "indexed.first " << indexed.first << " indexed.second " << indexed.second;
      o2::its::TrackITSExt outwardsTrack = tracks[iFirstTrack]; // outwards track: from innermost cluster to outermost
      o2::its::TrackITSExt inwardsTrack{outwardsTrack.getParamOut(),
                                        static_cast<short>(outwardsTrack.getNumberOfClusters()), -999, static_cast<std::uint32_t>(event.getROFrameId()),
                                        outwardsTrack.getParamOut(), outwardsTrack.getClusterIndexes()}; // inwards track: from outermost cluster to innermost

      initializeSmootherTracks(event, outwardsTrack, inwardsTrack, 0, 6); // Change here when less than 7 clusters

      //Iterate on all possible clusters owned by tracks in the same tree and check for best chi2
      for (auto smoothLevel{4}; smoothLevel > 1; --smoothLevel) { // inwards level, counting from outside to inside, as we are riding inwards track
        o2::its::TrackITSExt outwardsTrackCopy = outwardsTrack;
        // work with copies of the inner track to start from the same conditions at each iteration

        kalmanPropagateOutwardsTrack(event, outwardsTrackCopy, 2, smoothLevel);
        float bestChi2{o2::constants::math::VeryBig};
        for (auto iCandidate{indexed.first}; iCandidate < /*iFirstTrack + 1*/ indexed.first + indexed.second; ++iCandidate) { // shuffler

          o2::its::TrackITSExt inwardsTrackDegen = inwardsTrack;
          o2::its::TrackITSExt outwardsTrackCopyDegen = outwardsTrackCopy;
          const TrackingFrameInfo& testHit = event.getTrackingFrameInfoOnLayer(smoothLevel).at(tracks[iCandidate].getClusterIndex(smoothLevel));
          if (!inwardsTrackDegen.rotate(testHit.alphaTrackingFrame)) {
            LOG(INFO) << "Failed rotation inwards track";
            continue;
          }
          if (!inwardsTrackDegen.propagateTo(testHit.xTrackingFrame, getBz())) {
            LOG(INFO) << "Failed propagation inwards track";
            continue;
          }
          if (!outwardsTrackCopyDegen.rotate(testHit.alphaTrackingFrame)) {
            LOG(INFO) << "Failed rotation outwards track";
            continue;
          }
          if (!outwardsTrackCopyDegen.propagateTo(testHit.xTrackingFrame, getBz())) {
            LOG(INFO) << "Failed propagation outwards track";
            continue;
          }
          float localChi2 = getSmoothedPredictedChi2(outwardsTrackCopyDegen, inwardsTrackDegen, testHit.positionTrackingFrame, testHit.covarianceTrackingFrame);
          if (localChi2 < bestChi2) {
            bestChi2 = localChi2;
            outwardsTrackCopy.setExternalClusterIndex(smoothLevel, tracks[iCandidate].getClusterIndex(smoothLevel));
            outwardsTrack.setExternalClusterIndex(smoothLevel, tracks[iCandidate].getClusterIndex(smoothLevel));
            inwardsTrack.setExternalClusterIndex(smoothLevel, tracks[iCandidate].getClusterIndex(smoothLevel));
          }
          // FakeTrackInfo infoT{event, outwardsTrackCopy};
          // MCCompLabel testClusterLabel = event.getClusterLabels(smoothLevel, tracks[iCandidate].getClusterIndex// (smoothLevel));
          // bool isClusFake = testClusterLabel != infoT.mainLabel;
          // mDebugger->fillSmootherClusStats(smoothLevel, localChi2, isClusFake);
        }
        const TrackingFrameInfo& selectedHit = event.getTrackingFrameInfoOnLayer(smoothLevel).at(inwardsTrack.getClusterIndex(smoothLevel));
        if (!inwardsTrack.rotate(selectedHit.alphaTrackingFrame)) {
          LOG(INFO) << "Failed rotation inwards track out of shuffler";
          continue;
        }
        if (!inwardsTrack.propagateTo(selectedHit.xTrackingFrame, getBz())) {
          LOG(INFO) << "Failed propagation inwards track out of shuffler";
          continue;
        }
        inwardsTrack.setChi2(inwardsTrack.getChi2() +
                             inwardsTrack.getPredictedChi2(selectedHit.positionTrackingFrame, selectedHit.covarianceTrackingFrame));
        if (!inwardsTrack.o2::track::TrackParCov::update(selectedHit.positionTrackingFrame, selectedHit.covarianceTrackingFrame)) {
          LOG(INFO) << "Failed update inwards track out of shuffler";
          continue;
        }
        float xx0_inw = (smoothLevel > 3) ? 0.008f : 0.003;
        inwardsTrack.correctForMaterial(xx0_inw, xx0_inw * radiationLength * density, true); // (first < last) ? -1. : 1.)
      }
      for (int iLayer{2}; iLayer < 7; ++iLayer) { // finish propagation and kalman filter for winning track
        const TrackingFrameInfo& outwHit = event.getTrackingFrameInfoOnLayer(iLayer).at(outwardsTrack.getClusterIndex(iLayer));
        outwardsTrack.rotate(outwHit.alphaTrackingFrame);
        outwardsTrack.propagateTo(outwHit.xTrackingFrame, getBz());
        outwardsTrack.setChi2(outwardsTrack.getChi2() +
                              outwardsTrack.getPredictedChi2(outwHit.positionTrackingFrame, outwHit.covarianceTrackingFrame));
        outwardsTrack.o2::track::TrackParCov::update(outwHit.positionTrackingFrame, outwHit.covarianceTrackingFrame);
        float xx0_outw = (iLayer > 3) ? 0.008f : 0.003;
        outwardsTrack.correctForMaterial(xx0_outw, -xx0_outw * radiationLength * density, true); // (first < last) ? -1. : 1.)
      }
      if (tracks[iFirstTrack].getChi2() > outwardsTrack.getChi2()) {
        // FakeTrackInfo infoS{event, outwardsTrack};
        // FakeTrackInfo info{event, tracks[iFirstTrack]};
        // if (info.nFakeClusters == 0 && infoS.nFakeClusters > 0) {
        //   LOG(WARN) << "smoothing:";
        //   LOG(WARN) << "\tprevious: chi2=" << tracks[iFirstTrack].getChi2() << " fake=" << info.isFake << " N fake=" << info.nFakeClusters
        //             << "\n\tnew: chi2=" << outwardsTrack.getChi2() << " fake=" << infoS.isFake << " N fake=" << infoS.nFakeClusters;
        //   for (auto i{0}; i < 7; ++i) {
        //     LOG(WARN) << "\t\t" << event.getClusterLabels(i, tracks[iFirstTrack].getClusterIndex(i)) << " " << event.getClusterLabels(i, outwardsTrack.getClusterIndex(i));
        //   }
        // }
        tracks[iFirstTrack] = outwardsTrack;
      }
    }
  }
}

} // namespace its
} // namespace o2
