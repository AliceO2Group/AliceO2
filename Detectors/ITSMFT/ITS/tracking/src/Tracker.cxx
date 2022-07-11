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
/// \file Tracker.cxx
/// \brief
///

#include "ITStracking/Tracker.h"

#include "ITStracking/Cell.h"
#include "ITStracking/Constants.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/Smoother.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/TrackingConfigParam.h"

#include "ReconstructionDataFormats/Track.h"
#include <cassert>
#include <iostream>
#include <dlfcn.h>
#include <cstdlib>
#include <string>
#include <climits>

namespace o2
{
namespace its
{

Tracker::Tracker(o2::its::TrackerTraits* traits)
{
  /// Initialise standard configuration with 1 iteration
  mTrkParams.resize(1);
  mMemParams.resize(1);
  mTraits = traits;
}

Tracker::~Tracker() = default;

void Tracker::clustersToTracks(std::function<void(std::string s)> logger, std::function<void(std::string s)> error)
{
  double total{0};
  for (int iteration = 0; iteration < (int)mTrkParams.size(); ++iteration) {
    mTraits->UpdateTrackingParameters(mTrkParams[iteration]);

    total += evaluateTask(&Tracker::initialiseTimeFrame, "Timeframe initialisation",
                          logger, iteration, mMemParams[iteration], mTrkParams[iteration]);
    total += evaluateTask(&Tracker::computeTracklets, "Tracklet finding", logger);
    logger(fmt::format("\t- Number of tracklets: {}", mTimeFrame->getNumberOfTracklets()));
    if (!mTimeFrame->checkMemory(mTrkParams[iteration].MaxMemory)) {
      error("Too much memory used during trackleting, check the detector status and/or the selections.");
      break;
    }
    float trackletsPerCluster = mTimeFrame->getNumberOfClusters() > 0 ? float(mTimeFrame->getNumberOfTracklets()) / mTimeFrame->getNumberOfClusters() : 0.f;
    if (trackletsPerCluster > mTrkParams[iteration].TrackletsPerClusterLimit) {
      error(fmt::format("Too many tracklets per cluster ({}), check the detector status and/or the selections.", trackletsPerCluster));
      break;
    }

    total += evaluateTask(&Tracker::computeCells, "Cell finding", logger);
    logger(fmt::format("\t- Number of Cells: {}", mTimeFrame->getNumberOfCells()));
    if (!mTimeFrame->checkMemory(mTrkParams[iteration].MaxMemory)) {
      error("Too much memory used during cell finding, check the detector status and/or the selections.");
      break;
    }
    float cellsPerCluster = mTimeFrame->getNumberOfClusters() > 0 ? float(mTimeFrame->getNumberOfCells()) / mTimeFrame->getNumberOfClusters() : 0.f;
    if (cellsPerCluster > mTrkParams[iteration].CellsPerClusterLimit) {
      error(fmt::format("Too many cells per cluster ({}), check the detector status and/or the selections.", cellsPerCluster));
      break;
    }

    total += evaluateTask(&Tracker::findCellsNeighbours, "Neighbour finding", logger, iteration);
    total += evaluateTask(&Tracker::findRoads, "Road finding", logger, iteration);
    total += evaluateTask(&Tracker::findTracks, "Track finding", logger);
    total += evaluateTask(&Tracker::extendTracks, "Extending tracks", logger);
  }

  std::stringstream sstream;
  if (constants::DoTimeBenchmarks) {
    sstream << std::setw(2) << " - "
            << "Timeframe " << mTimeFrameCounter++ << " processing completed in: " << total << "ms";
  }
  logger(sstream.str());

  if (mTimeFrame->hasMCinformation()) {
    computeTracksMClabels();
  }
  rectifyClusterIndices();
  mNumberOfRuns++;
}

void Tracker::clustersToTracksGPU(std::function<void(std::string s)> logger)
{
  double total{0};
  for (int iteration = 0; iteration < mTrkParams.size(); ++iteration) {
    mTraits->UpdateTrackingParameters(mTrkParams[iteration]);
    total += evaluateTask(&Tracker::loadToDevice, "Device loading", logger);
    total += evaluateTask(&Tracker::computeTracklets, "Tracklet finding", logger);
    // total += evaluateTask(&Tracker::computeCells, "Cell finding", logger);
    // total += evaluateTask(&Tracker::findCellsNeighbours, "Neighbour finding", logger, iteration);
    // total += evaluateTask(&Tracker::findRoads, "Road finding", logger, iteration);
    // total += evaluateTask(&Tracker::findTracks, "Track finding", logger);
    // total += evaluateTask(&Tracker::extendTracks, "Extending tracks", logger);
  }

  std::stringstream sstream;
  if (constants::DoTimeBenchmarks) {
    sstream << std::setw(2) << " - "
            << "Timeframe " << mTimeFrameCounter++ << " GPU processing completed in: " << total << "ms";
  }
  logger(sstream.str());

  // if (mTimeFrame->hasMCinformation()) {
  //   computeTracksMClabels();
  // }
  // rectifyClusterIndices();
}

void Tracker::computeTracklets()
{
  mTraits->computeLayerTracklets();
}

void Tracker::computeCells()
{
  mTraits->computeLayerCells();
}

TimeFrame* Tracker::getTimeFrameGPU()
{
  return (TimeFrame*)mTraits->getTimeFrameGPU();
}

void Tracker::loadToDevice()
{
  mTraits->loadToDevice();
}

void Tracker::findCellsNeighbours(int& iteration)
{
#ifdef OPTIMISATION_OUTPUT
  std::ofstream off(fmt::format("cellneighs{}.txt", iteration));
#endif
  for (int iLayer{0}; iLayer < mTrkParams[iteration].CellsPerRoad() - 1; ++iLayer) {

    if (mTimeFrame->getCells()[iLayer + 1].empty() ||
        mTimeFrame->getCellsLookupTable()[iLayer].empty()) {
      continue;
    }

    int layerCellsNum{static_cast<int>(mTimeFrame->getCells()[iLayer].size())};
    const int nextLayerCellsNum{static_cast<int>(mTimeFrame->getCells()[iLayer + 1].size())};
    mTimeFrame->getCellsNeighbours()[iLayer].resize(nextLayerCellsNum);

    for (int iCell{0}; iCell < layerCellsNum; ++iCell) {

      const Cell& currentCell{mTimeFrame->getCells()[iLayer][iCell]};
      const int nextLayerTrackletIndex{currentCell.getSecondTrackletIndex()};
      const int nextLayerFirstCellIndex{mTimeFrame->getCellsLookupTable()[iLayer][nextLayerTrackletIndex]};
      const int nextLayerLastCellIndex{mTimeFrame->getCellsLookupTable()[iLayer][nextLayerTrackletIndex + 1]};
      for (int iNextCell{nextLayerFirstCellIndex}; iNextCell < nextLayerLastCellIndex; ++iNextCell) {

        Cell& nextCell{mTimeFrame->getCells()[iLayer + 1][iNextCell]};
        if (nextCell.getFirstTrackletIndex() != nextLayerTrackletIndex) {
          break;
        }

#ifdef OPTIMISATION_OUTPUT
        bool good{mTimeFrame->getCellsLabel(iLayer)[iCell] == mTimeFrame->getCellsLabel(iLayer + 1)[iNextCell]};
        float signedDelta{currentCell.getTanLambda() - nextCell.getTanLambda()};
        off << fmt::format("{}\t{:d}\t{}\t{}", iLayer, good, signedDelta, signedDelta / mTrkParams[iteration].CellDeltaTanLambdaSigma) << std::endl;
#endif
        mTimeFrame->getCellsNeighbours()[iLayer][iNextCell].push_back(iCell);

        const int currentCellLevel{currentCell.getLevel()};

        if (currentCellLevel >= nextCell.getLevel()) {
          nextCell.setLevel(currentCellLevel + 1);
        }
        // }
      }
    }
  }
}

void Tracker::findRoads(int& iteration)
{
  for (int iLevel{mTrkParams[iteration].CellsPerRoad()}; iLevel >= mTrkParams[iteration].CellMinimumLevel(); --iLevel) {
    CA_DEBUGGER(int nRoads = -mTimeFrame->getRoads().size());
    const int minimumLevel{iLevel - 1};

    for (int iLayer{mTrkParams[iteration].CellsPerRoad() - 1}; iLayer >= minimumLevel; --iLayer) {

      const int levelCellsNum{static_cast<int>(mTimeFrame->getCells()[iLayer].size())};

      for (int iCell{0}; iCell < levelCellsNum; ++iCell) {

        Cell& currentCell{mTimeFrame->getCells()[iLayer][iCell]};

        if (currentCell.getLevel() != iLevel) {
          continue;
        }

        mTimeFrame->getRoads().emplace_back(iLayer, iCell);

        /// For 3 clusters roads (useful for cascades and hypertriton) we just store the single cell
        /// and we do not do the candidate tree traversal
        if (iLevel == 1) {
          continue;
        }

        const int cellNeighboursNum{static_cast<int>(
          mTimeFrame->getCellsNeighbours()[iLayer - 1][iCell].size())};
        bool isFirstValidNeighbour = true;

        for (int iNeighbourCell{0}; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

          const int neighbourCellId = mTimeFrame->getCellsNeighbours()[iLayer - 1][iCell][iNeighbourCell];
          const Cell& neighbourCell = mTimeFrame->getCells()[iLayer - 1][neighbourCellId];

          if (iLevel - 1 != neighbourCell.getLevel()) {
            continue;
          }

          if (isFirstValidNeighbour) {

            isFirstValidNeighbour = false;

          } else {

            mTimeFrame->getRoads().emplace_back(iLayer, iCell);
          }

          traverseCellsTree(neighbourCellId, iLayer - 1);
        }

        // TODO: crosscheck for short track iterations
        // currentCell.setLevel(0);
      }
    }
#ifdef CA_DEBUG
    nRoads += mTimeFrame->getRoads().size();
    std::cout << "+++ Roads with " << iLevel + 2 << " clusters: " << nRoads << " / " << mTimeFrame->getRoads().size() << std::endl;
#endif
  }
}

void Tracker::findTracks()
{
  std::vector<TrackITSExt> tracks;
  tracks.reserve(mTimeFrame->getRoads().size());

  for (auto& road : mTimeFrame->getRoads()) {
    std::vector<int> clusters(mTrkParams[0].NLayers, constants::its::UnusedIndex);
    int lastCellLevel = constants::its::UnusedIndex;
    CA_DEBUGGER(int nClusters = 2);
    int firstTracklet{constants::its::UnusedIndex};
    std::vector<int> tracklets(mTrkParams[0].TrackletsPerRoad(), constants::its::UnusedIndex);

    for (int iCell{0}; iCell < mTrkParams[0].CellsPerRoad(); ++iCell) {
      const int cellIndex = road[iCell];
      if (cellIndex == constants::its::UnusedIndex) {
        continue;
      } else {
        if (firstTracklet == constants::its::UnusedIndex) {
          firstTracklet = iCell;
        }
        tracklets[iCell] = mTimeFrame->getCells()[iCell][cellIndex].getFirstTrackletIndex();
        tracklets[iCell + 1] = mTimeFrame->getCells()[iCell][cellIndex].getSecondTrackletIndex();
        clusters[iCell] = mTimeFrame->getCells()[iCell][cellIndex].getFirstClusterIndex();
        clusters[iCell + 1] = mTimeFrame->getCells()[iCell][cellIndex].getSecondClusterIndex();
        clusters[iCell + 2] = mTimeFrame->getCells()[iCell][cellIndex].getThirdClusterIndex();
        assert(clusters[iCell] != constants::its::UnusedIndex &&
               clusters[iCell + 1] != constants::its::UnusedIndex &&
               clusters[iCell + 2] != constants::its::UnusedIndex);
        lastCellLevel = iCell;
        CA_DEBUGGER(nClusters++);
      }
    }

    CA_DEBUGGER(assert(nClusters >= mTrkParams[0].MinTrackLength));
    int count{1};
    unsigned short rof{mTimeFrame->getTracklets()[firstTracklet][tracklets[firstTracklet]].rof[0]};
    for (int iT = firstTracklet; iT < 6; ++iT) {
      if (tracklets[iT] == constants::its::UnusedIndex) {
        continue;
      }
      if (rof == mTimeFrame->getTracklets()[iT][tracklets[iT]].rof[1]) {
        count++;
      } else {
        if (count == 1) {
          rof = mTimeFrame->getTracklets()[iT][tracklets[iT]].rof[1];
        } else {
          count--;
        }
      }
    }

    CA_DEBUGGER(assert(nClusters >= mTrkParams[0].MinTrackLength));
    CA_DEBUGGER(roadCounters[nClusters - 4]++);

    if (lastCellLevel == constants::its::UnusedIndex) {
      continue;
    }

    /// From primary vertex context index to event index (== the one used as input of the tracking code)
    for (size_t iC{0}; iC < clusters.size(); iC++) {
      if (clusters[iC] != constants::its::UnusedIndex) {
        clusters[iC] = mTimeFrame->getClusters()[iC][clusters[iC]].clusterId;
      }
    }

    /// Track seed preparation. Clusters are numbered progressively from the outermost to the innermost.
    const auto& cluster1_glo = mTimeFrame->getUnsortedClusters()[lastCellLevel + 2].at(clusters[lastCellLevel + 2]);
    const auto& cluster2_glo = mTimeFrame->getUnsortedClusters()[lastCellLevel + 1].at(clusters[lastCellLevel + 1]);
    const auto& cluster3_glo = mTimeFrame->getUnsortedClusters()[lastCellLevel].at(clusters[lastCellLevel]);

    const auto& cluster3_tf = mTimeFrame->getTrackingFrameInfoOnLayer(lastCellLevel).at(clusters[lastCellLevel]);

    /// FIXME!
    TrackITSExt temporaryTrack{buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_glo, cluster3_tf, mTimeFrame->getPositionResolution(lastCellLevel))};
    for (size_t iC = 0; iC < clusters.size(); ++iC) {
      temporaryTrack.setExternalClusterIndex(iC, clusters[iC], clusters[iC] != constants::its::UnusedIndex);
    }
    bool fitSuccess = fitTrack(temporaryTrack, mTrkParams[0].NLayers - 4, -1, -1);
    if (!fitSuccess) {
      continue;
    }
    CA_DEBUGGER(fitCounters[nClusters - 4]++);
    temporaryTrack.resetCovariance();
    fitSuccess = fitTrack(temporaryTrack, 0, mTrkParams[0].NLayers, 1, mTrkParams[0].FitIterationMaxChi2[0]);
    if (!fitSuccess) {
      continue;
    }
    CA_DEBUGGER(backpropagatedCounters[nClusters - 4]++);
    temporaryTrack.getParamOut() = temporaryTrack;
    temporaryTrack.resetCovariance();
    fitSuccess = fitTrack(temporaryTrack, mTrkParams[0].NLayers - 1, -1, -1, mTrkParams[0].FitIterationMaxChi2[1], 50.);
    if (!fitSuccess) {
      continue;
    }
    // temporaryTrack.setROFrame(rof);
    tracks.emplace_back(temporaryTrack);
  }

  if (mApplySmoothing) {
    // Smoothing tracks
  }
  std::sort(tracks.begin(), tracks.end(),
            [](TrackITSExt& track1, TrackITSExt& track2) { return track1.isBetter(track2, 1.e6f); });

  for (auto& track : tracks) {
    int nShared = 0;
    for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
      if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
        continue;
      }
      nShared += int(mTimeFrame->isClusterUsed(iLayer, track.getClusterIndex(iLayer)));
    }

    if (nShared > mTrkParams[0].ClusterSharing) {
      continue;
    }

    std::array<int, 3> rofs{INT_MAX, INT_MAX, INT_MAX};
    for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
      if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
        continue;
      }
      mTimeFrame->markUsedCluster(iLayer, track.getClusterIndex(iLayer));
      int currentROF = mTimeFrame->getClusterROF(iLayer, track.getClusterIndex(iLayer));
      for (int iR{0}; iR < 3; ++iR) {
        if (rofs[iR] == INT_MAX) {
          rofs[iR] = currentROF;
        }
        if (rofs[iR] == currentROF) {
          break;
        }
      }
    }
    if (rofs[2] != INT_MAX) {
      continue;
    }
    if (rofs[1] != INT_MAX) {
      track.setNextROFbit();
    }
    mTimeFrame->getTracks(std::min(rofs[0], rofs[1])).emplace_back(track);
  }
}

void Tracker::extendTracks()
{
  if (!mTrkParams.back().UseTrackFollower) {
    return;
  }
  for (int rof{0}; rof < mTimeFrame->getNrof(); ++rof) {
    for (auto& track : mTimeFrame->getTracks(rof)) {
      /// TODO: track refitting is missing!
      int ncl{track.getNClusters()};
      auto backup{track};
      bool success{false};
      if (track.getLastClusterLayer() != mTrkParams[0].NLayers - 1) {
        success = success || mTraits->trackFollowing(&track, rof, true);
      }
      if (track.getFirstClusterLayer() != 0) {
        success = success || mTraits->trackFollowing(&track, rof, false);
      }
      if (success) {
        /// We have to refit the track
        track.resetCovariance();
        bool fitSuccess = fitTrack(track, 0, mTrkParams[0].NLayers, 1, mTrkParams[0].FitIterationMaxChi2[0]);
        if (!fitSuccess) {
          track = backup;
          continue;
        }
        track.getParamOut() = track;
        track.resetCovariance();
        fitSuccess = fitTrack(track, mTrkParams[0].NLayers - 1, -1, -1, mTrkParams[0].FitIterationMaxChi2[1], 50.);
        if (!fitSuccess) {
          track = backup;
          continue;
        }
        /// Make sure that the newly attached clusters get marked as used
        for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
          if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
            continue;
          }
          mTimeFrame->markUsedCluster(iLayer, track.getClusterIndex(iLayer));
        }
      }
    }
  }
}

bool Tracker::fitTrack(TrackITSExt& track, int start, int end, int step, const float chi2cut, const float maxQoverPt)
{
  auto propInstance = o2::base::Propagator::Instance();
  track.setChi2(0);
  int nCl{0};
  for (int iLayer{start}; iLayer != end; iLayer += step) {
    if (track.getClusterIndex(iLayer) == constants::its::UnusedIndex) {
      continue;
    }
    const TrackingFrameInfo& trackingHit = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer).at(track.getClusterIndex(iLayer));

    if (!track.rotate(trackingHit.alphaTrackingFrame)) {
      return false;
    }

    if (!propInstance->propagateToX(track, trackingHit.xTrackingFrame, getBz(), o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, mCorrType)) {
      return false;
    }

    if (mCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
      float radl = 9.36f; // Radiation length of Si [cm]
      float rho = 2.33f;  // Density of Si [g/cm^3]
      if (!track.correctForMaterial(mTrkParams[0].LayerxX0[iLayer], mTrkParams[0].LayerxX0[iLayer] * radl * rho, true)) {
        continue;
      }
    }

    GPUArray<float, 3> cov{trackingHit.covarianceTrackingFrame};
    cov[0] = std::hypot(cov[0], mTrkParams[0].LayerMisalignment[iLayer]);
    cov[2] = std::hypot(cov[2], mTrkParams[0].LayerMisalignment[iLayer]);
    auto predChi2{track.getPredictedChi2(trackingHit.positionTrackingFrame, cov)};
    if (nCl >= 3 && predChi2 > chi2cut * (nCl * 2 - 5)) {
      return false;
    }
    track.setChi2(track.getChi2() + predChi2);
    if (!track.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, cov)) {
      return false;
    }
    nCl++;
  }
  return std::abs(track.getQ2Pt()) < maxQoverPt;
}

void Tracker::traverseCellsTree(const int currentCellId, const int currentLayerId)
{
  Cell& currentCell{mTimeFrame->getCells()[currentLayerId][currentCellId]};
  const int currentCellLevel = currentCell.getLevel();

  mTimeFrame->getRoads().back().addCell(currentLayerId, currentCellId);

  if (currentLayerId > 0 && currentCellLevel > 1) {
    const int cellNeighboursNum{static_cast<int>(
      mTimeFrame->getCellsNeighbours()[currentLayerId - 1][currentCellId].size())};
    bool isFirstValidNeighbour = true;

    for (int iNeighbourCell{0}; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

      const int neighbourCellId =
        mTimeFrame->getCellsNeighbours()[currentLayerId - 1][currentCellId][iNeighbourCell];
      const Cell& neighbourCell = mTimeFrame->getCells()[currentLayerId - 1][neighbourCellId];

      if (currentCellLevel - 1 != neighbourCell.getLevel()) {
        continue;
      }

      if (isFirstValidNeighbour) {
        isFirstValidNeighbour = false;
      } else {
        mTimeFrame->getRoads().push_back(mTimeFrame->getRoads().back());
      }

      traverseCellsTree(neighbourCellId, currentLayerId - 1);
    }
  }

  // TODO: crosscheck for short track iterations
  // currentCell.setLevel(0);
}

void Tracker::computeRoadsMClabels()
{
  /// Moore's Voting Algorithm
  if (!mTimeFrame->hasMCinformation()) {
    return;
  }

  mTimeFrame->initialiseRoadLabels();

  int roadsNum{static_cast<int>(mTimeFrame->getRoads().size())};

  for (int iRoad{0}; iRoad < roadsNum; ++iRoad) {

    Road& currentRoad{mTimeFrame->getRoads()[iRoad]};
    std::vector<std::pair<MCCompLabel, size_t>> occurrences;
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

      const Cell& currentCell{mTimeFrame->getCells()[iCell][currentCellIndex]};

      if (isFirstRoadCell) {

        const int cl0index{mTimeFrame->getClusters()[iCell][currentCell.getFirstClusterIndex()].clusterId};
        auto cl0labs{mTimeFrame->getClusterLabels(iCell, cl0index)};
        bool found{false};
        for (size_t iOcc{0}; iOcc < occurrences.size(); ++iOcc) {
          std::pair<o2::MCCompLabel, size_t>& occurrence = occurrences[iOcc];
          for (auto& label : cl0labs) {
            if (label == occurrence.first) {
              ++occurrence.second;
              found = true;
              // break; // uncomment to stop to the first hit
            }
          }
        }
        if (!found) {
          for (auto& label : cl0labs) {
            occurrences.emplace_back(label, 1);
          }
        }

        const int cl1index{mTimeFrame->getClusters()[iCell + 1][currentCell.getSecondClusterIndex()].clusterId};

        const auto& cl1labs{mTimeFrame->getClusterLabels(iCell + 1, cl1index)};
        found = false;
        for (size_t iOcc{0}; iOcc < occurrences.size(); ++iOcc) {
          std::pair<o2::MCCompLabel, size_t>& occurrence = occurrences[iOcc];
          for (auto& label : cl1labs) {
            if (label == occurrence.first) {
              ++occurrence.second;
              found = true;
              // break; // uncomment to stop to the first hit
            }
          }
        }
        if (!found) {
          for (auto& label : cl1labs) {
            occurrences.emplace_back(label, 1);
          }
        }

        isFirstRoadCell = false;
      }

      const int cl2index{mTimeFrame->getClusters()[iCell + 2][currentCell.getThirdClusterIndex()].clusterId};
      const auto& cl2labs{mTimeFrame->getClusterLabels(iCell + 2, cl2index)};
      bool found{false};
      for (size_t iOcc{0}; iOcc < occurrences.size(); ++iOcc) {
        std::pair<o2::MCCompLabel, size_t>& occurrence = occurrences[iOcc];
        for (auto& label : cl2labs) {
          if (label == occurrence.first) {
            ++occurrence.second;
            found = true;
            // break; // uncomment to stop to the first hit
          }
        }
      }
      if (!found) {
        for (auto& label : cl2labs) {
          occurrences.emplace_back(label, 1);
        }
      }
    }

    std::sort(occurrences.begin(), occurrences.end(), [](auto e1, auto e2) {
      return e1.second > e2.second;
    });

    auto maxOccurrencesValue = occurrences[0].first;
    mTimeFrame->setRoadLabel(iRoad, maxOccurrencesValue.getRawValue(), isFakeRoad);
  }
}

void Tracker::computeTracksMClabels()
{

  for (int iROF{0}; iROF < mTimeFrame->getNrof(); ++iROF) {
    for (auto& track : mTimeFrame->getTracks(iROF)) {
      std::vector<std::pair<MCCompLabel, size_t>> occurrences;
      occurrences.clear();

      for (int iCluster = 0; iCluster < TrackITSExt::MaxClusters; ++iCluster) {
        const int index = track.getClusterIndex(iCluster);
        if (index == constants::its::UnusedIndex) {
          continue;
        }
        auto labels = mTimeFrame->getClusterLabels(iCluster, index);
        bool found{false};
        for (size_t iOcc{0}; iOcc < occurrences.size(); ++iOcc) {
          std::pair<o2::MCCompLabel, size_t>& occurrence = occurrences[iOcc];
          for (auto& label : labels) {
            if (label == occurrence.first) {
              ++occurrence.second;
              found = true;
              // break; // uncomment to stop to the first hit
            }
          }
        }
        if (!found) {
          for (auto& label : labels) {
            occurrences.emplace_back(label, 1);
          }
        }
      }
      std::sort(std::begin(occurrences), std::end(occurrences), [](auto e1, auto e2) {
        return e1.second > e2.second;
      });

      auto maxOccurrencesValue = occurrences[0].first;
      uint32_t pattern = track.getPattern();
      // set fake clusters pattern
      for (int ic{TrackITSExt::MaxClusters}; ic--;) {
        auto clid = track.getClusterIndex(ic);
        if (clid != constants::its::UnusedIndex) {
          auto labelsSpan = mTimeFrame->getClusterLabels(ic, clid);
          for (auto& currentLabel : labelsSpan) {
            if (currentLabel == maxOccurrencesValue) {
              pattern |= 0x1 << (16 + ic); // set bit if correct
              break;
            }
          }
        }
      }
      track.setPattern(pattern);
      if (occurrences[0].second < track.getNumberOfClusters()) {
        maxOccurrencesValue.setFakeFlag();
      }
      mTimeFrame->getTracksLabel(iROF).emplace_back(maxOccurrencesValue);
    }
  }
}

void Tracker::rectifyClusterIndices()
{
  for (int iROF{0}; iROF < mTimeFrame->getNrof(); ++iROF) {
    for (auto& track : mTimeFrame->getTracks(iROF)) {
      for (int iCluster = 0; iCluster < TrackITSExt::MaxClusters; ++iCluster) {
        const int index = track.getClusterIndex(iCluster);
        if (index != constants::its::UnusedIndex) {
          track.setExternalClusterIndex(iCluster, mTimeFrame->getClusterExternalIndex(iCluster, index));
        }
      }
    }
  }
}

/// Clusters are given from outside inward (cluster1 is the outermost). The innermost cluster is given in the tracking
/// frame coordinates
/// whereas the others are referred to the global frame. This function is almost a clone of CookSeed, adapted to return
/// a TrackParCov
track::TrackParCov Tracker::buildTrackSeed(const Cluster& cluster1, const Cluster& cluster2,
                                           const Cluster& cluster3, const TrackingFrameInfo& tf3, float resolution)
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

  const float fy = 1. / (cluster2.radius - cluster3.radius);
  const float& tz = fy;
  float cy = 1.e15f;
  if (std::abs(getBz()) > o2::constants::math::Almost0) {
    cy = (math_utils::computeCurvature(x1, y1, x2, y2 + resolution, x3, y3) - crv) /
         (resolution * getBz() * o2::constants::math::B2C) *
         20.f; // FIXME: MS contribution to the cov[14] (*20 added)
  }
  const float s2 = resolution;

  return track::TrackParCov(tf3.xTrackingFrame, tf3.alphaTrackingFrame,
                            {y3, z3, crv * (x3 - x0), 0.5f * (tgl12 + tgl23),
                             std::abs(getBz()) < o2::constants::math::Almost0 ? 1.f / o2::track::kMostProbablePt
                                                                              : crv / (getBz() * o2::constants::math::B2C)},
                            {s2, 0.f, s2, s2 * fy, 0.f, s2 * fy * fy, 0.f, s2 * tz, 0.f, s2 * tz * tz, s2 * cy, 0.f,
                             s2 * fy * cy, 0.f, s2 * cy * cy});
}

void Tracker::getGlobalConfiguration()
{
  auto& tc = o2::its::TrackerParamConfig::Instance();
  if (tc.useMatCorrTGeo) {
    setCorrType(o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrTGeo);
  }
  for (auto& params : mTrkParams) {
    if (params.NLayers == 7) {
      for (int i{0}; i < 7; ++i) {
        params.LayerMisalignment[i] = tc.sysErrZ2[i] > 0 ? std::sqrt(tc.sysErrZ2[i]) : params.LayerMisalignment[i];
      }
    }
    params.PhiBins = tc.LUTbinsPhi > 0 ? tc.LUTbinsPhi : params.PhiBins;
    params.ZBins = tc.LUTbinsZ > 0 ? tc.LUTbinsZ : params.ZBins;
    params.PVres = tc.pvRes > 0 ? tc.pvRes : params.PVres;
    params.NSigmaCut *= tc.nSigmaCut > 0 ? tc.nSigmaCut : 1.f;
    params.CellDeltaTanLambdaSigma *= tc.deltaTanLres > 0 ? tc.deltaTanLres : 1.f;
    params.TrackletMinPt *= tc.minPt > 0 ? tc.minPt : 1.f;
    for (int iD{0}; iD < 3; ++iD) {
      params.Diamond[iD] = tc.diamondPos[iD];
    }
    params.UseDiamond = tc.useDiamond;
    if (tc.maxMemory) {
      params.MaxMemory = tc.maxMemory;
    }
    if (tc.useTrackFollower >= 0) {
      params.UseTrackFollower = tc.useTrackFollower;
    }
    if (tc.cellsPerClusterLimit >= 0) {
      params.CellsPerClusterLimit = tc.cellsPerClusterLimit;
    }
    if (tc.trackletsPerClusterLimit >= 0) {
      params.TrackletsPerClusterLimit = tc.trackletsPerClusterLimit;
    }
  }
}

void Tracker::adoptTimeFrame(TimeFrame& tf)
{
  mTimeFrame = &tf;
  mTraits->adoptTimeFrame(&tf);
}

void Tracker::setBz(float bz)
{
  mBz = bz;
  mTimeFrame->setBz(bz);
}

} // namespace its
} // namespace o2
