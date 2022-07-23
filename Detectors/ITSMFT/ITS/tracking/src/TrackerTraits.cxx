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
/// \file TrackerTraits.cxx
/// \brief
///

#include "ITStracking/TrackerTraits.h"

#include <cassert>
#include <iostream>

#include <fmt/format.h>

#include "CommonConstants/MathConstants.h"
#include "DetectorsBase/Propagator.h"
#include "GPUCommonMath.h"
#include "ITStracking/Cell.h"
#include "ITStracking/Constants.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/Tracklet.h"
#include "ReconstructionDataFormats/Track.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

using o2::base::PropagatorF;

namespace
{
float Sq(float q)
{
  return q * q;
}
} // namespace

namespace o2
{
namespace its
{

constexpr int debugLevel{0};

void TrackerTraits::computeLayerTracklets(const int iteration)
{
  TimeFrame* tf = mTimeFrame;

#ifdef OPTIMISATION_OUTPUT
  static int iter{0};
  std::ofstream off(fmt::format("tracklets{}.txt", iter++));
#endif

  const Vertex diamondVert({mTrkParams[iteration].Diamond[0], mTrkParams[iteration].Diamond[1], mTrkParams[iteration].Diamond[2]}, {25.e-6f, 0.f, 0.f, 25.e-6f, 0.f, 36.f}, 1, 1.f);
  gsl::span<const Vertex> diamondSpan(&diamondVert, 1);
  for (int rof0{0}; rof0 < tf->getNrof(); ++rof0) {
    gsl::span<const Vertex> primaryVertices = mTrkParams[iteration].UseDiamond ? diamondSpan : tf->getPrimaryVertices(rof0);
    int minRof = (rof0 >= mTrkParams[iteration].DeltaROF) ? rof0 - mTrkParams[iteration].DeltaROF : 0;
    int maxRof = (rof0 == tf->getNrof() - mTrkParams[iteration].DeltaROF) ? rof0 : rof0 + mTrkParams[iteration].DeltaROF;
    for (int iLayer{0}; iLayer < mTrkParams[iteration].TrackletsPerRoad(); ++iLayer) {
      gsl::span<const Cluster> layer0 = tf->getClustersOnLayer(rof0, iLayer);
      if (layer0.empty()) {
        continue;
      }
      float meanDeltaR{mTrkParams[iteration].LayerRadii[iLayer + 1] - mTrkParams[iteration].LayerRadii[iLayer]};

      const int currentLayerClustersNum{static_cast<int>(layer0.size())};
      for (int iCluster{0}; iCluster < currentLayerClustersNum; ++iCluster) {
        const Cluster& currentCluster{layer0[iCluster]};
        const int currentSortedIndex{tf->getSortedIndex(rof0, iLayer, iCluster)};

        if (tf->isClusterUsed(iLayer, currentCluster.clusterId)) {
          continue;
        }
        const float inverseR0{1.f / currentCluster.radius};

        for (auto& primaryVertex : primaryVertices) {
          const float resolution = std::sqrt(Sq(mTrkParams[iteration].PVres) / primaryVertex.getNContributors() + Sq(tf->getPositionResolution(iLayer)));

          const float tanLambda{(currentCluster.zCoordinate - primaryVertex.getZ()) * inverseR0};

          const float zAtRmin{tanLambda * (tf->getMinR(iLayer + 1) - currentCluster.radius) + currentCluster.zCoordinate};
          const float zAtRmax{tanLambda * (tf->getMaxR(iLayer + 1) - currentCluster.radius) + currentCluster.zCoordinate};

          const float sqInverseDeltaZ0{1.f / (Sq(currentCluster.zCoordinate - primaryVertex.getZ()) + 2.e-8f)}; /// protecting from overflows adding the detector resolution
          const float sigmaZ{std::sqrt(Sq(resolution) * Sq(tanLambda) * ((Sq(inverseR0) + sqInverseDeltaZ0) * Sq(meanDeltaR) + 1.f) + Sq(meanDeltaR * tf->getMSangle(iLayer)))};

          const int4 selectedBinsRect{getBinsRect(currentCluster, iLayer, zAtRmin, zAtRmax,
                                                  sigmaZ * mTrkParams[iteration].NSigmaCut, tf->getPhiCut(iLayer))};

          if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
            continue;
          }

          int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};

          if (phiBinsNum < 0) {
            phiBinsNum += mTrkParams[iteration].PhiBins;
          }

          for (int rof1{minRof}; rof1 <= maxRof; ++rof1) {
            gsl::span<const Cluster> layer1 = tf->getClustersOnLayer(rof1, iLayer + 1);
            if (layer1.empty()) {
              continue;
            }

            for (int iPhiCount{0}; iPhiCount < phiBinsNum; iPhiCount++) {
              int iPhiBin = (selectedBinsRect.y + iPhiCount) % mTrkParams[iteration].PhiBins;
              const int firstBinIndex{tf->mIndexTableUtils.getBinIndex(selectedBinsRect.x, iPhiBin)};
              const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
              if constexpr (debugLevel) {
                if (firstBinIndex < 0 || firstBinIndex > tf->getIndexTable(rof1, iLayer + 1).size() ||
                    maxBinIndex < 0 || maxBinIndex > tf->getIndexTable(rof1, iLayer + 1).size()) {
                  std::cout << iLayer << "\t" << iCluster << "\t" << zAtRmin << "\t" << zAtRmax << "\t" << sigmaZ * mTrkParams[iteration].NSigmaCut << "\t" << tf->getPhiCut(iLayer) << std::endl;
                  std::cout << currentCluster.zCoordinate << "\t" << primaryVertex.getZ() << "\t" << currentCluster.radius << std::endl;
                  std::cout << tf->getMinR(iLayer + 1) << "\t" << currentCluster.radius << "\t" << currentCluster.zCoordinate << std::endl;
                  std::cout << "Illegal access to IndexTable " << firstBinIndex << "\t" << maxBinIndex << "\t" << selectedBinsRect.z << "\t" << selectedBinsRect.x << std::endl;
                  exit(1);
                }
              }
              const int firstRowClusterIndex = tf->getIndexTable(rof1, iLayer + 1)[firstBinIndex];
              const int maxRowClusterIndex = tf->getIndexTable(rof1, iLayer + 1)[maxBinIndex];

              for (int iNextCluster{firstRowClusterIndex}; iNextCluster < maxRowClusterIndex; ++iNextCluster) {
                if (iNextCluster >= (int)layer1.size()) {
                  break;
                }
                const Cluster& nextCluster{layer1[iNextCluster]};

                if (tf->isClusterUsed(iLayer + 1, nextCluster.clusterId)) {
                  continue;
                }

                const float deltaPhi{gpu::GPUCommonMath::Abs(currentCluster.phi - nextCluster.phi)};
                const float deltaZ{gpu::GPUCommonMath::Abs(tanLambda * (nextCluster.radius - currentCluster.radius) +
                                                           currentCluster.zCoordinate - nextCluster.zCoordinate)};

#ifdef OPTIMISATION_OUTPUT
                MCCompLabel label;
                int currentId{currentCluster.clusterId};
                int nextId{nextCluster.clusterId};
                for (auto& lab1 : tf->getClusterLabels(iLayer, currentId)) {
                  for (auto& lab2 : tf->getClusterLabels(iLayer + 1, nextId)) {
                    if (lab1 == lab2 && lab1.isValid()) {
                      label = lab1;
                      break;
                    }
                  }
                  if (label.isValid()) {
                    break;
                  }
                }
                off << fmt::format("{}\t{:d}\t{}\t{}\t{}\t{}", iLayer, label.isValid(), (tanLambda * (nextCluster.radius - currentCluster.radius) + currentCluster.zCoordinate - nextCluster.zCoordinate) / sigmaZ, tanLambda, resolution, sigmaZ) << std::endl;
#endif

                if (deltaZ / sigmaZ < mTrkParams[iteration].NSigmaCut &&
                    (deltaPhi < tf->getPhiCut(iLayer) ||
                     gpu::GPUCommonMath::Abs(deltaPhi - constants::math::TwoPi) < tf->getPhiCut(iLayer))) {
                  if (iLayer > 0) {
                    tf->getTrackletsLookupTable()[iLayer - 1][currentSortedIndex]++;
                  }
                  const float phi{o2::gpu::GPUCommonMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate,
                                                                currentCluster.xCoordinate - nextCluster.xCoordinate)};
                  const float tanL{(currentCluster.zCoordinate - nextCluster.zCoordinate) /
                                   (currentCluster.radius - nextCluster.radius)};
                  tf->getTracklets()[iLayer].emplace_back(currentSortedIndex, tf->getSortedIndex(rof1, iLayer + 1, iNextCluster), tanL, phi, rof0, rof1);
                }
              }
            }
          }
        }
      }
      if (!tf->checkMemory(mTrkParams[iteration].MaxMemory)) {
        return;
      }
    }
  }
  /// Cold code, fixups

  for (int iLayer{0}; iLayer < mTrkParams[iteration].CellsPerRoad(); ++iLayer) {
    /// Sort tracklets
    auto& trkl{tf->getTracklets()[iLayer + 1]};
    std::sort(trkl.begin(), trkl.end(), [](const Tracklet& a, const Tracklet& b) {
      return a.firstClusterIndex < b.firstClusterIndex || (a.firstClusterIndex == b.firstClusterIndex && a.secondClusterIndex < b.secondClusterIndex);
    });
    /// Remove duplicates
    auto& lut{tf->getTrackletsLookupTable()[iLayer]};
    int id0{-1}, id1{-1};
    std::vector<Tracklet> newTrk;
    newTrk.reserve(trkl.size());
    for (auto& trk : trkl) {
      if (trk.firstClusterIndex == id0 && trk.secondClusterIndex == id1) {
        lut[id0]--;
      } else {
        id0 = trk.firstClusterIndex;
        id1 = trk.secondClusterIndex;
        newTrk.push_back(trk);
      }
    }
    trkl.swap(newTrk);

    /// Compute LUT
    std::exclusive_scan(lut.begin(), lut.end(), lut.begin(), 0);
    lut.push_back(trkl.size());
  }
  /// Layer 0 is done outside the loop
  std::sort(tf->getTracklets()[0].begin(), tf->getTracklets()[0].end(), [](const Tracklet& a, const Tracklet& b) {
    return a.firstClusterIndex < b.firstClusterIndex || (a.firstClusterIndex == b.firstClusterIndex && a.secondClusterIndex < b.secondClusterIndex);
  });
  int id0{-1}, id1{-1};
  std::vector<Tracklet> newTrk;
  newTrk.reserve(tf->getTracklets()[0].size());
  for (auto& trk : tf->getTracklets()[0]) {
    if (trk.firstClusterIndex != id0 || trk.secondClusterIndex != id1) {
      id0 = trk.firstClusterIndex;
      id1 = trk.secondClusterIndex;
      newTrk.push_back(trk);
    }
  }
  tf->getTracklets()[0].swap(newTrk);

  /// Create tracklets labels
  if (tf->hasMCinformation()) {
    for (int iLayer{0}; iLayer < mTrkParams[iteration].TrackletsPerRoad(); ++iLayer) {
      for (auto& trk : tf->getTracklets()[iLayer]) {
        MCCompLabel label;
        int currentId{tf->getClusters()[iLayer][trk.firstClusterIndex].clusterId};
        int nextId{tf->getClusters()[iLayer + 1][trk.secondClusterIndex].clusterId};
        for (auto& lab1 : tf->getClusterLabels(iLayer, currentId)) {
          for (auto& lab2 : tf->getClusterLabels(iLayer + 1, nextId)) {
            if (lab1 == lab2 && lab1.isValid()) {
              label = lab1;
              break;
            }
          }
          if (label.isValid()) {
            break;
          }
        }
        tf->getTrackletsLabel(iLayer).emplace_back(label);
      }
    }
  }
}

void TrackerTraits::computeLayerCells(const int iteration)
{

#ifdef OPTIMISATION_OUTPUT
  static int iter{0};
  std::ofstream off(fmt::format("cells{}.txt", iter++));
#endif

  TimeFrame* tf = mTimeFrame;
  for (int iLayer{0}; iLayer < mTrkParams[iteration].CellsPerRoad(); ++iLayer) {

    if (tf->getTracklets()[iLayer + 1].empty() ||
        tf->getTracklets()[iLayer].empty()) {
      continue;
    }

    float resolution{std::sqrt(Sq(mTrkParams[iteration].LayerMisalignment[iLayer]) + Sq(mTrkParams[iteration].LayerMisalignment[iLayer + 1]) + Sq(mTrkParams[iteration].LayerMisalignment[iLayer + 2])) / mTrkParams[iteration].LayerResolution[iLayer]};
    resolution = resolution > 1.e-12 ? resolution : 1.f;

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

      for (int iNextTracklet{nextLayerFirstTrackletIndex}; iNextTracklet < nextLayerLastTrackletIndex; ++iNextTracklet) {
        if (tf->getTracklets()[iLayer + 1][iNextTracklet].firstClusterIndex != nextLayerClusterIndex) {
          break;
        }
        const Tracklet& nextTracklet{tf->getTracklets()[iLayer + 1][iNextTracklet]};
        const float deltaTanLambda{std::abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};
        const float tanLambda{(currentTracklet.tanLambda + nextTracklet.tanLambda) * 0.5f};

#ifdef OPTIMISATION_OUTPUT
        bool good{tf->getTrackletsLabel(iLayer)[iTracklet] == tf->getTrackletsLabel(iLayer + 1)[iNextTracklet]};
        float signedDelta{currentTracklet.tanLambda - nextTracklet.tanLambda};
        off << fmt::format("{}\t{:d}\t{}\t{}\t{}\t{}", iLayer, good, signedDelta, signedDelta / (mTrkParams[iteration].CellDeltaTanLambdaSigma), tanLambda, resolution) << std::endl;
#endif

        if (deltaTanLambda / mTrkParams[iteration].CellDeltaTanLambdaSigma < mTrkParams[iteration].NSigmaCut) {

          if (iLayer > 0 && (int)tf->getCellsLookupTable()[iLayer - 1].size() <= iTracklet) {
            tf->getCellsLookupTable()[iLayer - 1].resize(iTracklet + 1, tf->getCells()[iLayer].size());
          }

          tf->getCells()[iLayer].emplace_back(
            currentTracklet.firstClusterIndex, nextTracklet.firstClusterIndex, nextTracklet.secondClusterIndex,
            iTracklet, iNextTracklet, tanLambda);
        }
      }
    }
    if (iLayer > 0) {
      tf->getCellsLookupTable()[iLayer - 1].resize(currentLayerTrackletsNum + 1, tf->getCells()[iLayer].size());
    }
    if (!tf->checkMemory(mTrkParams[iteration].MaxMemory)) {
      return;
    }
  }

  /// Create cells labels
  if (tf->hasMCinformation()) {
    for (int iLayer{0}; iLayer < mTrkParams[iteration].CellsPerRoad(); ++iLayer) {
      for (auto& cell : tf->getCells()[iLayer]) {
        MCCompLabel currentLab{tf->getTrackletsLabel(iLayer)[cell.getFirstTrackletIndex()]};
        MCCompLabel nextLab{tf->getTrackletsLabel(iLayer + 1)[cell.getSecondTrackletIndex()]};
        tf->getCellsLabel(iLayer).emplace_back(currentLab == nextLab ? currentLab : MCCompLabel());
      }
    }
  }

  if constexpr (debugLevel) {
    for (int iLayer{0}; iLayer < mTrkParams[iteration].CellsPerRoad(); ++iLayer) {
      std::cout << "Cells on layer " << iLayer << " " << tf->getCells()[iLayer].size() << std::endl;
    }
  }
}

void TrackerTraits::findCellsNeighbours(const int iteration)
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

void TrackerTraits::findRoads(const int iteration)
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

void TrackerTraits::findTracks()
{
  std::vector<std::vector<TrackITSExt>> tracks(mNThreads);
  for (auto& tracksV : tracks) {
    tracksV.reserve(mTimeFrame->getRoads().size() / mNThreads);
  }

#pragma omp parallel for num_threads(mNThreads)
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
#ifdef WITH_OPENMP
    int iThread = omp_get_thread_num();
#else
    int iThread = 0;
#endif
    tracks[iThread].emplace_back(temporaryTrack);
  }

  for (int iV{1}; iV < mNThreads; ++iV) {
    tracks[0].insert(tracks[0].end(), tracks[iV].begin(), tracks[iV].end());
  }

  if (mApplySmoothing) {
    // Smoothing tracks
  }
  std::sort(tracks[0].begin(), tracks[0].end(),
            [](TrackITSExt& track1, TrackITSExt& track2) { return track1.isBetter(track2, 1.e6f); });

  for (auto& track : tracks[0]) {
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

void TrackerTraits::extendTracks(const int iteration)
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
        success = success || trackFollowing(&track, rof, true, iteration);
      }
      if (track.getFirstClusterLayer() != 0) {
        success = success || trackFollowing(&track, rof, false, iteration);
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

void TrackerTraits::findShortPrimaries()
{
  if (!mTrkParams[0].FindShortTracks) {
    return;
  }
  auto propagator = o2::base::Propagator::Instance();
  mTimeFrame->fillPrimaryVerticesXandAlpha();

  for (auto& cell : mTimeFrame->getCells()[0]) {
    auto& cluster1_glo = mTimeFrame->getClusters()[2][cell.getThirdClusterIndex()];
    auto& cluster2_glo = mTimeFrame->getClusters()[1][cell.getSecondClusterIndex()];
    auto& cluster3_glo = mTimeFrame->getClusters()[0][cell.getFirstClusterIndex()];
    if (mTimeFrame->isClusterUsed(0, cluster1_glo.clusterId) ||
        mTimeFrame->isClusterUsed(1, cluster2_glo.clusterId) ||
        mTimeFrame->isClusterUsed(2, cluster3_glo.clusterId)) {
      continue;
    }

    std::array<int, 3> rofs{
      mTimeFrame->getClusterROF(0, cluster3_glo.clusterId),
      mTimeFrame->getClusterROF(1, cluster2_glo.clusterId),
      mTimeFrame->getClusterROF(2, cluster1_glo.clusterId)};
    if (rofs[0] != rofs[1] && rofs[1] != rofs[2] && rofs[0] != rofs[2]) {
      continue;
    }

    int rof{rofs[0]};
    if (rofs[1] == rofs[2]) {
      rof = rofs[2];
    }

    auto pvs{mTimeFrame->getPrimaryVertices(rof)};
    auto pvsXAlpha{mTimeFrame->getPrimaryVerticesXAlpha(rof)};

    const auto& cluster3_tf = mTimeFrame->getTrackingFrameInfoOnLayer(0).at(cluster3_glo.clusterId);
    TrackITSExt temporaryTrack{buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_glo, cluster3_tf, mTimeFrame->getPositionResolution(0))};
    temporaryTrack.setExternalClusterIndex(0, cluster3_glo.clusterId, true);
    temporaryTrack.setExternalClusterIndex(1, cluster2_glo.clusterId, true);
    temporaryTrack.setExternalClusterIndex(2, cluster1_glo.clusterId, true);

    /// add propagation to the primary vertices compatible with the ROF(s) of the cell
    bool fitSuccess{false};

    TrackITSExt bestTrack{temporaryTrack}, backup{temporaryTrack};
    float bestChi2{std::numeric_limits<float>::max()};
    for (int iV{0}; iV < (int)pvs.size(); ++iV) {
      temporaryTrack = backup;
      if (!temporaryTrack.rotate(pvsXAlpha[iV][1])) {
        continue;
      }
      if (!propagator->propagateTo(temporaryTrack, pvsXAlpha[iV][0], true)) {
        continue;
      }

      float pvRes{mTrkParams[0].PVres / std::sqrt(float(pvs[iV].getNContributors()))};
      const float posVtx[2]{0.f, pvs[iV].getZ()};
      const float covVtx[3]{pvRes, 0.f, pvRes};
      float chi2 = temporaryTrack.getPredictedChi2(posVtx, covVtx);
      if (chi2 < bestChi2) {
        if (!temporaryTrack.track::TrackParCov::update(posVtx, covVtx)) {
          continue;
        }
        bestTrack = temporaryTrack;
        bestChi2 = chi2;
      }
    }

    bestTrack.resetCovariance();
    fitSuccess = fitTrack(bestTrack, 0, mTrkParams[0].NLayers, 1, mTrkParams[0].FitIterationMaxChi2[0]);
    if (!fitSuccess) {
      continue;
    }
    bestTrack.getParamOut() = bestTrack;
    bestTrack.resetCovariance();
    fitSuccess = fitTrack(bestTrack, mTrkParams[0].NLayers - 1, -1, -1, mTrkParams[0].FitIterationMaxChi2[1], 50.);
    if (!fitSuccess) {
      continue;
    }
    mTimeFrame->markUsedCluster(0, bestTrack.getClusterIndex(0));
    mTimeFrame->markUsedCluster(1, bestTrack.getClusterIndex(1));
    mTimeFrame->markUsedCluster(2, bestTrack.getClusterIndex(2));
    mTimeFrame->getTracks(rof).emplace_back(bestTrack);
  }
}

bool TrackerTraits::fitTrack(TrackITSExt& track, int start, int end, int step, const float chi2cut, const float maxQoverPt)
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

void TrackerTraits::refitTracks(const int iteration, const std::vector<std::vector<TrackingFrameInfo>>& tf, std::vector<TrackITSExt>& tracks)
{
  std::vector<const Cell*> cells;
  for (int iLayer = 0; iLayer < mTrkParams[iteration].CellsPerRoad(); iLayer++) {
    cells.push_back(mTimeFrame->getCells()[iLayer].data());
  }
  std::vector<const Cluster*> clusters;
  for (int iLayer = 0; iLayer < mTrkParams[iteration].NLayers; iLayer++) {
    clusters.push_back(mTimeFrame->getClusters()[iLayer].data());
  }
  mChainRunITSTrackFit(*mChain, mTimeFrame->getRoads(), clusters, cells, tf, tracks);
}

bool TrackerTraits::trackFollowing(TrackITSExt* track, int rof, bool outward, const int iteration)
{
  auto propInstance = o2::base::Propagator::Instance();
  const int step = -1 + outward * 2;
  const int end = outward ? mTrkParams[iteration].NLayers - 1 : 0;
  std::vector<TrackITSExt> hypotheses(1, *track);
  for (auto& hypo : hypotheses) {
    int iLayer = outward ? track->getLastClusterLayer() : track->getFirstClusterLayer();
    while (iLayer != end) {
      iLayer += step;
      const float& r = mTrkParams[iteration].LayerRadii[iLayer];
      float x;
      if (!hypo.getXatLabR(r, x, mTimeFrame->getBz(), o2::track::DirAuto)) {
        continue;
      }
      bool success{false};
      auto& hypoParam{outward ? hypo.getParamOut() : hypo.getParamIn()};
      if (!propInstance->propagateToX(hypoParam, x, mTimeFrame->getBz(), PropagatorF::MAX_SIN_PHI,
                                      PropagatorF::MAX_STEP, mTrkParams[iteration].CorrType)) {
        continue;
      }

      if (mTrkParams[iteration].CorrType == PropagatorF::MatCorrType::USEMatCorrNONE) {
        float radl = 9.36f; // Radiation length of Si [cm]
        float rho = 2.33f;  // Density of Si [g/cm^3]
        if (!hypoParam.correctForMaterial(mTrkParams[iteration].LayerxX0[iLayer], mTrkParams[iteration].LayerxX0[iLayer] * radl * rho, true)) {
          continue;
        }
      }
      const float phi{hypoParam.getPhi()};
      const float ePhi{std::sqrt(hypoParam.getSigmaSnp2() / hypoParam.getCsp2())};
      const float z{hypoParam.getZ()};
      const float eZ{std::sqrt(hypoParam.getSigmaZ2())};
      const int4 selectedBinsRect{getBinsRect(iLayer, phi, mTrkParams[iteration].NSigmaCut * ePhi, z, mTrkParams[iteration].NSigmaCut * eZ)};

      if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
        continue;
      }

      int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};

      if (phiBinsNum < 0) {
        phiBinsNum += mTrkParams[iteration].PhiBins;
      }

      gsl::span<const Cluster> layer1 = mTimeFrame->getClustersOnLayer(rof, iLayer);
      if (layer1.empty()) {
        continue;
      }

      TrackITSExt currentHypo{hypo}, newHypo{hypo};
      bool first{true};
      for (int iPhiCount{0}; iPhiCount < phiBinsNum; iPhiCount++) {
        int iPhiBin = (selectedBinsRect.y + iPhiCount) % mTrkParams[iteration].PhiBins;
        const int firstBinIndex{mTimeFrame->mIndexTableUtils.getBinIndex(selectedBinsRect.x, iPhiBin)};
        const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
        const int firstRowClusterIndex = mTimeFrame->getIndexTable(rof, iLayer)[firstBinIndex];
        const int maxRowClusterIndex = mTimeFrame->getIndexTable(rof, iLayer)[maxBinIndex];

        for (int iNextCluster{firstRowClusterIndex}; iNextCluster < maxRowClusterIndex; ++iNextCluster) {
          if (iNextCluster >= (int)layer1.size()) {
            break;
          }
          const Cluster& nextCluster{layer1[iNextCluster]};

          if (mTimeFrame->isClusterUsed(iLayer, nextCluster.clusterId)) {
            continue;
          }

          const TrackingFrameInfo& trackingHit = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer).at(nextCluster.clusterId);

          TrackITSExt& tbupdated = first ? hypo : newHypo;
          auto& tbuParams = outward ? tbupdated.getParamOut() : tbupdated.getParamIn();
          if (!tbuParams.rotate(trackingHit.alphaTrackingFrame)) {
            continue;
          }

          if (!propInstance->propagateToX(tbuParams, trackingHit.xTrackingFrame, mTimeFrame->getBz(),
                                          PropagatorF::MAX_SIN_PHI, PropagatorF::MAX_STEP, PropagatorF::MatCorrType::USEMatCorrNONE)) {
            continue;
          }

          GPUArray<float, 3> cov{trackingHit.covarianceTrackingFrame};
          cov[0] = std::hypot(cov[0], mTrkParams[iteration].LayerMisalignment[iLayer]);
          cov[2] = std::hypot(cov[2], mTrkParams[iteration].LayerMisalignment[iLayer]);
          auto predChi2{tbuParams.getPredictedChi2(trackingHit.positionTrackingFrame, cov)};
          if (predChi2 >= track->getChi2() * mTrkParams[iteration].NSigmaCut) {
            continue;
          }

          if (!tbuParams.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, cov)) {
            continue;
          }
          tbupdated.setChi2(tbupdated.getChi2() + predChi2); /// This is wrong for outward propagation as the chi2 refers to inward parameters
          tbupdated.setExternalClusterIndex(iLayer, nextCluster.clusterId, true);

          if (!first) {
            hypotheses.emplace_back(tbupdated);
            newHypo = currentHypo;
          }
          first = false;
        }
      }
    }
  }

  TrackITSExt* bestHypo{track};
  bool swapped{false};
  for (auto& hypo : hypotheses) {
    if (hypo.isBetter(*bestHypo, track->getChi2() * mTrkParams[iteration].NSigmaCut)) {
      bestHypo = &hypo;
      swapped = true;
    }
  }
  *track = *bestHypo;
  return swapped;
}

/// Clusters are given from outside inward (cluster1 is the outermost). The innermost cluster is given in the tracking
/// frame coordinates
/// whereas the others are referred to the global frame. This function is almost a clone of CookSeed, adapted to return
/// a TrackParCov
track::TrackParCov TrackerTraits::buildTrackSeed(const Cluster& cluster1, const Cluster& cluster2, const Cluster& cluster3, const TrackingFrameInfo& tf3, float resolution)
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

void TrackerTraits::traverseCellsTree(const int currentCellId, const int currentLayerId)
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

void TrackerTraits::setBz(float bz)
{
  mBz = bz;
  mTimeFrame->setBz(bz);
}

bool TrackerTraits::isMatLUT() const { return o2::base::Propagator::Instance()->getMatLUT() && (mCorrType == o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT); }

void TrackerTraits::setNThreads(int n)
{
#ifdef WITH_OPENMP
  mNThreads = n > 0 ? n : 1;
#else
  mNThreads = 1;
#endif
}
} // namespace its
} // namespace o2
