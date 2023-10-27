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

#include <algorithm>
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

#ifdef OPTIMISATION_OUTPUT
    float resolution{std::sqrt(0.5f * (mTrkParams[iteration].SystErrorZ2[iLayer] + mTrkParams[iteration].SystErrorZ2[iLayer + 1] + mTrkParams[iteration].SystErrorZ2[iLayer + 2] + mTrkParams[iteration].SystErrorY2[iLayer] + mTrkParams[iteration].SystErrorY2[iLayer + 1] + mTrkParams[iteration].SystErrorY2[iLayer + 2])) / mTrkParams[iteration].LayerResolution[iLayer]};
    resolution = resolution > 1.e-12 ? resolution : 1.f;
#endif
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

#ifdef OPTIMISATION_OUTPUT
        bool good{tf->getTrackletsLabel(iLayer)[iTracklet] == tf->getTrackletsLabel(iLayer + 1)[iNextTracklet]};
        float signedDelta{currentTracklet.tanLambda - nextTracklet.tanLambda};
        off << fmt::format("{}\t{:d}\t{}\t{}\t{}\t{}", iLayer, good, signedDelta, signedDelta / (mTrkParams[iteration].CellDeltaTanLambdaSigma), tanLambda, resolution) << std::endl;
#endif

        if (deltaTanLambda / mTrkParams[iteration].CellDeltaTanLambdaSigma < mTrkParams[iteration].NSigmaCut) {

          if (iLayer > 0 && (int)tf->getCellsLookupTable()[iLayer - 1].size() <= iTracklet) {
            tf->getCellsLookupTable()[iLayer - 1].resize(iTracklet + 1, tf->getCells()[iLayer].size());
          }

          /// Track seed preparation. Clusters are numbered progressively from the innermost going outward.
          const int clusId[3]{
            mTimeFrame->getClusters()[iLayer][currentTracklet.firstClusterIndex].clusterId,
            mTimeFrame->getClusters()[iLayer + 1][nextTracklet.firstClusterIndex].clusterId,
            mTimeFrame->getClusters()[iLayer + 2][nextTracklet.secondClusterIndex].clusterId};
          const auto& cluster1_glo = mTimeFrame->getUnsortedClusters()[iLayer].at(clusId[0]);
          const auto& cluster2_glo = mTimeFrame->getUnsortedClusters()[iLayer + 1].at(clusId[1]);
          const auto& cluster3_glo = mTimeFrame->getUnsortedClusters()[iLayer + 2].at(clusId[2]);
          const auto& cluster3_tf = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer + 2).at(clusId[2]);
          auto track{buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_glo, cluster3_tf)};

          float chi2{0.f};
          bool good{false};
          for (int iC{2}; iC--;) {
            const TrackingFrameInfo& trackingHit = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer + iC).at(clusId[iC]);

            if (!track.rotate(trackingHit.alphaTrackingFrame)) {
              break;
            }

            if (!track.propagateTo(trackingHit.xTrackingFrame, getBz())) {
              break;
            }

            constexpr float radl = 9.36f; // Radiation length of Si [cm]
            constexpr float rho = 2.33f;  // Density of Si [g/cm^3]
            if (!track.correctForMaterial(mTrkParams[0].LayerxX0[iLayer + iC], mTrkParams[0].LayerxX0[iLayer] * radl * rho, true)) {
              break;
            }

            auto predChi2{track.getPredictedChi2(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
            if (!track.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
              break;
            }
            if (!iC && predChi2 > mTrkParams[iteration].MaxChi2ClusterAttachment) {
              break;
            }
            good = !iC;
            chi2 += predChi2;
          }
          if (!good) {
            continue;
          }
          tf->getCells()[iLayer].emplace_back(
            currentTracklet.firstClusterIndex, nextTracklet.firstClusterIndex, nextTracklet.secondClusterIndex,
            iTracklet, iNextTracklet);
          tf->getCellSeeds()[iLayer].emplace_back(track);
          tf->getCellSeedsChi2()[iLayer].emplace_back(chi2);
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

    mTimeFrame->getCellsNeighboursLUT()[iLayer].clear();
    mTimeFrame->getCellsNeighboursLUT()[iLayer].resize(nextLayerCellsNum, 0);
    std::vector<std::pair<int, int>> cellsNeighbours;
    cellsNeighbours.reserve(nextLayerCellsNum);

    for (int iCell{0}; iCell < layerCellsNum; ++iCell) {

      const Cell& currentCell{mTimeFrame->getCells()[iLayer][iCell]};
      const auto& currentCellSeed{mTimeFrame->getCellSeeds()[iLayer][iCell]};
      const int nextLayerTrackletIndex{currentCell.getSecondTrackletIndex()};
      const int nextLayerFirstCellIndex{mTimeFrame->getCellsLookupTable()[iLayer][nextLayerTrackletIndex]};
      const int nextLayerLastCellIndex{mTimeFrame->getCellsLookupTable()[iLayer][nextLayerTrackletIndex + 1]};
      for (int iNextCell{nextLayerFirstCellIndex}; iNextCell < nextLayerLastCellIndex; ++iNextCell) {

        Cell& nextCell{mTimeFrame->getCells()[iLayer + 1][iNextCell]};
        auto nextCellSeed{mTimeFrame->getCellSeeds()[iLayer + 1][iNextCell]}; /// copy
        if (nextCell.getFirstTrackletIndex() != nextLayerTrackletIndex) {
          break;
        }

        if (!nextCellSeed.rotate(currentCellSeed.getAlpha()) ||
            !nextCellSeed.propagateTo(currentCellSeed.getX(), getBz())) {
          continue;
        }
        float chi2 = currentCellSeed.getPredictedChi2(nextCellSeed); /// TODO: switch to the chi2 wrt cluster to avoid correlation

#ifdef OPTIMISATION_OUTPUT
        bool good{mTimeFrame->getCellsLabel(iLayer)[iCell] == mTimeFrame->getCellsLabel(iLayer + 1)[iNextCell]};
        off << fmt::format("{}\t{:d}\t{}", iLayer, good, chi2) << std::endl;
#endif

        if (chi2 > mTrkParams[iteration].MaxChi2ClusterAttachment) {
          continue;
        }

        mTimeFrame->getCellsNeighboursLUT()[iLayer][iNextCell]++;
        cellsNeighbours.push_back(std::make_pair(iCell, iNextCell));
        const int currentCellLevel{currentCell.getLevel()};

        if (currentCellLevel >= nextCell.getLevel()) {
          nextCell.setLevel(currentCellLevel + 1);
        }
      }
    }
    std::sort(cellsNeighbours.begin(), cellsNeighbours.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
      return a.second < b.second;
    });
    mTimeFrame->getCellsNeighbours()[iLayer].clear();
    mTimeFrame->getCellsNeighbours()[iLayer].reserve(cellsNeighbours.size());
    for (auto& cellNeighboursIndex : cellsNeighbours) {
      mTimeFrame->getCellsNeighbours()[iLayer].push_back(cellNeighboursIndex.first);
    }
    std::inclusive_scan(mTimeFrame->getCellsNeighboursLUT()[iLayer].begin(), mTimeFrame->getCellsNeighboursLUT()[iLayer].end(), mTimeFrame->getCellsNeighboursLUT()[iLayer].begin());
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
        const int startNeighbourId{iCell ? mTimeFrame->getCellsNeighboursLUT()[iLayer - 1][iCell - 1] : 0};
        const int endNeighbourId{mTimeFrame->getCellsNeighboursLUT()[iLayer - 1][iCell]};
        bool isFirstValidNeighbour = true;
        for (int iNeighbourCell{startNeighbourId}; iNeighbourCell < endNeighbourId; ++iNeighbourCell) {
          const int neighbourCellId = mTimeFrame->getCellsNeighbours()[iLayer - 1][iNeighbourCell];
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
  std::vector<TrackITSExt> tracks(mTimeFrame->getRoads().size());

  std::atomic<size_t> trackIndex{0};
#pragma omp parallel for num_threads(mNThreads)
  for (size_t ri = 0; ri < mTimeFrame->getRoads().size(); ++ri) {
    auto& road = mTimeFrame->getRoads()[ri];
    std::vector<int> clusters(mTrkParams[0].NLayers, constants::its::UnusedIndex);
    int lastCellLevel = constants::its::UnusedIndex;
    int lastCellIndex{-1};
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
        lastCellIndex = cellIndex;
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

    TrackITSExt temporaryTrack{mTimeFrame->getCellSeeds()[lastCellLevel][lastCellIndex]};
    temporaryTrack.setChi2(mTimeFrame->getCellSeedsChi2()[lastCellLevel][lastCellIndex]);
    for (size_t iC = 0; iC < clusters.size(); ++iC) {
      temporaryTrack.setExternalClusterIndex(iC, clusters[iC], clusters[iC] != constants::its::UnusedIndex);
    }
    bool fitSuccess = fitTrack(temporaryTrack, lastCellLevel - 1, -1, -1, mTrkParams[0].MaxChi2ClusterAttachment, mTrkParams[0].MaxChi2NDF, 1.e3, 3);
    if (!fitSuccess) {
      continue;
    }
    CA_DEBUGGER(fitCounters[nClusters - 4]++);
    temporaryTrack.resetCovariance();
    temporaryTrack.setChi2(0);
    fitSuccess = fitTrack(temporaryTrack, 0, mTrkParams[0].NLayers, 1, mTrkParams[0].MaxChi2ClusterAttachment, mTrkParams[0].MaxChi2NDF);
    if (!fitSuccess) {
      continue;
    }
    CA_DEBUGGER(backpropagatedCounters[nClusters - 4]++);
    temporaryTrack.getParamOut() = temporaryTrack;
    temporaryTrack.resetCovariance();
    temporaryTrack.setChi2(0);
    fitSuccess = fitTrack(temporaryTrack, mTrkParams[0].NLayers - 1, -1, -1, mTrkParams[0].MaxChi2ClusterAttachment, mTrkParams[0].MaxChi2NDF, 50.);
    if (!fitSuccess) {
      continue;
    }
    // temporaryTrack.setROFrame(rof);
    tracks[trackIndex++] = temporaryTrack;
  }

  tracks.resize(trackIndex);

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
        track.setChi2(0);
        bool fitSuccess = fitTrack(track, 0, mTrkParams[0].NLayers, 1, mTrkParams[0].MaxChi2ClusterAttachment, mTrkParams[0].MaxChi2NDF);
        if (!fitSuccess) {
          track = backup;
          continue;
        }
        track.getParamOut() = track;
        track.resetCovariance();
        track.setChi2(0);
        fitSuccess = fitTrack(track, mTrkParams[0].NLayers - 1, -1, -1, mTrkParams[0].MaxChi2ClusterAttachment, mTrkParams[0].MaxChi2NDF, 50.);
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
    auto& cluster3_glo = mTimeFrame->getClusters()[2][cell.getThirdClusterIndex()];
    auto& cluster2_glo = mTimeFrame->getClusters()[1][cell.getSecondClusterIndex()];
    auto& cluster1_glo = mTimeFrame->getClusters()[0][cell.getFirstClusterIndex()];
    if (mTimeFrame->isClusterUsed(2, cluster1_glo.clusterId) ||
        mTimeFrame->isClusterUsed(1, cluster2_glo.clusterId) ||
        mTimeFrame->isClusterUsed(0, cluster3_glo.clusterId)) {
      continue;
    }

    std::array<int, 3> rofs{
      mTimeFrame->getClusterROF(2, cluster3_glo.clusterId),
      mTimeFrame->getClusterROF(1, cluster2_glo.clusterId),
      mTimeFrame->getClusterROF(0, cluster1_glo.clusterId)};
    if (rofs[0] != rofs[1] && rofs[1] != rofs[2] && rofs[0] != rofs[2]) {
      continue;
    }

    int rof{rofs[0]};
    if (rofs[1] == rofs[2]) {
      rof = rofs[2];
    }

    auto pvs{mTimeFrame->getPrimaryVertices(rof)};
    auto pvsXAlpha{mTimeFrame->getPrimaryVerticesXAlpha(rof)};

    const auto& cluster3_tf = mTimeFrame->getTrackingFrameInfoOnLayer(2).at(cluster3_glo.clusterId);
    TrackITSExt temporaryTrack{buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_glo, cluster3_tf)};
    temporaryTrack.setExternalClusterIndex(0, cluster1_glo.clusterId, true);
    temporaryTrack.setExternalClusterIndex(1, cluster2_glo.clusterId, true);
    temporaryTrack.setExternalClusterIndex(2, cluster3_glo.clusterId, true);

    /// add propagation to the primary vertices compatible with the ROF(s) of the cell
    bool fitSuccess = fitTrack(temporaryTrack, 1, -1, -1);
    if (!fitSuccess) {
      continue;
    }
    fitSuccess = false;

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
    bestTrack.setChi2(0.f);
    fitSuccess = fitTrack(bestTrack, 0, mTrkParams[0].NLayers, 1, mTrkParams[0].MaxChi2ClusterAttachment, mTrkParams[0].MaxChi2NDF);
    if (!fitSuccess) {
      continue;
    }
    bestTrack.getParamOut() = bestTrack;
    bestTrack.resetCovariance();
    bestTrack.setChi2(0.f);
    fitSuccess = fitTrack(bestTrack, mTrkParams[0].NLayers - 1, -1, -1, mTrkParams[0].MaxChi2ClusterAttachment, mTrkParams[0].MaxChi2NDF, 50.);
    if (!fitSuccess) {
      continue;
    }
    mTimeFrame->markUsedCluster(0, bestTrack.getClusterIndex(0));
    mTimeFrame->markUsedCluster(1, bestTrack.getClusterIndex(1));
    mTimeFrame->markUsedCluster(2, bestTrack.getClusterIndex(2));
    mTimeFrame->getTracks(rof).emplace_back(bestTrack);
  }
}

bool TrackerTraits::fitTrack(TrackITSExt& track, int start, int end, int step, float chi2clcut, float chi2ndfcut, float maxQoverPt, int nCl)
{
  auto propInstance = o2::base::Propagator::Instance();

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

    auto predChi2{track.getPredictedChi2(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
    if ((nCl >= 3 && predChi2 > chi2clcut) || predChi2 < 0.f) {
      return false;
    }
    track.setChi2(track.getChi2() + predChi2);
    if (!track.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
      return false;
    }
    nCl++;
  }
  return std::abs(track.getQ2Pt()) < maxQoverPt && track.getChi2() < chi2ndfcut * (nCl * 2 - 5);
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

          auto predChi2{tbuParams.getPredictedChi2(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
          if (predChi2 >= track->getChi2() * mTrkParams[iteration].NSigmaCut) {
            continue;
          }

          if (!tbuParams.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
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

/// Clusters are given from inside outward (cluster3 is the outermost). The outermost cluster is given in the tracking
/// frame coordinates whereas the others are referred to the global frame.
track::TrackParCov TrackerTraits::buildTrackSeed(const Cluster& cluster1, const Cluster& cluster2, const Cluster& cluster3, const TrackingFrameInfo& tf3)
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

  const bool zeroField{std::abs(getBz()) < o2::constants::math::Almost0};
  const float tgp = zeroField ? std::atan2(y3 - y1, x3 - x1) : 1.f;
  const float crv = zeroField ? 1.f : math_utils::computeCurvature(x3, y3, x2, y2, x1, y1);
  const float snp = zeroField ? tgp / std::sqrt(1.f + tgp * tgp) : crv * (x3 - math_utils::computeCurvatureCentreX(x3, y3, x2, y2, x1, y1));
  const float tgl12 = math_utils::computeTanDipAngle(x1, y1, x2, y2, z1, z2);
  const float tgl23 = math_utils::computeTanDipAngle(x2, y2, x3, y3, z2, z3);

  return track::TrackParCov(tf3.xTrackingFrame, tf3.alphaTrackingFrame,
                            {y3, z3, snp, 0.5f * (tgl12 + tgl23),
                             zeroField ? 1.f / o2::track::kMostProbablePt
                                       : crv / (getBz() * o2::constants::math::B2C)},
                            {tf3.covarianceTrackingFrame[0],
                             tf3.covarianceTrackingFrame[1], tf3.covarianceTrackingFrame[2],
                             0.f, 0.f, track::kCSnp2max,
                             0.f, 0.f, 0.f, track::kCTgl2max,
                             0.f, 0.f, 0.f, 0.f, track::kC1Pt2max});
}

void TrackerTraits::traverseCellsTree(const int currentCellId, const int currentLayerId)
{
  Cell& currentCell{mTimeFrame->getCells()[currentLayerId][currentCellId]};
  const int currentCellLevel = currentCell.getLevel();

  mTimeFrame->getRoads().back().addCell(currentLayerId, currentCellId);

  if (currentLayerId > 0 && currentCellLevel > 1) {
    bool isFirstValidNeighbour = true;
    const int startNeighbourId = currentCellId ? mTimeFrame->getCellsNeighboursLUT()[currentLayerId - 1][currentCellId - 1] : 0;
    const int endNeighbourId = mTimeFrame->getCellsNeighboursLUT()[currentLayerId - 1][currentCellId];
    for (int iNeighbourCell{startNeighbourId}; iNeighbourCell < endNeighbourId; ++iNeighbourCell) {
      const int neighbourCellId = mTimeFrame->getCellsNeighbours()[currentLayerId - 1][iNeighbourCell];
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

int TrackerTraits::getTFNumberOfClusters() const
{
  return mTimeFrame->getNumberOfClusters();
}

int TrackerTraits::getTFNumberOfTracklets() const
{
  return mTimeFrame->getNumberOfTracklets();
}

int TrackerTraits::getTFNumberOfCells() const
{
  return mTimeFrame->getNumberOfCells();
}

void TrackerTraits::adoptTimeFrame(TimeFrame* tf)
{
  mTimeFrame = tf;
}

// bool TrackerTraits::checkTFMemory(const int iteration)
// {
//   return mTimeFrame->checkMemory(mTrkParams[iteration].MaxMemory);
// }

} // namespace its
} // namespace o2
