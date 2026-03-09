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

#include <algorithm>
#include <iostream>
#include <iterator>
#include <ranges>
#include <type_traits>

#ifdef OPTIMISATION_OUTPUT
#include <format>
#include <fstream>
#endif

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_sort.h>

#include "CommonConstants/MathConstants.h"
#include "DetectorsBase/Propagator.h"
#include "GPUCommonMath.h"
#include "ITStracking/Cell.h"
#include "ITStracking/Constants.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/Tracklet.h"
#include "ReconstructionDataFormats/Track.h"

using o2::base::PropagatorF;

namespace o2::its
{

struct PassMode {
  using OnePass = std::integral_constant<int, 0>;
  using TwoPassCount = std::integral_constant<int, 1>;
  using TwoPassInsert = std::integral_constant<int, 2>;
};

template <int nLayers>
void TrackerTraits<nLayers>::computeLayerTracklets(const int iteration, int iROFslice, int iVertex)
{
#ifdef OPTIMISATION_OUTPUT
  static int iter{0};
  std::ofstream off(std::format("tracklets{}.txt", iter++));
#endif

  for (int iLayer = 0; iLayer < mTrkParams[iteration].TrackletsPerRoad(); ++iLayer) {
    mTimeFrame->getTracklets()[iLayer].clear();
    mTimeFrame->getTrackletsLabel(iLayer).clear();
    if (iLayer > 0) {
      std::fill(mTimeFrame->getTrackletsLookupTable()[iLayer - 1].begin(),
                mTimeFrame->getTrackletsLookupTable()[iLayer - 1].end(), 0);
    }
  }

  const Vertex diamondVert({mTrkParams[iteration].Diamond[0], mTrkParams[iteration].Diamond[1], mTrkParams[iteration].Diamond[2]}, {25.e-6f, 0.f, 0.f, 25.e-6f, 0.f, 36.f}, 1, 1.f);
  gsl::span<const Vertex> diamondSpan(&diamondVert, 1);
  int startROF{mTrkParams[iteration].nROFsPerIterations > 0 ? iROFslice * mTrkParams[iteration].nROFsPerIterations : 0};
  int endROF{o2::gpu::GPUCommonMath::Min(mTrkParams[iteration].nROFsPerIterations > 0 ? (iROFslice + 1) * mTrkParams[iteration].nROFsPerIterations + mTrkParams[iteration].DeltaROF : mTimeFrame->getNrof(), mTimeFrame->getNrof())};

  mTaskArena->execute([&] {
    auto forTracklets = [&](auto Tag, int iLayer, int pivotROF, int base, int& offset) -> int {
      if (!mTimeFrame->mMultiplicityCutMask[pivotROF]) {
        return 0;
      }
      int minROF = o2::gpu::CAMath::Max(startROF, pivotROF - mTrkParams[iteration].DeltaROF);
      int maxROF = o2::gpu::CAMath::Min(endROF - 1, pivotROF + mTrkParams[iteration].DeltaROF);
      gsl::span<const Vertex> primaryVertices = mTrkParams[iteration].UseDiamond ? diamondSpan : mTimeFrame->getPrimaryVertices(minROF, maxROF);
      if (primaryVertices.empty()) {
        return 0;
      }
      const int startVtx = iVertex >= 0 ? iVertex : 0;
      const int endVtx = iVertex >= 0 ? o2::gpu::CAMath::Min(iVertex + 1, int(primaryVertices.size())) : int(primaryVertices.size());
      if (endVtx <= startVtx) {
        return 0;
      }

      int localCount = 0;
      auto& tracklets = mTimeFrame->getTracklets()[iLayer];
      auto layer0 = mTimeFrame->getClustersOnLayer(pivotROF, iLayer);
      if (layer0.empty()) {
        return 0;
      }

      const float meanDeltaR = mTrkParams[iteration].LayerRadii[iLayer + 1] - mTrkParams[iteration].LayerRadii[iLayer];

      for (int iCluster = 0; iCluster < int(layer0.size()); ++iCluster) {
        const Cluster& currentCluster = layer0[iCluster];
        const int currentSortedIndex = mTimeFrame->getSortedIndex(pivotROF, iLayer, iCluster);
        if (mTimeFrame->isClusterUsed(iLayer, currentCluster.clusterId)) {
          continue;
        }
        const float inverseR0 = 1.f / currentCluster.radius;

        for (int iV = startVtx; iV < endVtx; ++iV) {
          const auto& pv = primaryVertices[iV];
          if ((pv.isFlagSet(Vertex::Flags::UPCMode) && iteration != 3) || (iteration == 3 && !pv.isFlagSet(Vertex::Flags::UPCMode))) {
            continue;
          }

          const float resolution = o2::gpu::CAMath::Sqrt(math_utils::Sq(mTimeFrame->getPositionResolution(iLayer)) + math_utils::Sq(mTrkParams[iteration].PVres) / float(pv.getNContributors()));
          const float tanLambda = (currentCluster.zCoordinate - pv.getZ()) * inverseR0;
          const float zAtRmin = tanLambda * (mTimeFrame->getMinR(iLayer + 1) - currentCluster.radius) + currentCluster.zCoordinate;
          const float zAtRmax = tanLambda * (mTimeFrame->getMaxR(iLayer + 1) - currentCluster.radius) + currentCluster.zCoordinate;
          const float sqInvDeltaZ0 = 1.f / (math_utils::Sq(currentCluster.zCoordinate - pv.getZ()) + constants::Tolerance);
          const float sigmaZ = o2::gpu::CAMath::Sqrt(
            math_utils::Sq(resolution) * math_utils::Sq(tanLambda) * ((math_utils::Sq(inverseR0) + sqInvDeltaZ0) * math_utils::Sq(meanDeltaR) + 1.f) + math_utils::Sq(meanDeltaR * mTimeFrame->getMSangle(iLayer)));

          auto bins = getBinsRect(currentCluster, iLayer + 1, zAtRmin, zAtRmax, sigmaZ * mTrkParams[iteration].NSigmaCut, mTimeFrame->getPhiCut(iLayer));
          if (bins.x == 0 && bins.y == 0 && bins.z == 0 && bins.w == 0) {
            continue;
          }
          int phiBinsNum = bins.w - bins.y + 1;
          if (phiBinsNum < 0) {
            phiBinsNum += mTrkParams[iteration].PhiBins;
          }

          for (int targetROF{minROF}; targetROF <= maxROF; ++targetROF) {
            if (!mTimeFrame->mMultiplicityCutMask[targetROF]) {
              continue;
            }
            auto layer1 = mTimeFrame->getClustersOnLayer(targetROF, iLayer + 1);
            if (layer1.empty()) {
              continue;
            }
            for (int iPhi = 0; iPhi < phiBinsNum; ++iPhi) {
              const int iPhiBin = (bins.y + iPhi) % mTrkParams[iteration].PhiBins;
              const int firstBinIdx = mTimeFrame->mIndexTableUtils.getBinIndex(bins.x, iPhiBin);
              const int maxBinIdx = firstBinIdx + (bins.z - bins.x) + 1;
              const int firstRow = mTimeFrame->getIndexTable(targetROF, iLayer + 1)[firstBinIdx];
              const int lastRow = mTimeFrame->getIndexTable(targetROF, iLayer + 1)[maxBinIdx];
              for (int iNext = firstRow; iNext < lastRow; ++iNext) {
                if (iNext >= int(layer1.size())) {
                  break;
                }
                const Cluster& nextCluster = layer1[iNext];
                if (mTimeFrame->isClusterUsed(iLayer + 1, nextCluster.clusterId)) {
                  continue;
                }
                float deltaPhi = o2::gpu::GPUCommonMath::Abs(currentCluster.phi - nextCluster.phi);
                float deltaZ = o2::gpu::GPUCommonMath::Abs((tanLambda * (nextCluster.radius - currentCluster.radius)) + currentCluster.zCoordinate - nextCluster.zCoordinate);

#ifdef OPTIMISATION_OUTPUT
                MCCompLabel label;
                int currentId{currentCluster.clusterId};
                int nextId{nextCluster.clusterId};
                for (auto& lab1 : mTimeFrame->getClusterLabels(iLayer, currentId)) {
                  for (auto& lab2 : mTimeFrame->getClusterLabels(iLayer + 1, nextId)) {
                    if (lab1 == lab2 && lab1.isValid()) {
                      label = lab1;
                      break;
                    }
                  }
                  if (label.isValid()) {
                    break;
                  }
                }
                off << std::format("{}\t{:d}\t{}\t{}\t{}\t{}", iLayer, label.isValid(), (tanLambda * (nextCluster.radius - currentCluster.radius) + currentCluster.zCoordinate - nextCluster.zCoordinate) / sigmaZ, tanLambda, resolution, sigmaZ) << std::endl;
#endif

                if (deltaZ / sigmaZ < mTrkParams[iteration].NSigmaCut &&
                    ((deltaPhi < mTimeFrame->getPhiCut(iLayer) || o2::gpu::GPUCommonMath::Abs(deltaPhi - o2::constants::math::TwoPI) < mTimeFrame->getPhiCut(iLayer)))) {
                  const float phi{o2::gpu::CAMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
                  const float tanL = (currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius);
                  if constexpr (decltype(Tag)::value == PassMode::OnePass::value) {
                    tracklets.emplace_back(currentSortedIndex, mTimeFrame->getSortedIndex(targetROF, iLayer + 1, iNext), tanL, phi, pivotROF, targetROF);
                  } else if constexpr (decltype(Tag)::value == PassMode::TwoPassCount::value) {
                    ++localCount;
                  } else if constexpr (decltype(Tag)::value == PassMode::TwoPassInsert::value) {
                    const int idx = base + offset++;
                    tracklets[idx] = Tracklet(currentSortedIndex, mTimeFrame->getSortedIndex(targetROF, iLayer + 1, iNext), tanL, phi, pivotROF, targetROF);
                  }
                }
              }
            }
          }
        }
      }
      return localCount;
    };

    int dummy{0};
    if (mTaskArena->max_concurrency() <= 1) {
      for (int pivotROF{startROF}; pivotROF < endROF; ++pivotROF) {
        for (int iLayer{0}; iLayer < mTrkParams[iteration].TrackletsPerRoad(); ++iLayer) {
          forTracklets(PassMode::OnePass{}, iLayer, pivotROF, 0, dummy);
        }
      }
    } else {
      bounded_vector<bounded_vector<int>> perROFCount(mTrkParams[iteration].TrackletsPerRoad(), bounded_vector<int>(endROF - startROF + 1, 0, mMemoryPool.get()), mMemoryPool.get());
      tbb::parallel_for(
        tbb::blocked_range2d<int, int>(0, mTrkParams[iteration].TrackletsPerRoad(), 1,
                                       startROF, endROF, 1),
        [&](auto const& Range) {
          for (int iLayer{Range.rows().begin()}; iLayer < Range.rows().end(); ++iLayer) {
            for (int pivotROF = Range.cols().begin(); pivotROF < Range.cols().end(); ++pivotROF) {
              perROFCount[iLayer][pivotROF - startROF] = forTracklets(PassMode::TwoPassCount{}, iLayer, pivotROF, 0, dummy);
            }
          }
        });

      tbb::parallel_for(0, mTrkParams[iteration].TrackletsPerRoad(), [&](const int iLayer) {
        std::exclusive_scan(perROFCount[iLayer].begin(), perROFCount[iLayer].end(), perROFCount[iLayer].begin(), 0);
        mTimeFrame->getTracklets()[iLayer].resize(perROFCount[iLayer].back());
      });

      tbb::parallel_for(
        tbb::blocked_range2d<int, int>(0, mTrkParams[iteration].TrackletsPerRoad(), 1,
                                       startROF, endROF, 1),
        [&](auto const& Range) {
          for (int iLayer{Range.rows().begin()}; iLayer < Range.rows().end(); ++iLayer) {
            if (perROFCount[iLayer].back() == 0) {
              continue;
            }
            for (int pivotROF = Range.cols().begin(); pivotROF < Range.cols().end(); ++pivotROF) {
              int baseIdx = perROFCount[iLayer][pivotROF - startROF];
              if (baseIdx == perROFCount[iLayer][pivotROF - startROF + 1]) {
                continue;
              }
              int localIdx = 0;
              forTracklets(PassMode::TwoPassInsert{}, iLayer, pivotROF, baseIdx, localIdx);
            }
          }
        });
    }

    tbb::parallel_for(0, mTrkParams[iteration].TrackletsPerRoad(), [&](const int iLayer) {
      /// Sort tracklets
      auto& trkl{mTimeFrame->getTracklets()[iLayer]};
      tbb::parallel_sort(trkl.begin(), trkl.end(), [](const Tracklet& a, const Tracklet& b) -> bool {
        if (a.firstClusterIndex != b.firstClusterIndex) {
          return a.firstClusterIndex < b.firstClusterIndex;
        }
        return a.secondClusterIndex < b.secondClusterIndex;
      });
      /// Remove duplicates
      trkl.erase(std::unique(trkl.begin(), trkl.end(), [](const Tracklet& a, const Tracklet& b) -> bool {
                   return a.firstClusterIndex == b.firstClusterIndex && a.secondClusterIndex == b.secondClusterIndex;
                 }),
                 trkl.end());
      trkl.shrink_to_fit();
      if (iLayer > 0) { /// recalculate lut
        auto& lut{mTimeFrame->getTrackletsLookupTable()[iLayer - 1]};
        if (!trkl.empty()) {
          for (const auto& tkl : trkl) {
            lut[tkl.firstClusterIndex + 1]++;
          }
          std::inclusive_scan(lut.begin(), lut.end(), lut.begin());
        }
      }
    });

    /// Create tracklets labels
    if (mTimeFrame->hasMCinformation() && mTrkParams[iteration].createArtefactLabels) {
      tbb::parallel_for(0, mTrkParams[iteration].TrackletsPerRoad(), [&](const int iLayer) {
        for (auto& trk : mTimeFrame->getTracklets()[iLayer]) {
          MCCompLabel label;
          int currentId{mTimeFrame->getClusters()[iLayer][trk.firstClusterIndex].clusterId};
          int nextId{mTimeFrame->getClusters()[iLayer + 1][trk.secondClusterIndex].clusterId};
          for (const auto& lab1 : mTimeFrame->getClusterLabels(iLayer, currentId)) {
            for (const auto& lab2 : mTimeFrame->getClusterLabels(iLayer + 1, nextId)) {
              if (lab1 == lab2 && lab1.isValid()) {
                label = lab1;
                break;
              }
            }
            if (label.isValid()) {
              break;
            }
          }
          mTimeFrame->getTrackletsLabel(iLayer).emplace_back(label);
        }
      });
    }
  });
} // namespace o2::its

template <int nLayers>
void TrackerTraits<nLayers>::computeLayerCells(const int iteration)
{
#ifdef OPTIMISATION_OUTPUT
  static int iter{0};
  std::ofstream off(std::format("cells{}.txt", iter++));
#endif

  for (int iLayer = 0; iLayer < mTrkParams[iteration].CellsPerRoad(); ++iLayer) {
    deepVectorClear(mTimeFrame->getCells()[iLayer]);
    if (iLayer > 0) {
      deepVectorClear(mTimeFrame->getCellsLookupTable()[iLayer - 1]);
    }
    if (mTimeFrame->hasMCinformation() && mTrkParams[iteration].createArtefactLabels) {
      deepVectorClear(mTimeFrame->getCellsLabel(iLayer));
    }
  }

  mTaskArena->execute([&] {
    auto forTrackletCells = [&](auto Tag, int iLayer, bounded_vector<CellSeedN>& layerCells, int iTracklet, int offset = 0) -> int {
      const Tracklet& currentTracklet{mTimeFrame->getTracklets()[iLayer][iTracklet]};
      const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
      const int nextLayerFirstTrackletIndex{mTimeFrame->getTrackletsLookupTable()[iLayer][nextLayerClusterIndex]};
      const int nextLayerLastTrackletIndex{mTimeFrame->getTrackletsLookupTable()[iLayer][nextLayerClusterIndex + 1]};
      int foundCells{0};
      for (int iNextTracklet{nextLayerFirstTrackletIndex}; iNextTracklet < nextLayerLastTrackletIndex; ++iNextTracklet) {
        const Tracklet& nextTracklet{mTimeFrame->getTracklets()[iLayer + 1][iNextTracklet]};
        const auto& nextLbl = mTimeFrame->getTrackletsLabel(iLayer + 1)[iNextTracklet];
        if (mTimeFrame->getTracklets()[iLayer + 1][iNextTracklet].firstClusterIndex != nextLayerClusterIndex) {
          break;
        }
        if (mTrkParams[iteration].DeltaROF && currentTracklet.getSpanRof(nextTracklet) > mTrkParams[iteration].DeltaROF) { // TODO this has to be improved for the staggering
          continue;
        }
        const float deltaTanLambda{std::abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};

#ifdef OPTIMISATION_OUTPUT
        float resolution{o2::gpu::CAMath::Sqrt(0.5f * (mTrkParams[iteration].SystErrorZ2[iLayer] + mTrkParams[iteration].SystErrorZ2[iLayer + 1] + mTrkParams[iteration].SystErrorZ2[iLayer + 2] + mTrkParams[iteration].SystErrorY2[iLayer] + mTrkParams[iteration].SystErrorY2[iLayer + 1] + mTrkParams[iteration].SystErrorY2[iLayer + 2])) / mTrkParams[iteration].LayerResolution[iLayer]};
        resolution = resolution > 1.e-12 ? resolution : 1.f;
        bool good{mTimeFrame->getTrackletsLabel(iLayer)[iTracklet] == mTimeFrame->getTrackletsLabel(iLayer + 1)[iNextTracklet]};
        float signedDelta{currentTracklet.tanLambda - nextTracklet.tanLambda};
        off << std::format("{}\t{:d}\t{}\t{}\t{}\t{}", iLayer, good, signedDelta, signedDelta / (mTrkParams[iteration].CellDeltaTanLambdaSigma), tanLambda, resolution) << std::endl;
#endif

        if (deltaTanLambda / mTrkParams[iteration].CellDeltaTanLambdaSigma < mTrkParams[iteration].NSigmaCut) {

          /// Track seed preparation. Clusters are numbered progressively from the innermost going outward.
          const int clusId[3]{
            mTimeFrame->getClusters()[iLayer][currentTracklet.firstClusterIndex].clusterId,
            mTimeFrame->getClusters()[iLayer + 1][nextTracklet.firstClusterIndex].clusterId,
            mTimeFrame->getClusters()[iLayer + 2][nextTracklet.secondClusterIndex].clusterId};
          const auto& cluster1_glo = mTimeFrame->getUnsortedClusters()[iLayer][clusId[0]];
          const auto& cluster2_glo = mTimeFrame->getUnsortedClusters()[iLayer + 1][clusId[1]];
          const auto& cluster3_tf = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer + 2)[clusId[2]];
          auto track{buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_tf)};

          float chi2{0.f};
          bool good{false};
          for (int iC{2}; iC--;) {
            const TrackingFrameInfo& trackingHit = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer + iC)[clusId[iC]];

            if (!track.rotate(trackingHit.alphaTrackingFrame)) {
              break;
            }

            if (!track.propagateTo(trackingHit.xTrackingFrame, getBz())) {
              break;
            }

            if (!track.correctForMaterial(mTrkParams[0].LayerxX0[iLayer + iC], mTrkParams[0].LayerxX0[iLayer + iC] * constants::Radl * constants::Rho, true)) {
              break;
            }

            const auto predChi2{track.getPredictedChi2Quiet(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
            if (!iC && predChi2 > mTrkParams[iteration].MaxChi2ClusterAttachment) {
              break;
            }

            if (!track.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
              break;
            }

            good = !iC;
            chi2 += predChi2;
          }
          if (good) {
            if constexpr (decltype(Tag)::value == PassMode::OnePass::value) {
              layerCells.emplace_back(iLayer, clusId[0], clusId[1], clusId[2], iTracklet, iNextTracklet, track, chi2);
              ++foundCells;
            } else if constexpr (decltype(Tag)::value == PassMode::TwoPassCount::value) {
              ++foundCells;
            } else if constexpr (decltype(Tag)::value == PassMode::TwoPassInsert::value) {
              layerCells[offset++] = CellSeedN(iLayer, clusId[0], clusId[1], clusId[2], iTracklet, iNextTracklet, track, chi2);
            } else {
              static_assert(false, "Unknown mode!");
            }
          }
        }
      }
      return foundCells;
    };

    tbb::parallel_for(0, mTrkParams[iteration].CellsPerRoad(), [&](const int iLayer) {
      if (mTimeFrame->getTracklets()[iLayer + 1].empty() ||
          mTimeFrame->getTracklets()[iLayer].empty()) {
        return;
      }

      auto& layerCells = mTimeFrame->getCells()[iLayer];
      const int currentLayerTrackletsNum{static_cast<int>(mTimeFrame->getTracklets()[iLayer].size())};
      bounded_vector<int> perTrackletCount(currentLayerTrackletsNum + 1, 0, mMemoryPool.get());
      if (mTaskArena->max_concurrency() <= 1) {
        for (int iTracklet{0}; iTracklet < currentLayerTrackletsNum; ++iTracklet) {
          perTrackletCount[iTracklet] = forTrackletCells(PassMode::OnePass{}, iLayer, layerCells, iTracklet);
        }
        std::exclusive_scan(perTrackletCount.begin(), perTrackletCount.end(), perTrackletCount.begin(), 0);
      } else {
        tbb::parallel_for(0, currentLayerTrackletsNum, [&](const int iTracklet) {
          perTrackletCount[iTracklet] = forTrackletCells(PassMode::TwoPassCount{}, iLayer, layerCells, iTracklet);
        });

        std::exclusive_scan(perTrackletCount.begin(), perTrackletCount.end(), perTrackletCount.begin(), 0);
        auto totalCells{perTrackletCount.back()};
        if (totalCells == 0) {
          return;
        }
        layerCells.resize(totalCells);

        tbb::parallel_for(0, currentLayerTrackletsNum, [&](const int iTracklet) {
          int offset = perTrackletCount[iTracklet];
          if (offset == perTrackletCount[iTracklet + 1]) {
            return;
          }
          forTrackletCells(PassMode::TwoPassInsert{}, iLayer, layerCells, iTracklet, offset);
        });
      }

      if (iLayer > 0) {
        auto& lut = mTimeFrame->getCellsLookupTable()[iLayer - 1];
        lut.resize(currentLayerTrackletsNum + 1);
        std::copy_n(perTrackletCount.begin(), currentLayerTrackletsNum + 1, lut.begin());
      }
    });

    /// Create cells labels
    if (mTimeFrame->hasMCinformation() && mTrkParams[iteration].createArtefactLabels) {
      tbb::parallel_for(0, mTrkParams[iteration].CellsPerRoad(), [&](const int iLayer) {
        mTimeFrame->getCellsLabel(iLayer).reserve(mTimeFrame->getCells()[iLayer].size());
        for (const auto& cell : mTimeFrame->getCells()[iLayer]) {
          MCCompLabel currentLab{mTimeFrame->getTrackletsLabel(iLayer)[cell.getFirstTrackletIndex()]};
          MCCompLabel nextLab{mTimeFrame->getTrackletsLabel(iLayer + 1)[cell.getSecondTrackletIndex()]};
          mTimeFrame->getCellsLabel(iLayer).emplace_back(currentLab == nextLab ? currentLab : MCCompLabel());
        }
      });
    }
  });
}

template <int nLayers>
void TrackerTraits<nLayers>::findCellsNeighbours(const int iteration)
{
#ifdef OPTIMISATION_OUTPUT
  std::ofstream off(std::format("cellneighs{}.txt", iteration));
#endif

  struct Neighbor {
    int cell{-1}, nextCell{-1}, level{-1};
  };

  mTaskArena->execute([&] {
    for (int iLayer{0}; iLayer < mTrkParams[iteration].NeighboursPerRoad(); ++iLayer) {
      deepVectorClear(mTimeFrame->getCellsNeighbours()[iLayer]);
      deepVectorClear(mTimeFrame->getCellsNeighboursLUT()[iLayer]);
      if (mTimeFrame->getCells()[iLayer + 1].empty() ||
          mTimeFrame->getCellsLookupTable()[iLayer].empty()) {
        continue;
      }

      int nCells{static_cast<int>(mTimeFrame->getCells()[iLayer].size())};
      bounded_vector<Neighbor> cellsNeighbours(mMemoryPool.get());

      auto forCellNeighbour = [&](auto Tag, int iCell, int offset = 0) -> int {
        const auto& currentCellSeed{mTimeFrame->getCells()[iLayer][iCell]};
        const int nextLayerTrackletIndex{currentCellSeed.getSecondTrackletIndex()};
        const int nextLayerFirstCellIndex{mTimeFrame->getCellsLookupTable()[iLayer][nextLayerTrackletIndex]};
        const int nextLayerLastCellIndex{mTimeFrame->getCellsLookupTable()[iLayer][nextLayerTrackletIndex + 1]};
        int foundNextCells{0};
        for (int iNextCell{nextLayerFirstCellIndex}; iNextCell < nextLayerLastCellIndex; ++iNextCell) {
          auto nextCellSeed{mTimeFrame->getCells()[iLayer + 1][iNextCell]}; /// copy
          if (nextCellSeed.getFirstTrackletIndex() != nextLayerTrackletIndex) {
            break;
          }

          if (mTrkParams[iteration].DeltaROF) { // TODO this has to be improved for the staggering
            const auto& trkl00 = mTimeFrame->getTracklets()[iLayer][currentCellSeed.getFirstTrackletIndex()];
            const auto& trkl01 = mTimeFrame->getTracklets()[iLayer + 1][currentCellSeed.getSecondTrackletIndex()];
            const auto& trkl10 = mTimeFrame->getTracklets()[iLayer + 1][nextCellSeed.getFirstTrackletIndex()];
            const auto& trkl11 = mTimeFrame->getTracklets()[iLayer + 2][nextCellSeed.getSecondTrackletIndex()];
            if ((std::max({trkl00.getMaxRof(), trkl01.getMaxRof(), trkl10.getMaxRof(), trkl11.getMaxRof()}) -
                 std::min({trkl00.getMinRof(), trkl01.getMinRof(), trkl10.getMinRof(), trkl11.getMinRof()})) > mTrkParams[0].DeltaROF) {
              continue;
            }
          }

          if (!nextCellSeed.rotate(currentCellSeed.getAlpha()) ||
              !nextCellSeed.propagateTo(currentCellSeed.getX(), getBz())) {
            continue;
          }
          float chi2 = currentCellSeed.getPredictedChi2(nextCellSeed); /// TODO: switch to the chi2 wrt cluster to avoid correlation

#ifdef OPTIMISATION_OUTPUT
          bool good{mTimeFrame->getCellsLabel(iLayer)[iCell] == mTimeFrame->getCellsLabel(iLayer + 1)[iNextCell]};
          off << std::format("{}\t{:d}\t{}", iLayer, good, chi2) << std::endl;
#endif

          if (chi2 > mTrkParams[0].MaxChi2ClusterAttachment) {
            continue;
          }

          if constexpr (decltype(Tag)::value == PassMode::OnePass::value) {
            cellsNeighbours.emplace_back(iCell, iNextCell, currentCellSeed.getLevel() + 1);
          } else if constexpr (decltype(Tag)::value == PassMode::TwoPassCount::value) {
            ++foundNextCells;
          } else if constexpr (decltype(Tag)::value == PassMode::TwoPassInsert::value) {
            cellsNeighbours[offset++] = {iCell, iNextCell, currentCellSeed.getLevel() + 1};
          } else {
            static_assert(false, "Unknown mode!");
          }
        }
        return foundNextCells;
      };

      if (mTaskArena->max_concurrency() <= 1) {
        for (int iCell{0}; iCell < nCells; ++iCell) {
          forCellNeighbour(PassMode::OnePass{}, iCell);
        }
      } else {
        bounded_vector<int> perCellCount(nCells + 1, 0, mMemoryPool.get());
        tbb::parallel_for(0, nCells, [&](const int iCell) {
          perCellCount[iCell] = forCellNeighbour(PassMode::TwoPassCount{}, iCell);
        });

        std::exclusive_scan(perCellCount.begin(), perCellCount.end(), perCellCount.begin(), 0);
        int totalCellNeighbours = perCellCount.back();
        if (totalCellNeighbours == 0) {
          deepVectorClear(mTimeFrame->getCellsNeighbours()[iLayer]);
          continue;
        }
        cellsNeighbours.resize(totalCellNeighbours);

        tbb::parallel_for(0, nCells, [&](const int iCell) {
          int offset = perCellCount[iCell];
          if (offset == perCellCount[iCell + 1]) {
            return;
          }
          forCellNeighbour(PassMode::TwoPassInsert{}, iCell, offset);
        });
      }

      if (cellsNeighbours.empty()) {
        continue;
      }

      tbb::parallel_sort(cellsNeighbours.begin(), cellsNeighbours.end(), [](const auto& a, const auto& b) {
        return a.nextCell < b.nextCell;
      });

      auto& cellsNeighbourLUT = mTimeFrame->getCellsNeighboursLUT()[iLayer];
      cellsNeighbourLUT.assign(mTimeFrame->getCells()[iLayer + 1].size(), 0);
      for (const auto& neigh : cellsNeighbours) {
        ++cellsNeighbourLUT[neigh.nextCell];
      }
      std::inclusive_scan(cellsNeighbourLUT.begin(), cellsNeighbourLUT.end(), cellsNeighbourLUT.begin());

      mTimeFrame->getCellsNeighbours()[iLayer].reserve(cellsNeighbours.size());
      std::ranges::transform(cellsNeighbours, std::back_inserter(mTimeFrame->getCellsNeighbours()[iLayer]), [](const auto& neigh) { return neigh.cell; });

      for (auto it = cellsNeighbours.begin(); it != cellsNeighbours.end();) {
        int cellIdx = it->nextCell;
        int maxLvl = it->level;
        while (++it != cellsNeighbours.end() && it->nextCell == cellIdx) {
          maxLvl = std::max(maxLvl, it->level);
        }
        mTimeFrame->getCells()[iLayer + 1][cellIdx].setLevel(maxLvl);
      }
    }
  });
}

template <int nLayers>
void TrackerTraits<nLayers>::processNeighbours(int iLayer, int iLevel, const bounded_vector<CellSeedN>& currentCellSeed, const bounded_vector<int>& currentCellId, bounded_vector<CellSeedN>& updatedCellSeeds, bounded_vector<int>& updatedCellsIds)
{
  CA_DEBUGGER(std::cout << "Processing neighbours layer " << iLayer << " level " << iLevel << ", size of the cell seeds: " << currentCellSeed.size() << std::endl);
  auto propagator = o2::base::Propagator::Instance();

#ifdef CA_DEBUG
  int failed[5]{0, 0, 0, 0, 0}, attempts{0}, failedByMismatch{0};
#endif

  mTaskArena->execute([&] {
    auto forCellNeighbours = [&](auto Tag, int iCell, int offset = 0) -> int {
      const auto& currentCell{currentCellSeed[iCell]};

      if constexpr (decltype(Tag)::value != PassMode::TwoPassInsert::value) {
        if (currentCell.getLevel() != iLevel) {
          return 0;
        }
        if (currentCellId.empty() && (mTimeFrame->isClusterUsed(iLayer, currentCell.getFirstClusterIndex()) ||
                                      mTimeFrame->isClusterUsed(iLayer + 1, currentCell.getSecondClusterIndex()) ||
                                      mTimeFrame->isClusterUsed(iLayer + 2, currentCell.getThirdClusterIndex()))) {
          return 0; /// this we do only on the first iteration, hence the check on currentCellId
        }
      }

      const int cellId = currentCellId.empty() ? iCell : currentCellId[iCell];
      const int startNeighbourId{cellId ? mTimeFrame->getCellsNeighboursLUT()[iLayer - 1][cellId - 1] : 0};
      const int endNeighbourId{mTimeFrame->getCellsNeighboursLUT()[iLayer - 1][cellId]};
      int foundSeeds{0};
      for (int iNeighbourCell{startNeighbourId}; iNeighbourCell < endNeighbourId; ++iNeighbourCell) {
        CA_DEBUGGER(attempts++);
        const int neighbourCellId = mTimeFrame->getCellsNeighbours()[iLayer - 1][iNeighbourCell];
        const auto& neighbourCell = mTimeFrame->getCells()[iLayer - 1][neighbourCellId];
        if (neighbourCell.getSecondTrackletIndex() != currentCell.getFirstTrackletIndex()) {
          CA_DEBUGGER(failedByMismatch++);
          continue;
        }
        if (mTimeFrame->isClusterUsed(iLayer - 1, neighbourCell.getFirstClusterIndex())) {
          continue;
        }
        if (currentCell.getLevel() - 1 != neighbourCell.getLevel()) {
          CA_DEBUGGER(failed[0]++);
          continue;
        }

        /// Let's start the fitting procedure
        CellSeedN seed{currentCell};
        const auto& trHit = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer - 1)[neighbourCell.getFirstClusterIndex()];

        if (!seed.rotate(trHit.alphaTrackingFrame)) {
          CA_DEBUGGER(failed[1]++);
          continue;
        }

        if (!propagator->propagateToX(seed, trHit.xTrackingFrame, getBz(), o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, mTrkParams[0].CorrType)) {
          CA_DEBUGGER(failed[2]++);
          continue;
        }

        if (mTrkParams[0].CorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
          if (!seed.correctForMaterial(mTrkParams[0].LayerxX0[iLayer - 1], mTrkParams[0].LayerxX0[iLayer - 1] * constants::Radl * constants::Rho, true)) {
            continue;
          }
        }

        auto predChi2{seed.getPredictedChi2Quiet(trHit.positionTrackingFrame, trHit.covarianceTrackingFrame)};
        if ((predChi2 > mTrkParams[0].MaxChi2ClusterAttachment) || predChi2 < 0.f) {
          CA_DEBUGGER(failed[3]++);
          continue;
        }
        seed.setChi2(seed.getChi2() + predChi2);
        if (!seed.o2::track::TrackParCov::update(trHit.positionTrackingFrame, trHit.covarianceTrackingFrame)) {
          CA_DEBUGGER(failed[4]++);
          continue;
        }

        if constexpr (decltype(Tag)::value != PassMode::TwoPassCount::value) {
          seed.getClusters()[iLayer - 1] = neighbourCell.getFirstClusterIndex();
          seed.setLevel(neighbourCell.getLevel());
          seed.setFirstTrackletIndex(neighbourCell.getFirstTrackletIndex());
          seed.setSecondTrackletIndex(neighbourCell.getSecondTrackletIndex());
        }

        if constexpr (decltype(Tag)::value == PassMode::OnePass::value) {
          updatedCellSeeds.push_back(seed);
          updatedCellsIds.push_back(neighbourCellId);
        } else if constexpr (decltype(Tag)::value == PassMode::TwoPassCount::value) {
          ++foundSeeds;
        } else if constexpr (decltype(Tag)::value == PassMode::TwoPassInsert::value) {
          updatedCellSeeds[offset] = seed;
          updatedCellsIds[offset++] = neighbourCellId;
        } else {
          static_assert(false, "Unknown mode!");
        }
      }
      return foundSeeds;
    };

    const int nCells = static_cast<int>(currentCellSeed.size());
    if (mTaskArena->max_concurrency() <= 1) {
      for (int iCell{0}; iCell < nCells; ++iCell) {
        forCellNeighbours(PassMode::OnePass{}, iCell);
      }
    } else {
      bounded_vector<int> perCellCount(nCells + 1, 0, mMemoryPool.get());
      tbb::parallel_for(0, nCells, [&](const int iCell) {
        perCellCount[iCell] = forCellNeighbours(PassMode::TwoPassCount{}, iCell);
      });

      std::exclusive_scan(perCellCount.begin(), perCellCount.end(), perCellCount.begin(), 0);
      auto totalNeighbours{perCellCount.back()};
      if (totalNeighbours == 0) {
        return;
      }
      updatedCellSeeds.resize(totalNeighbours);
      updatedCellsIds.resize(totalNeighbours);

      tbb::parallel_for(0, nCells, [&](const int iCell) {
        int offset = perCellCount[iCell];
        if (offset == perCellCount[iCell + 1]) {
          return;
        }
        forCellNeighbours(PassMode::TwoPassInsert{}, iCell, offset);
      });
    }
  });

#ifdef CA_DEBUG
  std::cout << "\t\t- Found " << updatedCellSeeds.size() << " cell seeds out of " << attempts << " attempts" << std::endl;
  std::cout << "\t\t\t> " << failed[0] << " failed because of level" << std::endl;
  std::cout << "\t\t\t> " << failed[1] << " failed because of rotation" << std::endl;
  std::cout << "\t\t\t> " << failed[2] << " failed because of propagation" << std::endl;
  std::cout << "\t\t\t> " << failed[3] << " failed because of chi2 cut" << std::endl;
  std::cout << "\t\t\t> " << failed[4] << " failed because of update" << std::endl;
  std::cout << "\t\t\t> " << failedByMismatch << " failed because of mismatch" << std::endl;
#endif
}

template <int nLayers>
void TrackerTraits<nLayers>::findRoads(const int iteration)
{
  bounded_vector<bounded_vector<int>> firstClusters(mTrkParams[iteration].NLayers, bounded_vector<int>(mMemoryPool.get()), mMemoryPool.get());
  bounded_vector<bounded_vector<int>> sharedFirstClusters(mTrkParams[iteration].NLayers, bounded_vector<int>(mMemoryPool.get()), mMemoryPool.get());
  firstClusters.resize(mTrkParams[iteration].NLayers);
  sharedFirstClusters.resize(mTrkParams[iteration].NLayers);
  for (int startLevel{mTrkParams[iteration].CellsPerRoad()}; startLevel >= mTrkParams[iteration].CellMinimumLevel(); --startLevel) {

    auto seedFilter = [&](const auto& seed) {
      return seed.getQ2Pt() <= 1.e3 && seed.getChi2() <= mTrkParams[0].MaxChi2NDF * ((startLevel + 2) * 2 - 5);
    };

    bounded_vector<CellSeedN> trackSeeds(mMemoryPool.get());
    for (int startLayer{mTrkParams[iteration].NeighboursPerRoad()}; startLayer >= startLevel - 1; --startLayer) {
      if ((mTrkParams[iteration].StartLayerMask & (1 << (startLayer + 2))) == 0) {
        continue;
      }

      bounded_vector<int> lastCellId(mMemoryPool.get()), updatedCellId(mMemoryPool.get());
      bounded_vector<CellSeedN> lastCellSeed(mMemoryPool.get()), updatedCellSeed(mMemoryPool.get());

      processNeighbours(startLayer, startLevel, mTimeFrame->getCells()[startLayer], lastCellId, updatedCellSeed, updatedCellId);

      int level = startLevel;
      for (int iLayer{startLayer - 1}; iLayer > 0 && level > 2; --iLayer) {
        lastCellSeed.swap(updatedCellSeed);
        lastCellId.swap(updatedCellId);
        deepVectorClear(updatedCellSeed); /// tame the memory peaks
        deepVectorClear(updatedCellId);   /// tame the memory peaks
        processNeighbours(iLayer, --level, lastCellSeed, lastCellId, updatedCellSeed, updatedCellId);
      }
      deepVectorClear(lastCellId);   /// tame the memory peaks
      deepVectorClear(lastCellSeed); /// tame the memory peaks

      if (!updatedCellSeed.empty()) {
        trackSeeds.reserve(trackSeeds.size() + std::count_if(updatedCellSeed.begin(), updatedCellSeed.end(), seedFilter));
        std::copy_if(updatedCellSeed.begin(), updatedCellSeed.end(), std::back_inserter(trackSeeds), seedFilter);
      }
    }

    if (trackSeeds.empty()) {
      continue;
    }

    bounded_vector<TrackITSExt> tracks(mMemoryPool.get());
    mTaskArena->execute([&] {
      auto forSeed = [&](auto Tag, int iSeed, int offset = 0) {
        TrackITSExt temporaryTrack = seedTrackForRefit(trackSeeds[iSeed]);
        o2::track::TrackPar linRef{temporaryTrack};
        bool fitSuccess = fitTrack(temporaryTrack, 0, mTrkParams[0].NLayers, 1, mTrkParams[0].MaxChi2ClusterAttachment, mTrkParams[0].MaxChi2NDF, o2::constants::math::VeryBig, 0, &linRef);
        if (!fitSuccess) {
          return 0;
        }
        temporaryTrack.getParamOut() = temporaryTrack.getParamIn();
        linRef = temporaryTrack.getParamOut(); // use refitted track as lin.reference
        temporaryTrack.resetCovariance();
        temporaryTrack.setCov(temporaryTrack.getQ2Pt() * temporaryTrack.getQ2Pt() * temporaryTrack.getCov()[o2::track::CovLabels::kSigQ2Pt2], o2::track::CovLabels::kSigQ2Pt2);
        temporaryTrack.setChi2(0);
        fitSuccess = fitTrack(temporaryTrack, mTrkParams[0].NLayers - 1, -1, -1, mTrkParams[0].MaxChi2ClusterAttachment, mTrkParams[0].MaxChi2NDF, 50.f, 0, &linRef);
        if (!fitSuccess || temporaryTrack.getPt() < mTrkParams[iteration].MinPt[mTrkParams[iteration].NLayers - temporaryTrack.getNClusters()]) {
          return 0;
        }
        if (mTrkParams[0].RepeatRefitOut) { // repeat outward refit seeding and linearizing with the stable inward fit result
          o2::track::TrackParCov saveInw{temporaryTrack};
          linRef = saveInw; // use refitted track as lin.reference
          float saveChi2 = temporaryTrack.getChi2();
          temporaryTrack.resetCovariance();
          temporaryTrack.setCov(temporaryTrack.getQ2Pt() * temporaryTrack.getQ2Pt() * temporaryTrack.getCov()[o2::track::CovLabels::kSigQ2Pt2], o2::track::CovLabels::kSigQ2Pt2);
          temporaryTrack.setChi2(0);
          fitSuccess = fitTrack(temporaryTrack, 0, mTrkParams[0].NLayers, 1, mTrkParams[0].MaxChi2ClusterAttachment, mTrkParams[0].MaxChi2NDF, o2::constants::math::VeryBig, 0, &linRef);
          if (!fitSuccess) {
            return 0;
          }
          temporaryTrack.getParamOut() = temporaryTrack.getParamIn();
          temporaryTrack.getParamIn() = saveInw;
          temporaryTrack.setChi2(saveChi2);
        }

        if constexpr (decltype(Tag)::value == PassMode::OnePass::value) {
          tracks.push_back(temporaryTrack);
        } else if constexpr (decltype(Tag)::value == PassMode::TwoPassCount::value) {
          // nothing to do
        } else if constexpr (decltype(Tag)::value == PassMode::TwoPassInsert::value) {
          tracks[offset] = temporaryTrack;
        } else {
          static_assert(false, "Unknown mode!");
        }
        return 1;
      };

      const int nSeeds = static_cast<int>(trackSeeds.size());
      if (mTaskArena->max_concurrency() <= 1) {
        for (int iSeed{0}; iSeed < nSeeds; ++iSeed) {
          forSeed(PassMode::OnePass{}, iSeed);
        }
      } else {
        bounded_vector<int> perSeedCount(nSeeds + 1, 0, mMemoryPool.get());
        tbb::parallel_for(0, nSeeds, [&](const int iSeed) {
          perSeedCount[iSeed] = forSeed(PassMode::TwoPassCount{}, iSeed);
        });

        std::exclusive_scan(perSeedCount.begin(), perSeedCount.end(), perSeedCount.begin(), 0);
        auto totalTracks{perSeedCount.back()};
        if (totalTracks == 0) {
          return;
        }
        tracks.resize(totalTracks);

        tbb::parallel_for(0, nSeeds, [&](const int iSeed) {
          if (perSeedCount[iSeed] == perSeedCount[iSeed + 1]) {
            return;
          }
          forSeed(PassMode::TwoPassInsert{}, iSeed, perSeedCount[iSeed]);
        });
      }

      deepVectorClear(trackSeeds);
      tbb::parallel_sort(tracks.begin(), tracks.end(), [](const auto& a, const auto& b) {
        return a.getChi2() < b.getChi2();
      });
    });

    for (auto& track : tracks) {
      int nShared = 0;
      bool isFirstShared{false};
      int firstLayer{-1}, firstCluster{-1};
      for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
        if (track.getClusterIndex(iLayer) == constants::UnusedIndex) {
          continue;
        }
        bool isShared = mTimeFrame->isClusterUsed(iLayer, track.getClusterIndex(iLayer));
        nShared += int(isShared);
        if (firstLayer < 0) {
          firstCluster = track.getClusterIndex(iLayer);
          isFirstShared = isShared && mTrkParams[0].AllowSharingFirstCluster && std::find(firstClusters[iLayer].begin(), firstClusters[iLayer].end(), firstCluster) != firstClusters[iLayer].end();
          firstLayer = iLayer;
        }
      }

      /// do not account for the first cluster in the shared clusters number if it is allowed
      if (nShared - int(isFirstShared && mTrkParams[0].AllowSharingFirstCluster) > mTrkParams[0].ClusterSharing) {
        continue;
      }

      std::array<int, 3> rofs{INT_MAX, INT_MAX, INT_MAX};
      for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
        if (track.getClusterIndex(iLayer) == constants::UnusedIndex) {
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
      track.setUserField(0);
      track.getParamOut().setUserField(0);
      if (rofs[1] != INT_MAX) {
        track.setNextROFbit();
      }
      mTimeFrame->getTracks(o2::gpu::CAMath::Min(rofs[0], rofs[1])).emplace_back(track);

      firstClusters[firstLayer].push_back(firstCluster);
      if (isFirstShared) {
        sharedFirstClusters[firstLayer].push_back(firstCluster);
      }
    }
  }

  /// Now we have to set the shared cluster flag
  for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
    std::sort(sharedFirstClusters[iLayer].begin(), sharedFirstClusters[iLayer].end());
  }

  for (int iROF{0}; iROF < mTimeFrame->getNrof(); ++iROF) {
    for (auto& track : mTimeFrame->getTracks(iROF)) {
      int firstLayer{mTrkParams[0].NLayers}, firstCluster{constants::UnusedIndex};
      for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
        if (track.getClusterIndex(iLayer) == constants::UnusedIndex) {
          continue;
        }
        firstLayer = iLayer;
        firstCluster = track.getClusterIndex(iLayer);
        break;
      }
      if (std::binary_search(sharedFirstClusters[firstLayer].begin(), sharedFirstClusters[firstLayer].end(), firstCluster)) {
        track.setSharedClusters();
      }
    }
  }
}

template <int nLayers>
void TrackerTraits<nLayers>::extendTracks(const int iteration)
{
  for (int rof{0}; rof < mTimeFrame->getNrof(); ++rof) {
    for (auto& track : mTimeFrame->getTracks(rof)) {
      auto backup{track};
      bool success{false};
      // the order here biases towards top extension, tracks should probably be fitted separately in the directions and then compared.
      if ((mTrkParams[iteration].UseTrackFollowerMix || mTrkParams[iteration].UseTrackFollowerTop) && track.getLastClusterLayer() != mTrkParams[iteration].NLayers - 1) {
        success = success || trackFollowing(&track, rof, true, iteration);
      }
      if ((mTrkParams[iteration].UseTrackFollowerMix || (mTrkParams[iteration].UseTrackFollowerBot && !success)) && track.getFirstClusterLayer() != 0) {
        success = success || trackFollowing(&track, rof, false, iteration);
      }
      if (success) {
        /// We have to refit the track
        track.resetCovariance();
        track.setChi2(0);
        bool fitSuccess = fitTrack(track, 0, mTrkParams[iteration].NLayers, 1, mTrkParams[iteration].MaxChi2ClusterAttachment, mTrkParams[0].MaxChi2NDF);
        if (!fitSuccess) {
          track = backup;
          continue;
        }
        track.getParamOut() = track;
        track.resetCovariance();
        track.setChi2(0);
        fitSuccess = fitTrack(track, mTrkParams[iteration].NLayers - 1, -1, -1, mTrkParams[iteration].MaxChi2ClusterAttachment, mTrkParams[0].MaxChi2NDF, 50.);
        if (!fitSuccess) {
          track = backup;
          continue;
        }
        mTimeFrame->mNExtendedTracks++;
        mTimeFrame->mNExtendedUsedClusters += track.getNClusters() - backup.getNClusters();
        auto pattern = track.getPattern();
        auto diff = (pattern & ~backup.getPattern()) & 0xff;
        pattern |= (diff << 24);
        track.setPattern(pattern);
        /// Make sure that the newly attached clusters get marked as used
        for (int iLayer{0}; iLayer < mTrkParams[iteration].NLayers; ++iLayer) {
          if (track.getClusterIndex(iLayer) == constants::UnusedIndex) {
            continue;
          }
          mTimeFrame->markUsedCluster(iLayer, track.getClusterIndex(iLayer));
        }
      }
    }
  }
}

template <int nLayers>
void TrackerTraits<nLayers>::findShortPrimaries()
{
  const auto propagator = o2::base::Propagator::Instance();
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

    const auto& cluster3_tf = mTimeFrame->getTrackingFrameInfoOnLayer(2)[cluster3_glo.clusterId];
    TrackITSExt temporaryTrack{buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_tf)};
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

      float pvRes{mTrkParams[0].PVres / o2::gpu::CAMath::Sqrt(float(pvs[iV].getNContributors()))};
      const float posVtx[2]{0.f, pvs[iV].getZ()};
      const float covVtx[3]{pvRes, 0.f, pvRes};
      float chi2 = temporaryTrack.getPredictedChi2Quiet(posVtx, covVtx);
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

template <int nLayers>
bool TrackerTraits<nLayers>::fitTrack(TrackITSExt& track, int start, int end, int step, float chi2clcut, float chi2ndfcut, float maxQoverPt, int nCl, o2::track::TrackPar* linRef)
{
  auto propInstance = o2::base::Propagator::Instance();

  for (int iLayer{start}; iLayer != end; iLayer += step) {
    if (track.getClusterIndex(iLayer) == constants::UnusedIndex) {
      continue;
    }
    const TrackingFrameInfo& trackingHit = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer)[track.getClusterIndex(iLayer)];
    if (linRef) {
      if (!track.rotate(trackingHit.alphaTrackingFrame, *linRef, getBz())) {
        return false;
      }
      if (!propInstance->propagateToX(track, *linRef, trackingHit.xTrackingFrame, getBz(), o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, mTrkParams[0].CorrType)) {
        return false;
      }
      if (mTrkParams[0].CorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
        if (!track.correctForMaterial(*linRef, mTrkParams[0].LayerxX0[iLayer], mTrkParams[0].LayerxX0[iLayer] * constants::Radl * constants::Rho, true)) {
          continue;
        }
      }
    } else {
      if (!track.rotate(trackingHit.alphaTrackingFrame)) {
        return false;
      }
      if (!propInstance->propagateToX(track, trackingHit.xTrackingFrame, getBz(), o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, mTrkParams[0].CorrType)) {
        return false;
      }
      if (mTrkParams[0].CorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
        if (!track.correctForMaterial(mTrkParams[0].LayerxX0[iLayer], mTrkParams[0].LayerxX0[iLayer] * constants::Radl * constants::Rho, true)) {
          continue;
        }
      }
    }
    auto predChi2{track.getPredictedChi2Quiet(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
    if ((nCl >= 3 && predChi2 > chi2clcut) || predChi2 < 0.f) {
      return false;
    }
    track.setChi2(track.getChi2() + predChi2);
    if (!track.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
      return false;
    }
    if (linRef && mTrkParams[0].ShiftRefToCluster) { // displace the reference to the last updated cluster
      linRef->setY(trackingHit.positionTrackingFrame[0]);
      linRef->setZ(trackingHit.positionTrackingFrame[1]);
    }
    nCl++;
  }
  return std::abs(track.getQ2Pt()) < maxQoverPt && track.getChi2() < chi2ndfcut * (nCl * 2 - 5);
}

template <int nLayers>
bool TrackerTraits<nLayers>::trackFollowing(TrackITSExt* track, int rof, bool outward, const int iteration)
{
  auto propInstance = o2::base::Propagator::Instance();
  const int step = -1 + outward * 2;
  const int end = outward ? mTrkParams[iteration].NLayers - 1 : 0;
  bounded_vector<TrackITSExt> hypotheses(1, *track, mMemoryPool.get()); // possibly avoid reallocation
  for (size_t iHypo{0}; iHypo < hypotheses.size(); ++iHypo) {
    auto hypo{hypotheses[iHypo]};
    int iLayer = static_cast<int>(outward ? hypo.getLastClusterLayer() : hypo.getFirstClusterLayer());
    // per layer we add new hypotheses
    while (iLayer != end) {
      iLayer += step; // step through all layers until we reach the end, this allows for skipping on empty layers
      const float r = mTrkParams[iteration].LayerRadii[iLayer];
      // get an estimate of the trackinf-frame x for the next step
      float x{-999};
      if (!hypo.getXatLabR(r, x, mTimeFrame->getBz(), o2::track::DirAuto) || x <= 0.f) {
        continue;
      }
      // estimate hypo's trk parameters at that x
      auto& hypoParam{outward ? hypo.getParamOut() : hypo.getParamIn()};
      if (!propInstance->propagateToX(hypoParam, x, mTimeFrame->getBz(), PropagatorF::MAX_SIN_PHI,
                                      PropagatorF::MAX_STEP, mTrkParams[iteration].CorrType)) {
        continue;
      }

      if (mTrkParams[iteration].CorrType == PropagatorF::MatCorrType::USEMatCorrNONE) { // account for material affects if propagator does not
        if (!hypoParam.correctForMaterial(mTrkParams[iteration].LayerxX0[iLayer], mTrkParams[iteration].LayerxX0[iLayer] * constants::Radl * constants::Rho, true)) {
          continue;
        }
      }

      // calculate the search window on this layer
      const float phi{hypoParam.getPhi()};
      const float ePhi{o2::gpu::CAMath::Sqrt(hypoParam.getSigmaSnp2() / hypoParam.getCsp2())};
      const float z{hypoParam.getZ()};
      const float eZ{o2::gpu::CAMath::Sqrt(hypoParam.getSigmaZ2())};
      const int4 selectedBinsRect{getBinsRect(iLayer, phi, mTrkParams[iteration].TrackFollowerNSigmaCutPhi * ePhi, z, mTrkParams[iteration].TrackFollowerNSigmaCutZ * eZ)};
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

      // check all clusters in search windows for possible new hypotheses
      for (int iPhiCount = 0; iPhiCount < phiBinsNum; iPhiCount++) {
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

          const TrackingFrameInfo& trackingHit = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer)[nextCluster.clusterId];

          auto tbupdated{hypo};
          auto& tbuParams = outward ? tbupdated.getParamOut() : tbupdated.getParamIn();
          if (!tbuParams.rotate(trackingHit.alphaTrackingFrame)) {
            continue;
          }

          if (!propInstance->propagateToX(tbuParams, trackingHit.xTrackingFrame, mTimeFrame->getBz(),
                                          PropagatorF::MAX_SIN_PHI, PropagatorF::MAX_STEP, PropagatorF::MatCorrType::USEMatCorrNONE)) {
            continue;
          }

          auto predChi2{tbuParams.getPredictedChi2Quiet(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
          if (predChi2 >= track->getChi2() * mTrkParams[iteration].NSigmaCut) {
            continue;
          }

          if (!tbuParams.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
            continue;
          }
          tbupdated.setChi2(tbupdated.getChi2() + predChi2); /// This is wrong for outward propagation as the chi2 refers to inward parameters
          tbupdated.setExternalClusterIndex(iLayer, nextCluster.clusterId, true);
          hypotheses.emplace_back(tbupdated);
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

// create a new seed either from the existing track inner param or reseed from the edgepointd and cluster in the middle
template <int nLayers>
TrackITSExt TrackerTraits<nLayers>::seedTrackForRefit(const CellSeedN& seed)
{
  TrackITSExt temporaryTrack(seed);
  int lrMin = nLayers, lrMax = 0, lrMid = 0;
  for (int iL = 0; iL < nLayers; ++iL) {
    const int idx = seed.getCluster(iL);
    temporaryTrack.setExternalClusterIndex(iL, idx, idx != constants::UnusedIndex);
    if (idx != constants::UnusedIndex) {
      lrMin = o2::gpu::CAMath::Min(lrMin, iL);
      lrMax = o2::gpu::CAMath::Max(lrMax, iL);
    }
  }
  int ncl = temporaryTrack.getNClusters();
  if (ncl < mTrkParams[0].ReseedIfShorter) { // reseed with circle passing via edges and the midpoint
    if (ncl == mTrkParams[0].NLayers) {
      lrMin = 0;
      lrMax = mTrkParams[0].NLayers - 1;
      lrMid = (lrMin + lrMax) / 2;
    } else {
      lrMid = lrMin + 1;
      float midR = 0.5 * (mTrkParams[0].LayerRadii[lrMax] + mTrkParams[0].LayerRadii[lrMin]), dstMidR = o2::gpu::GPUCommonMath::Abs(midR - mTrkParams[0].LayerRadii[lrMid]);
      for (int iL = lrMid + 1; iL < lrMax; ++iL) { // find the midpoint as closest to the midR
        auto dst = o2::gpu::GPUCommonMath::Abs(midR - mTrkParams[0].LayerRadii[iL]);
        if (dst < dstMidR) {
          lrMid = iL;
          dstMidR = dst;
        }
      }
    }
    const auto& cluster0_tf = mTimeFrame->getTrackingFrameInfoOnLayer(lrMin)[seed.getCluster(lrMin)]; // if the sensor frame!
    const auto& cluster1_gl = mTimeFrame->getUnsortedClusters()[lrMid][seed.getCluster(lrMid)];       // global frame
    const auto& cluster2_gl = mTimeFrame->getUnsortedClusters()[lrMax][seed.getCluster(lrMax)];       // global frame
    temporaryTrack.getParamIn() = buildTrackSeed(cluster2_gl, cluster1_gl, cluster0_tf, true);
  }
  temporaryTrack.resetCovariance();
  temporaryTrack.setCov(temporaryTrack.getQ2Pt() * temporaryTrack.getQ2Pt() * temporaryTrack.getCov()[o2::track::CovLabels::kSigQ2Pt2], o2::track::CovLabels::kSigQ2Pt2);
  return temporaryTrack;
}

/// Clusters are given from inside outward (cluster3 is the outermost). The outermost cluster is given in the tracking
/// frame coordinates whereas the others are referred to the global frame.
template <int nLayers>
track::TrackParCov TrackerTraits<nLayers>::buildTrackSeed(const Cluster& cluster1, const Cluster& cluster2, const TrackingFrameInfo& tf3, bool reverse)
{
  const float sign = reverse ? -1.f : 1.f;

  float ca, sa;
  o2::gpu::CAMath::SinCos(tf3.alphaTrackingFrame, sa, ca);

  const float x1 = cluster1.xCoordinate * ca + cluster1.yCoordinate * sa;
  const float y1 = -cluster1.xCoordinate * sa + cluster1.yCoordinate * ca;
  const float x2 = cluster2.xCoordinate * ca + cluster2.yCoordinate * sa;
  const float y2 = -cluster2.xCoordinate * sa + cluster2.yCoordinate * ca;
  const float x3 = tf3.xTrackingFrame;
  const float y3 = tf3.positionTrackingFrame[0];

  float snp, q2pt, q2pt2;
  if (mIsZeroField) {
    const float dx = x3 - x1;
    const float dy = y3 - y1;
    snp = sign * dy / o2::gpu::CAMath::Hypot(dx, dy);
    q2pt = sign / track::kMostProbablePt;
    q2pt2 = 1.f;
  } else {
    const float crv = math_utils::computeCurvature(x3, y3, x2, y2, x1, y1);
    snp = sign * crv * (x3 - math_utils::computeCurvatureCentreX(x3, y3, x2, y2, x1, y1));
    q2pt = sign * crv / (mBz * o2::constants::math::B2C);
    q2pt2 = crv * crv;
  }

  const float tgl = 0.5f * (math_utils::computeTanDipAngle(x1, y1, x2, y2, cluster1.zCoordinate, cluster2.zCoordinate) +
                            math_utils::computeTanDipAngle(x2, y2, x3, y3, cluster2.zCoordinate, tf3.positionTrackingFrame[1]));
  const float sg2q2pt = track::kC1Pt2max * (q2pt2 > 0.0005f ? (q2pt2 < 1.f ? q2pt2 : 1.f) : 0.0005f);

  return {x3, tf3.alphaTrackingFrame, {y3, tf3.positionTrackingFrame[1], snp, tgl, q2pt}, {tf3.covarianceTrackingFrame[0], tf3.covarianceTrackingFrame[1], tf3.covarianceTrackingFrame[2], 0.f, 0.f, track::kCSnp2max, 0.f, 0.f, 0.f, track::kCTgl2max, 0.f, 0.f, 0.f, 0.f, sg2q2pt}};
}

template <int nLayers>
void TrackerTraits<nLayers>::setBz(float bz)
{
  mBz = bz;
  mIsZeroField = std::abs(mBz) < 0.01;
  mTimeFrame->setBz(bz);
}

template <int nLayers>
bool TrackerTraits<nLayers>::isMatLUT() const
{
  return o2::base::Propagator::Instance()->getMatLUT() && (mTrkParams[0].CorrType == o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT);
}

template <int nLayers>
void TrackerTraits<nLayers>::setNThreads(int n, std::shared_ptr<tbb::task_arena>& arena)
{
#if defined(OPTIMISATION_OUTPUT) || defined(CA_DEBUG)
  mTaskArena = std::make_shared<tbb::task_arena>(1);
#else
  if (arena == nullptr) {
    mTaskArena = std::make_shared<tbb::task_arena>(std::abs(n));
    LOGP(info, "Setting tracker with {} threads.", n);
  } else {
    mTaskArena = arena;
    LOGP(info, "Attaching tracker to calling thread's arena");
  }
#endif
}

template class TrackerTraits<7>;
// ALICE3 upgrade
#ifdef ENABLE_UPGRADES
template class TrackerTraits<11>;
#endif

} // namespace o2::its
