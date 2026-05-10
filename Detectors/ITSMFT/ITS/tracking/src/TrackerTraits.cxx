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
#include <iterator>
#include <ranges>
#include <cmath>
#include <type_traits>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_sort.h>

#include "CommonConstants/MathConstants.h"
#include "DetectorsBase/Propagator.h"
#include "GPUCommonMath.h"
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/Cell.h"
#include "ITStracking/Constants.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/ROFLookupTables.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/TrackHelpers.h"
#include "ITStracking/Tracklet.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2::its
{

struct PassMode {
  using OnePass = std::integral_constant<int, 0>;
  using TwoPassCount = std::integral_constant<int, 1>;
  using TwoPassInsert = std::integral_constant<int, 2>;
};

template <int NLayers>
void TrackerTraits<NLayers>::computeLayerTracklets(const int iteration, int iVertex)
{
  for (int iLayer = 0; iLayer < mTrkParams[iteration].TrackletsPerRoad(); ++iLayer) {
    mTimeFrame->getTracklets()[iLayer].clear();
    mTimeFrame->getTrackletsLabel(iLayer).clear();
    if (iLayer > 0) {
      std::fill(mTimeFrame->getTrackletsLookupTable()[iLayer - 1].begin(), mTimeFrame->getTrackletsLookupTable()[iLayer - 1].end(), 0);
    }
  }

  const Vertex diamondVert(mTrkParams[iteration].Diamond, mTrkParams[iteration].DiamondCov, 1, 1.f);
  gsl::span<const Vertex> diamondSpan(&diamondVert, 1);

  mTaskArena->execute([&] {
    auto forTracklets = [&](auto Tag, int iLayer, int pivotROF, int base, int& offset) -> int {
      if (!mTimeFrame->getROFMaskView().isROFEnabled(iLayer, pivotROF)) {
        return 0;
      }
      gsl::span<const Vertex> primaryVertices = mTrkParams[iteration].UseDiamond ? diamondSpan : mTimeFrame->getPrimaryVertices(iLayer, pivotROF);
      if (primaryVertices.empty()) {
        return 0;
      }
      const int startVtx = iVertex >= 0 ? iVertex : 0;
      const int endVtx = iVertex >= 0 ? o2::gpu::CAMath::Min(iVertex + 1, int(primaryVertices.size())) : int(primaryVertices.size());
      if (endVtx <= startVtx || (iVertex + 1) > primaryVertices.size()) {
        return 0;
      }

      // does this layer have any overlap with the next layer
      const auto& rofOverlap = mTimeFrame->getROFOverlapTableView().getOverlap(iLayer, iLayer + 1, pivotROF);
      if (!rofOverlap.getEntries()) {
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
          if (!mTimeFrame->getROFVertexLookupTableView().isVertexCompatible(iLayer, pivotROF, pv)) {
            continue;
          }
          if (pv.isFlagSet(Vertex::Flags::UPCMode) != mTrkParams[iteration].PassFlags[IterationStep::SelectUPCVertices]) {
            continue;
          }
          const float resolution = o2::gpu::CAMath::Sqrt(math_utils::Sq(mTimeFrame->getPositionResolution(iLayer)) + math_utils::Sq(mTrkParams[iteration].PVres) / float(pv.getNContributors()));
          const float tanLambda = (currentCluster.zCoordinate - pv.getZ()) * inverseR0;
          const float zAtRmin = tanLambda * (mTimeFrame->getMinR(iLayer + 1) - currentCluster.radius) + currentCluster.zCoordinate;
          const float zAtRmax = tanLambda * (mTimeFrame->getMaxR(iLayer + 1) - currentCluster.radius) + currentCluster.zCoordinate;
          const float sqInvDeltaZ0 = 1.f / (math_utils::Sq(currentCluster.zCoordinate - pv.getZ()) + constants::Tolerance);
          const float sigmaZ = o2::gpu::CAMath::Sqrt(
            math_utils::Sq(resolution) * math_utils::Sq(tanLambda) * ((math_utils::Sq(inverseR0) + sqInvDeltaZ0) * math_utils::Sq(meanDeltaR) + 1.f) + math_utils::Sq(meanDeltaR * mTimeFrame->getMSangle(iLayer)));
          const auto bins = o2::its::getBinsRect(currentCluster, iLayer + 1, zAtRmin, zAtRmax,
                                                 sigmaZ * mTrkParams[iteration].NSigmaCut, mTimeFrame->getPhiCut(iLayer),
                                                 mTimeFrame->getIndexTableUtils());
          if (bins.x < 0) {
            continue;
          }
          int phiBinsNum = bins.w - bins.y + 1;
          if (phiBinsNum < 0) {
            phiBinsNum += mTrkParams[iteration].PhiBins;
          }

          for (int targetROF = rofOverlap.getFirstEntry(); targetROF < rofOverlap.getEntriesBound(); ++targetROF) {
            if (!mTimeFrame->getROFMaskView().isROFEnabled(iLayer + 1, targetROF)) {
              continue;
            }
            auto layer1 = mTimeFrame->getClustersOnLayer(targetROF, iLayer + 1);
            if (layer1.empty()) {
              continue;
            }
            const auto ts = mTimeFrame->getROFOverlapTableView().getTimeStamp(iLayer, pivotROF, iLayer + 1, targetROF);
            if (!ts.isCompatible(pv.getTimeStamp())) {
              continue;
            }
            const auto& targetIndexTable = mTimeFrame->getIndexTable(targetROF, iLayer + 1);
            const int zBinRange = (bins.z - bins.x) + 1;
            for (int iPhi = 0; iPhi < phiBinsNum; ++iPhi) {
              const int iPhiBin = (bins.y + iPhi) % mTrkParams[iteration].PhiBins;
              const int firstBinIdx = mTimeFrame->getIndexTableUtils().getBinIndex(bins.x, iPhiBin);
              const int maxBinIdx = firstBinIdx + zBinRange;
              const int firstRow = targetIndexTable[firstBinIdx];
              const int lastRow = targetIndexTable[maxBinIdx];
              for (int iNext = firstRow; iNext < lastRow; ++iNext) {
                if (iNext >= int(layer1.size())) {
                  break;
                }
                const Cluster& nextCluster = layer1[iNext];
                if (mTimeFrame->isClusterUsed(iLayer + 1, nextCluster.clusterId)) {
                  continue;
                }
                const float deltaZ = o2::gpu::CAMath::Abs((tanLambda * (nextCluster.radius - currentCluster.radius)) + currentCluster.zCoordinate - nextCluster.zCoordinate);

                if (deltaZ / sigmaZ < mTrkParams[iteration].NSigmaCut &&
                    math_utils::isPhiDifferenceBelow(currentCluster.phi, nextCluster.phi, mTimeFrame->getPhiCut(iLayer))) {
                  const float phi{o2::gpu::CAMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
                  const float tanL = (currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius);
                  if constexpr (decltype(Tag)::value == PassMode::OnePass::value) {
                    tracklets.emplace_back(currentSortedIndex, mTimeFrame->getSortedIndex(targetROF, iLayer + 1, iNext), tanL, phi, ts);
                  } else if constexpr (decltype(Tag)::value == PassMode::TwoPassCount::value) {
                    ++localCount;
                  } else if constexpr (decltype(Tag)::value == PassMode::TwoPassInsert::value) {
                    const int idx = base + offset++;
                    tracklets[idx] = Tracklet(currentSortedIndex, mTimeFrame->getSortedIndex(targetROF, iLayer + 1, iNext), tanL, phi, ts);
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
      for (int iLayer{0}; iLayer < mTrkParams[iteration].TrackletsPerRoad(); ++iLayer) {
        const int startROF = 0, endROF = mTimeFrame->getROFOverlapTableView().getLayer(iLayer).mNROFsTF;
        for (int pivotROF{startROF}; pivotROF < endROF; ++pivotROF) {
          forTracklets(PassMode::OnePass{}, iLayer, pivotROF, 0, dummy);
        }
      }
    } else {
      tbb::parallel_for(0, mTrkParams[iteration].TrackletsPerRoad(), [&](const int iLayer) {
        const int startROF = 0, endROF = mTimeFrame->getROFOverlapTableView().getLayer(iLayer).mNROFsTF;
        bounded_vector<int> perROFCount((endROF - startROF) + 1, mMemoryPool.get());
        tbb::parallel_for(startROF, endROF, [&](const int pivotROF) {
          perROFCount[pivotROF - startROF] = forTracklets(PassMode::TwoPassCount{}, iLayer, pivotROF, 0, dummy);
        });
        std::exclusive_scan(perROFCount.begin(), perROFCount.end(), perROFCount.begin(), 0);
        const int nTracklets = perROFCount.back();
        mTimeFrame->getTracklets()[iLayer].resize(nTracklets);
        if (nTracklets == 0) {
          return;
        }
        tbb::parallel_for(startROF, endROF, [&](const int pivotROF) {
          int baseIdx = perROFCount[pivotROF - startROF];
          if (baseIdx == perROFCount[pivotROF + 1 - startROF]) {
            return;
          }
          int localIdx = 0;
          forTracklets(PassMode::TwoPassInsert{}, iLayer, pivotROF, baseIdx, localIdx);
        });
      });
    }

    tbb::parallel_for(0, mTrkParams[iteration].TrackletsPerRoad(), [&](const int iLayer) {
      /// Sort tracklets & remove duplicates
      // duplicates can exist simply since we evaluate per vertex
      auto& trkl{mTimeFrame->getTracklets()[iLayer]};
      std::sort(trkl.begin(), trkl.end());
      trkl.erase(std::unique(trkl.begin(), trkl.end()), trkl.end());
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
    if (mTimeFrame->hasMCinformation() && mTrkParams[iteration].CreateArtefactLabels) {
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
}

template <int NLayers>
void TrackerTraits<NLayers>::computeLayerCells(const int iteration)
{
  for (int iLayer = 0; iLayer < mTrkParams[iteration].CellsPerRoad(); ++iLayer) {
    deepVectorClear(mTimeFrame->getCells()[iLayer]);
    if (iLayer > 0) {
      deepVectorClear(mTimeFrame->getCellsLookupTable()[iLayer - 1]);
    }
    if (mTimeFrame->hasMCinformation() && mTrkParams[iteration].CreateArtefactLabels) {
      deepVectorClear(mTimeFrame->getCellsLabel(iLayer));
    }
  }

  mTaskArena->execute([&] {
    auto forTrackletCells = [&](auto Tag, int iLayer, bounded_vector<CellSeed>& layerCells, int iTracklet, int offset = 0) -> int {
      const Tracklet& currentTracklet{mTimeFrame->getTracklets()[iLayer][iTracklet]};
      const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
      const int nextLayerFirstTrackletIndex{mTimeFrame->getTrackletsLookupTable()[iLayer][nextLayerClusterIndex]};
      const int nextLayerLastTrackletIndex{mTimeFrame->getTrackletsLookupTable()[iLayer][nextLayerClusterIndex + 1]};
      int foundCells{0};
      for (int iNextTracklet{nextLayerFirstTrackletIndex}; iNextTracklet < nextLayerLastTrackletIndex; ++iNextTracklet) {
        const Tracklet& nextTracklet{mTimeFrame->getTracklets()[iLayer + 1][iNextTracklet]};
        if (mTimeFrame->getTracklets()[iLayer + 1][iNextTracklet].firstClusterIndex != nextLayerClusterIndex) {
          break;
        }
        if (!currentTracklet.getTimeStamp().isCompatible(nextTracklet.getTimeStamp())) {
          continue;
        }

        const float deltaTanLambdaSigma = std::abs(currentTracklet.tanLambda - nextTracklet.tanLambda) / mTrkParams[iteration].CellDeltaTanLambdaSigma;
        if (deltaTanLambdaSigma < mTrkParams[iteration].NSigmaCut) {

          /// Track seed preparation. Clusters are numbered progressively from the innermost going outward.
          const int clusId[3]{
            mTimeFrame->getClusters()[iLayer][currentTracklet.firstClusterIndex].clusterId,
            mTimeFrame->getClusters()[iLayer + 1][nextTracklet.firstClusterIndex].clusterId,
            mTimeFrame->getClusters()[iLayer + 2][nextTracklet.secondClusterIndex].clusterId};
          const auto& cluster1_glo = mTimeFrame->getUnsortedClusters()[iLayer][clusId[0]];
          const auto& cluster2_glo = mTimeFrame->getUnsortedClusters()[iLayer + 1][clusId[1]];
          const auto& cluster3_tf = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer + 2)[clusId[2]];
          auto track{o2::its::track::buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_tf, mBz)};

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

            if (!track.correctForMaterial(mTrkParams[iteration].LayerxX0[iLayer + iC], mTrkParams[iteration].LayerxX0[iLayer + iC] * constants::Radl * constants::Rho, true)) {
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
            TimeEstBC ts = currentTracklet.getTimeStamp();
            ts += nextTracklet.getTimeStamp();
            if constexpr (decltype(Tag)::value == PassMode::OnePass::value) {
              layerCells.emplace_back(iLayer, clusId[0], clusId[1], clusId[2], iTracklet, iNextTracklet, track, chi2, ts);
              ++foundCells;
            } else if constexpr (decltype(Tag)::value == PassMode::TwoPassCount::value) {
              ++foundCells;
            } else if constexpr (decltype(Tag)::value == PassMode::TwoPassInsert::value) {
              layerCells[offset++] = CellSeed(iLayer, clusId[0], clusId[1], clusId[2], iTracklet, iNextTracklet, track, chi2, ts);
            } else {
              static_assert(false, "Unknown mode!");
            }
          }
        }
      }
      return foundCells;
    };

    for (int iLayer = 0; iLayer < mTrkParams[iteration].CellsPerRoad(); ++iLayer) {
      if (mTimeFrame->getTracklets()[iLayer + 1].empty() ||
          mTimeFrame->getTracklets()[iLayer].empty()) {
        if (iLayer < mTrkParams[iteration].TrackletsPerRoad()) {
          deepVectorClear(mTimeFrame->getTracklets()[iLayer]);
          deepVectorClear(mTimeFrame->getTrackletsLabel(iLayer));
        }
        continue;
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
          if (iLayer > 0) {
            auto& lut = mTimeFrame->getCellsLookupTable()[iLayer - 1];
            lut.resize(currentLayerTrackletsNum + 1);
            std::fill(lut.begin(), lut.end(), 0);
          }
          deepVectorClear(mTimeFrame->getTracklets()[iLayer]);
          deepVectorClear(mTimeFrame->getTrackletsLabel(iLayer));
          continue;
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

      if (mTimeFrame->hasMCinformation() && mTrkParams[iteration].CreateArtefactLabels) {
        auto& labels = mTimeFrame->getCellsLabel(iLayer);
        labels.reserve(layerCells.size());
        for (const auto& cell : layerCells) {
          MCCompLabel currentLab{mTimeFrame->getTrackletsLabel(iLayer)[cell.getFirstTrackletIndex()]};
          MCCompLabel nextLab{mTimeFrame->getTrackletsLabel(iLayer + 1)[cell.getSecondTrackletIndex()]};
          labels.emplace_back(currentLab == nextLab ? currentLab : MCCompLabel());
        }
      }

      // Once layer i cells are built and labelled, the corresponding tracklet artefacts are no longer needed.
      deepVectorClear(mTimeFrame->getTracklets()[iLayer]);
      deepVectorClear(mTimeFrame->getTrackletsLabel(iLayer));
    }
  });

  // Clear the trailing tracklet artefacts that are not consumed as the first leg of a cell.
  for (int iLayer = mTrkParams[iteration].CellsPerRoad(); iLayer < mTrkParams[iteration].TrackletsPerRoad(); ++iLayer) {
    deepVectorClear(mTimeFrame->getTracklets()[iLayer]);
    deepVectorClear(mTimeFrame->getTrackletsLabel(iLayer));
  }
}

template <int NLayers>
void TrackerTraits<NLayers>::findCellsNeighbours(const int iteration)
{
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
          if (nextCellSeed.getFirstTrackletIndex() != nextLayerTrackletIndex || !currentCellSeed.getTimeStamp().isCompatible(nextCellSeed.getTimeStamp())) {
            break;
          }

          if (!nextCellSeed.rotate(currentCellSeed.getAlpha()) ||
              !nextCellSeed.propagateTo(currentCellSeed.getX(), getBz())) {
            continue;
          }

          float chi2 = currentCellSeed.getPredictedChi2(nextCellSeed); /// TODO: switch to the chi2 wrt cluster to avoid correlation
          if (chi2 > mTrkParams[iteration].MaxChi2ClusterAttachment) {
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

      // clear cells LUT
      deepVectorClear(mTimeFrame->getCellsLookupTable()[iLayer]);
    }
  });
}

template <int NLayers>
template <typename InputSeed>
void TrackerTraits<NLayers>::processNeighbours(int iteration, int iLayer, int iLevel, const bounded_vector<InputSeed>& currentCellSeed, const bounded_vector<int>& currentCellId, bounded_vector<TrackSeedN>& updatedCellSeeds, bounded_vector<int>& updatedCellsIds)
{
  auto propagator = o2::base::Propagator::Instance();

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
        const int neighbourCellId = mTimeFrame->getCellsNeighbours()[iLayer - 1][iNeighbourCell];
        const auto& neighbourCell = mTimeFrame->getCells()[iLayer - 1][neighbourCellId];
        if (neighbourCell.getSecondTrackletIndex() != currentCell.getFirstTrackletIndex()) {
          continue;
        }
        if (!currentCell.getTimeStamp().isCompatible(neighbourCell.getTimeStamp())) {
          continue;
        }
        if (currentCell.getLevel() - 1 != neighbourCell.getLevel()) {
          continue;
        }
        if (mTimeFrame->isClusterUsed(iLayer - 1, neighbourCell.getFirstClusterIndex())) {
          continue;
        }

        /// Let's start the fitting procedure
        TrackSeedN seed{currentCell};
        seed.getTimeStamp() = currentCell.getTimeStamp();
        seed.getTimeStamp() += neighbourCell.getTimeStamp();
        const auto& trHit = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer - 1)[neighbourCell.getFirstClusterIndex()];

        if (!seed.rotate(trHit.alphaTrackingFrame)) {
          continue;
        }

        if (!propagator->propagateToX(seed, trHit.xTrackingFrame, getBz(), o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, mTrkParams[iteration].CorrType)) {
          continue;
        }

        if (mTrkParams[iteration].CorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
          if (!seed.correctForMaterial(mTrkParams[iteration].LayerxX0[iLayer - 1], mTrkParams[iteration].LayerxX0[iLayer - 1] * constants::Radl * constants::Rho, true)) {
            continue;
          }
        }

        auto predChi2{seed.getPredictedChi2Quiet(trHit.positionTrackingFrame, trHit.covarianceTrackingFrame)};
        if ((predChi2 > mTrkParams[iteration].MaxChi2ClusterAttachment) || predChi2 < 0.f) {
          continue;
        }
        seed.setChi2(seed.getChi2() + predChi2);
        if (!seed.o2::track::TrackParCov::update(trHit.positionTrackingFrame, trHit.covarianceTrackingFrame)) {
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
}

template <int NLayers>
void TrackerTraits<NLayers>::findRoads(const int iteration)
{
  bounded_vector<bounded_vector<int>> firstClusters(mTrkParams[iteration].NLayers, bounded_vector<int>(mMemoryPool.get()), mMemoryPool.get());
  bounded_vector<bounded_vector<int>> sharedFirstClusters(mTrkParams[iteration].NLayers, bounded_vector<int>(mMemoryPool.get()), mMemoryPool.get());
  firstClusters.resize(mTrkParams[iteration].NLayers);
  sharedFirstClusters.resize(mTrkParams[iteration].NLayers);
  const auto propagator = o2::base::Propagator::Instance();
  const TrackingFrameInfo* tfInfos[NLayers]{};
  const Cluster* unsortedClusters[NLayers]{};
  for (int iLayer = 0; iLayer < NLayers; ++iLayer) {
    tfInfos[iLayer] = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer).data();
    unsortedClusters[iLayer] = mTimeFrame->getUnsortedClusters()[iLayer].data();
  }
  for (int startLevel{mTrkParams[iteration].CellsPerRoad()}; startLevel >= mTrkParams[iteration].CellMinimumLevel(); --startLevel) {

    auto seedFilter = [&](const auto& seed) {
      return seed.getQ2Pt() <= 1.e3 && seed.getChi2() <= mTrkParams[iteration].MaxChi2NDF * ((startLevel + 2) * 2 - 5);
    };

    bounded_vector<TrackSeedN> trackSeeds(mMemoryPool.get());
    for (int startLayer{mTrkParams[iteration].NeighboursPerRoad()}; startLayer >= startLevel - 1; --startLayer) {
      if ((mTrkParams[iteration].StartLayerMask & (1 << (startLayer + 2))) == 0) {
        continue;
      }

      bounded_vector<int> lastCellId(mMemoryPool.get()), updatedCellId(mMemoryPool.get());
      bounded_vector<TrackSeedN> lastCellSeed(mMemoryPool.get()), updatedCellSeed(mMemoryPool.get());

      processNeighbours(iteration, startLayer, startLevel, mTimeFrame->getCells()[startLayer], lastCellId, updatedCellSeed, updatedCellId);

      int level = startLevel;
      for (int iLayer{startLayer - 1}; iLayer > 0 && level > 2; --iLayer) {
        lastCellSeed.swap(updatedCellSeed);
        lastCellId.swap(updatedCellId);
        deepVectorClear(updatedCellSeed); /// tame the memory peaks
        deepVectorClear(updatedCellId);   /// tame the memory peaks
        processNeighbours(iteration, iLayer, --level, lastCellSeed, lastCellId, updatedCellSeed, updatedCellId);
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
        TrackITSExt temporaryTrack;
        bool refitSuccess = track::refitTrack<NLayers>(trackSeeds[iSeed],
                                                       temporaryTrack,
                                                       mTrkParams[iteration].MaxChi2ClusterAttachment,
                                                       mTrkParams[iteration].MaxChi2NDF,
                                                       mBz,
                                                       tfInfos,
                                                       unsortedClusters,
                                                       mTrkParams[iteration].LayerxX0.data(),
                                                       mTrkParams[iteration].LayerRadii.data(),
                                                       mTrkParams[iteration].MinPt.data(),
                                                       propagator,
                                                       mTrkParams[iteration].CorrType,
                                                       mTrkParams[iteration].ReseedIfShorter,
                                                       mTrkParams[iteration].ShiftRefToCluster,
                                                       mTrkParams[iteration].RepeatRefitOut);

        if (refitSuccess) {
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
        }
        return 0;
      };

      const int nSeeds = static_cast<int>(trackSeeds.size());
      if (mTaskArena->max_concurrency() <= 1) {
        for (int iSeed{0}; iSeed < nSeeds; ++iSeed) {
          forSeed(PassMode::OnePass{}, iSeed);
        }
      } else {
        // The double-pass allows us to avoid sizeable memory spikes
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
    });

    std::sort(tracks.begin(), tracks.end(), [](const auto& a, const auto& b) {
      return a.getChi2() < b.getChi2();
    });

    acceptTracks(iteration, tracks, firstClusters, sharedFirstClusters);
  }
  markTracks(iteration, sharedFirstClusters);
}

template <int NLayers>
void TrackerTraits<NLayers>::acceptTracks(int iteration, bounded_vector<TrackITSExt>& tracks, bounded_vector<bounded_vector<int>>& firstClusters, bounded_vector<bounded_vector<int>>& sharedFirstClusters)
{
  const float smallestROFHalf = mTimeFrame->getROFOverlapTableView().getClockLayer().mROFLength * 0.5f;
  for (auto& track : tracks) {
    int nShared = 0;
    bool isFirstShared{false};
    int firstLayer{-1}, firstCluster{-1};
    for (int iLayer{0}; iLayer < mTrkParams[iteration].NLayers; ++iLayer) {
      if (track.getClusterIndex(iLayer) == constants::UnusedIndex) {
        continue;
      }
      bool isShared = mTimeFrame->isClusterUsed(iLayer, track.getClusterIndex(iLayer));
      nShared += int(isShared);
      if (firstLayer < 0) {
        firstCluster = track.getClusterIndex(iLayer);
        isFirstShared = isShared && mTrkParams[iteration].AllowSharingFirstCluster && std::find(firstClusters[iLayer].begin(), firstClusters[iLayer].end(), firstCluster) != firstClusters[iLayer].end();
        firstLayer = iLayer;
      }
    }

    /// do not account for the first cluster in the shared clusters number if it is allowed
    if (nShared - int(isFirstShared && mTrkParams[iteration].AllowSharingFirstCluster) > mTrkParams[iteration].ClusterSharing) {
      continue;
    }

    bool firstCls{true}, nominalCompatible{true};
    TimeEstBC nominalTS, expandedTS;
    for (int iLayer{0}; iLayer < mTrkParams[iteration].NLayers; ++iLayer) {
      if (track.getClusterIndex(iLayer) == constants::UnusedIndex) {
        continue;
      }
      mTimeFrame->markUsedCluster(iLayer, track.getClusterIndex(iLayer));
      int currentROF = mTimeFrame->getClusterROF(iLayer, track.getClusterIndex(iLayer));
      const auto nominalROFTS = mTimeFrame->getROFOverlapTableView().getLayer(iLayer).getROFTimeBounds(currentROF);
      const auto expandedROFTS = mTimeFrame->getROFOverlapTableView().getLayer(iLayer).getROFTimeBounds(currentROF, true);
      if (firstCls) {
        firstCls = false;
        nominalTS = nominalROFTS;
        expandedTS = expandedROFTS;
      } else {
        if (nominalCompatible) {
          if (nominalTS.isCompatible(nominalROFTS)) {
            nominalTS += nominalROFTS;
          } else {
            nominalCompatible = false;
          }
        }
        if (!expandedTS.isCompatible(expandedROFTS)) {
          LOGP(fatal, "TS {}+/-{} are incompatible with {}+/-{}, this should not happen!", expandedROFTS.getTimeStamp(), expandedROFTS.getTimeStampError(), expandedTS.getTimeStamp(), expandedTS.getTimeStampError());
        }
        expandedTS += expandedROFTS;
      }
    }
    track.getTimeStamp() = (nominalCompatible ? nominalTS : expandedTS).makeSymmetrical();
    // this is a sanity clamp
    // we cannot be worse than the clock so we clamp to this
    if (track.getTimeStamp().getTimeStampError() > smallestROFHalf) {
      track.getTimeStamp().setTimeStampError(smallestROFHalf);
    }
    track.setUserField(0);
    track.getParamOut().setUserField(0);
    mTimeFrame->getTracks().emplace_back(track);

    if (mTrkParams[iteration].AllowSharingFirstCluster) {
      firstClusters[firstLayer].push_back(firstCluster);
      if (isFirstShared) {
        sharedFirstClusters[firstLayer].push_back(firstCluster);
      }
    }
  }
}

template <int NLayers>
void TrackerTraits<NLayers>::markTracks(int iteration, bounded_vector<bounded_vector<int>>& sharedFirstClusters)
{
  if (mTrkParams[iteration].AllowSharingFirstCluster) {
    /// Now we have to set the shared cluster flag
    for (int iLayer{0}; iLayer < mTrkParams[iteration].NLayers; ++iLayer) {
      std::sort(sharedFirstClusters[iLayer].begin(), sharedFirstClusters[iLayer].end());
    }

    for (auto& track : mTimeFrame->getTracks()) {
      int firstLayer{mTrkParams[iteration].NLayers}, firstCluster{constants::UnusedIndex};
      for (int iLayer{0}; iLayer < mTrkParams[iteration].NLayers; ++iLayer) {
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

template <int NLayers>
void TrackerTraits<NLayers>::setBz(float bz)
{
  mBz = bz;
  mTimeFrame->setBz(bz);
}

template <int NLayers>
void TrackerTraits<NLayers>::setNThreads(int n, std::shared_ptr<tbb::task_arena>& arena)
{
#if defined(OPTIMISATION_OUTPUT)
  mTaskArena = std::make_shared<tbb::task_arena>(1);
#else
  if (arena == nullptr) {
    mTaskArena = std::make_shared<tbb::task_arena>(std::abs(n));
    LOGP(info, "Setting tracker with {} threads.", n);
  } else {
    mTaskArena = arena;
  }
#endif
}

template class TrackerTraits<7>;
template void TrackerTraits<7>::processNeighbours<CellSeed>(int, int, int, const bounded_vector<CellSeed>&, const bounded_vector<int>&, bounded_vector<TrackSeed<7>>&, bounded_vector<int>&);
template void TrackerTraits<7>::processNeighbours<TrackSeed<7>>(int, int, int, const bounded_vector<TrackSeed<7>>&, const bounded_vector<int>&, bounded_vector<TrackSeed<7>>&, bounded_vector<int>&);
// ALICE3 upgrade
#ifdef ENABLE_UPGRADES
template class TrackerTraits<11>;
template void TrackerTraits<11>::processNeighbours<CellSeed>(int, int, int, const bounded_vector<CellSeed>&, const bounded_vector<int>&, bounded_vector<TrackSeed<11>>&, bounded_vector<int>&);
template void TrackerTraits<11>::processNeighbours<TrackSeed<11>>(int, int, int, const bounded_vector<TrackSeed<11>>&, const bounded_vector<int>&, bounded_vector<TrackSeed<11>>&, bounded_vector<int>&);
#endif

} // namespace o2::its
