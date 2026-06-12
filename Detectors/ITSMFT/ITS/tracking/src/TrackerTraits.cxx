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
#include <array>
#include <atomic>
#include <iterator>
#include <mutex>
#include <ranges>
#include <cmath>
#include <type_traits>
#include <vector>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>

#include "DetectorsBase/Propagator.h"
#include "GPUCommonMath.h"
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/Cell.h"
#include "ITStracking/Constants.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/LayerMask.h"
#include "ITStracking/ROFLookupTables.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/TrackFollower.h"
#include "ITStracking/TrackHelpers.h"
#include "ITStracking/Tracklet.h"

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
  const auto topology = mTimeFrame->getTrackingTopologyView();
  for (int linkId = 0; linkId < topology.nLinks; ++linkId) {
    mTimeFrame->getTracklets()[linkId].clear();
    mTimeFrame->getTrackletsLabel(linkId).clear();
    std::fill(mTimeFrame->getTrackletsLookupTable()[linkId].begin(), mTimeFrame->getTrackletsLookupTable()[linkId].end(), 0);
  }

  const Vertex diamondVert(mTrkParams[iteration].Diamond, mTrkParams[iteration].DiamondCov, 1, 1.f);
  gsl::span<const Vertex> diamondSpan(&diamondVert, 1);

  mTaskArena->execute([&] {
    auto forTracklets = [&](auto Tag, int linkId, int pivotROF, int base, int& offset) -> int {
      const auto& link = topology.getLink(linkId);
      if (!mTimeFrame->getROFMaskView().isROFEnabled(link.fromLayer, pivotROF)) {
        return 0;
      }
      gsl::span<const Vertex> primaryVertices = mTrkParams[iteration].UseDiamond ? diamondSpan : mTimeFrame->getPrimaryVertices(link.fromLayer, pivotROF);
      if (primaryVertices.empty()) {
        return 0;
      }
      const int startVtx = iVertex >= 0 ? iVertex : 0;
      const int endVtx = iVertex >= 0 ? o2::gpu::CAMath::Min(iVertex + 1, int(primaryVertices.size())) : int(primaryVertices.size());
      if (endVtx <= startVtx || (iVertex + 1) > primaryVertices.size()) {
        return 0;
      }

      const auto& rofOverlap = mTimeFrame->getROFOverlapTableView().getOverlap(link.fromLayer, link.toLayer, pivotROF);
      if (!rofOverlap.getEntries()) {
        return 0;
      }

      int localCount = 0;
      auto& tracklets = mTimeFrame->getTracklets()[linkId];
      auto layer0 = mTimeFrame->getClustersOnLayer(pivotROF, link.fromLayer);
      if (layer0.empty()) {
        return 0;
      }

      const float meanDeltaR = mTrkParams[iteration].LayerRadii[link.toLayer] - mTrkParams[iteration].LayerRadii[link.fromLayer];
      const float phiCut = mTimeFrame->getLinkPhiCut(linkId);
      const float msAngle = mTimeFrame->getLinkMSAngle(linkId);

      for (int iCluster = 0; iCluster < int(layer0.size()); ++iCluster) {
        const Cluster& currentCluster = layer0[iCluster];
        const int currentSortedIndex = mTimeFrame->getSortedIndex(pivotROF, link.fromLayer, iCluster);
        if (mTimeFrame->isClusterUsed(link.fromLayer, currentCluster.clusterId)) {
          continue;
        }
        const float inverseR0 = 1.f / currentCluster.radius;

        for (int iV = startVtx; iV < endVtx; ++iV) {
          const auto& pv = primaryVertices[iV];
          if (!mTimeFrame->getROFVertexLookupTableView().isVertexCompatible(link.fromLayer, pivotROF, pv)) {
            continue;
          }
          if (pv.isFlagSet(Vertex::Flags::UPCMode) != mTrkParams[iteration].PassFlags[IterationStep::SelectUPCVertices]) {
            continue;
          }
          const float resolution = o2::gpu::CAMath::Sqrt(math_utils::Sq(mTimeFrame->getPositionResolution(link.fromLayer)) + math_utils::Sq(mTrkParams[iteration].PVres) / float(pv.getNContributors()));
          const float tanLambda = (currentCluster.zCoordinate - pv.getZ()) * inverseR0;
          const float zAtRmin = tanLambda * (mTimeFrame->getMinR(link.toLayer) - currentCluster.radius) + currentCluster.zCoordinate;
          const float zAtRmax = tanLambda * (mTimeFrame->getMaxR(link.toLayer) - currentCluster.radius) + currentCluster.zCoordinate;
          const float sqInvDeltaZ0 = 1.f / (math_utils::Sq(currentCluster.zCoordinate - pv.getZ()) + constants::Tolerance);
          const float sigmaZ = o2::gpu::CAMath::Sqrt((math_utils::Sq(resolution) * math_utils::Sq(tanLambda) * ((math_utils::Sq(inverseR0) + sqInvDeltaZ0) * math_utils::Sq(meanDeltaR) + 1.f)) + math_utils::Sq(meanDeltaR * msAngle));
          const auto bins = o2::its::getBinsRect(currentCluster, link.toLayer, zAtRmin, zAtRmax,
                                                 sigmaZ * mTrkParams[iteration].NSigmaCut, phiCut,
                                                 mTimeFrame->getIndexTableUtils());
          if (bins.x < 0) {
            continue;
          }
          int phiBinsNum = bins.w - bins.y + 1;
          if (phiBinsNum < 0) {
            phiBinsNum += mTrkParams[iteration].PhiBins;
          }

          for (int targetROF = rofOverlap.getFirstEntry(); targetROF < rofOverlap.getEntriesBound(); ++targetROF) {
            if (!mTimeFrame->getROFMaskView().isROFEnabled(link.toLayer, targetROF)) {
              continue;
            }
            auto layer1 = mTimeFrame->getClustersOnLayer(targetROF, link.toLayer);
            if (layer1.empty()) {
              continue;
            }
            const auto ts = mTimeFrame->getROFOverlapTableView().getTimeStamp(link.fromLayer, pivotROF, link.toLayer, targetROF);
            if (!ts.isCompatible(pv.getTimeStamp())) {
              continue;
            }
            const auto& targetIndexTable = mTimeFrame->getIndexTable(targetROF, link.toLayer);
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
                if (mTimeFrame->isClusterUsed(link.toLayer, nextCluster.clusterId)) {
                  continue;
                }
                const float deltaZ = o2::gpu::CAMath::Abs((tanLambda * (nextCluster.radius - currentCluster.radius)) + currentCluster.zCoordinate - nextCluster.zCoordinate);

                if (deltaZ / sigmaZ < mTrkParams[iteration].NSigmaCut &&
                    math_utils::isPhiDifferenceBelow(currentCluster.phi, nextCluster.phi, phiCut)) {
                  const float phi{o2::gpu::CAMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
                  const float tanL = (currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius);
                  if constexpr (decltype(Tag)::value == PassMode::OnePass::value) {
                    tracklets.emplace_back(currentSortedIndex, mTimeFrame->getSortedIndex(targetROF, link.toLayer, iNext), tanL, phi, ts);
                  } else if constexpr (decltype(Tag)::value == PassMode::TwoPassCount::value) {
                    ++localCount;
                  } else if constexpr (decltype(Tag)::value == PassMode::TwoPassInsert::value) {
                    const int idx = base + offset++;
                    tracklets[idx] = Tracklet(currentSortedIndex, mTimeFrame->getSortedIndex(targetROF, link.toLayer, iNext), tanL, phi, ts);
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
      for (int linkId{0}; linkId < topology.nLinks; ++linkId) {
        const int fromLayer = topology.getLink(linkId).fromLayer;
        const int startROF = 0, endROF = mTimeFrame->getROFOverlapTableView().getLayer(fromLayer).mNROFsTF;
        for (int pivotROF{startROF}; pivotROF < endROF; ++pivotROF) {
          forTracklets(PassMode::OnePass{}, linkId, pivotROF, 0, dummy);
        }
      }
    } else {
      tbb::parallel_for(0, static_cast<int>(topology.nLinks), [&](const int linkId) {
        const int fromLayer = topology.getLink(linkId).fromLayer;
        const int startROF = 0, endROF = mTimeFrame->getROFOverlapTableView().getLayer(fromLayer).mNROFsTF;
        bounded_vector<int> perROFCount((endROF - startROF) + 1, mMemoryPool.get());
        tbb::parallel_for(startROF, endROF, [&](const int pivotROF) {
          perROFCount[pivotROF - startROF] = forTracklets(PassMode::TwoPassCount{}, linkId, pivotROF, 0, dummy);
        });
        std::exclusive_scan(perROFCount.begin(), perROFCount.end(), perROFCount.begin(), 0);
        const int nTracklets = perROFCount.back();
        mTimeFrame->getTracklets()[linkId].resize(nTracklets);
        if (nTracklets == 0) {
          return;
        }
        tbb::parallel_for(startROF, endROF, [&](const int pivotROF) {
          int baseIdx = perROFCount[pivotROF - startROF];
          if (baseIdx == perROFCount[pivotROF + 1 - startROF]) {
            return;
          }
          int localIdx = 0;
          forTracklets(PassMode::TwoPassInsert{}, linkId, pivotROF, baseIdx, localIdx);
        });
      });
    }

    tbb::parallel_for(0, static_cast<int>(topology.nLinks), [&](const int linkId) {
      /// Sort tracklets & remove duplicates
      // duplicates can exist simply since we evaluate per vertex
      auto& trkl{mTimeFrame->getTracklets()[linkId]};
      std::sort(trkl.begin(), trkl.end());
      trkl.erase(std::unique(trkl.begin(), trkl.end()), trkl.end());
      trkl.shrink_to_fit();
      auto& lut{mTimeFrame->getTrackletsLookupTable()[linkId]};
      if (!trkl.empty()) {
        for (const auto& tkl : trkl) {
          lut[tkl.firstClusterIndex + 1]++;
        }
        std::inclusive_scan(lut.begin(), lut.end(), lut.begin());
      }
    });

    /// Create tracklets labels
    if (mTimeFrame->hasMCinformation() && mTrkParams[iteration].CreateArtefactLabels) {
      tbb::parallel_for(0, static_cast<int>(topology.nLinks), [&](const int linkId) {
        const auto& link = topology.getLink(linkId);
        for (auto& trk : mTimeFrame->getTracklets()[linkId]) {
          MCCompLabel label;
          int currentId{mTimeFrame->getClusters()[link.fromLayer][trk.firstClusterIndex].clusterId};
          int nextId{mTimeFrame->getClusters()[link.toLayer][trk.secondClusterIndex].clusterId};
          for (const auto& lab1 : mTimeFrame->getClusterLabels(link.fromLayer, currentId)) {
            for (const auto& lab2 : mTimeFrame->getClusterLabels(link.toLayer, nextId)) {
              if (lab1 == lab2 && lab1.isValid()) {
                label = lab1;
                break;
              }
            }
            if (label.isValid()) {
              break;
            }
          }
          mTimeFrame->getTrackletsLabel(linkId).emplace_back(label);
        }
      });
    }
  });
}

template <int NLayers>
void TrackerTraits<NLayers>::computeLayerCells(const int iteration)
{
  const auto topology = mTimeFrame->getTrackingTopologyView();
  for (int cellTopologyId = 0; cellTopologyId < topology.nCells; ++cellTopologyId) {
    deepVectorClear(mTimeFrame->getCells()[cellTopologyId]);
    deepVectorClear(mTimeFrame->getCellsLookupTable()[cellTopologyId]);
    if (mTimeFrame->hasMCinformation() && mTrkParams[iteration].CreateArtefactLabels) {
      deepVectorClear(mTimeFrame->getCellsLabel(cellTopologyId));
    }
  }

  mTaskArena->execute([&] {
    auto forTrackletCells = [&](auto Tag, int cellTopologyId, bounded_vector<CellSeed>& layerCells, int iTracklet, int offset = 0) -> int {
      const auto& cellTopology = topology.getCell(cellTopologyId);
      const auto& firstLink = topology.getLink(cellTopology.firstLink);
      const auto& secondLink = topology.getLink(cellTopology.secondLink);
      const Tracklet& currentTracklet{mTimeFrame->getTracklets()[cellTopology.firstLink][iTracklet]};
      const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
      const int nextLayerFirstTrackletIndex{mTimeFrame->getTrackletsLookupTable()[cellTopology.secondLink][nextLayerClusterIndex]};
      const int nextLayerLastTrackletIndex{mTimeFrame->getTrackletsLookupTable()[cellTopology.secondLink][nextLayerClusterIndex + 1]};
      int foundCells{0};
      for (int iNextTracklet{nextLayerFirstTrackletIndex}; iNextTracklet < nextLayerLastTrackletIndex; ++iNextTracklet) {
        const Tracklet& nextTracklet{mTimeFrame->getTracklets()[cellTopology.secondLink][iNextTracklet]};
        if (nextTracklet.firstClusterIndex != nextLayerClusterIndex) {
          break;
        }
        if (!currentTracklet.getTimeStamp().isCompatible(nextTracklet.getTimeStamp())) {
          continue;
        }

        const float deltaTanLambdaSigma = std::abs(currentTracklet.tanLambda - nextTracklet.tanLambda) / mTrkParams[iteration].CellDeltaTanLambdaSigma;
        if (deltaTanLambdaSigma < mTrkParams[iteration].NSigmaCut) {

          /// Track seed preparation. Clusters are numbered progressively from the innermost going outward.
          const int clusId[3]{
            mTimeFrame->getClusters()[firstLink.fromLayer][currentTracklet.firstClusterIndex].clusterId,
            mTimeFrame->getClusters()[firstLink.toLayer][nextTracklet.firstClusterIndex].clusterId,
            mTimeFrame->getClusters()[secondLink.toLayer][nextTracklet.secondClusterIndex].clusterId};
          const int hitLayers[3]{firstLink.fromLayer, firstLink.toLayer, secondLink.toLayer};
          const auto& cluster1_glo = mTimeFrame->getUnsortedClusters()[firstLink.fromLayer][clusId[0]];
          const auto& cluster2_glo = mTimeFrame->getUnsortedClusters()[firstLink.toLayer][clusId[1]];
          const auto& cluster3_tf = mTimeFrame->getTrackingFrameInfoOnLayer(secondLink.toLayer)[clusId[2]];
          auto track{o2::its::track::buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_tf, mBz)};

          float chi2{0.f};
          bool good{false};
          for (int iC{2}; iC--;) {
            const int hitLayer = hitLayers[iC];
            const TrackingFrameInfo& trackingHit = mTimeFrame->getTrackingFrameInfoOnLayer(hitLayer)[clusId[iC]];

            if (!track.rotate(trackingHit.alphaTrackingFrame)) {
              break;
            }

            if (!track.propagateTo(trackingHit.xTrackingFrame, getBz())) {
              break;
            }

            if (!track.correctForMaterial(mTrkParams[iteration].LayerxX0[hitLayer], mTrkParams[iteration].LayerxX0[hitLayer] * constants::Radl * constants::Rho, true)) {
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
              layerCells.emplace_back(cellTopology.hitLayerMask, clusId[0], clusId[1], clusId[2], iTracklet, iNextTracklet, track, chi2, ts);
              ++foundCells;
            } else if constexpr (decltype(Tag)::value == PassMode::TwoPassCount::value) {
              ++foundCells;
            } else if constexpr (decltype(Tag)::value == PassMode::TwoPassInsert::value) {
              layerCells[offset++] = CellSeed(cellTopology.hitLayerMask, clusId[0], clusId[1], clusId[2], iTracklet, iNextTracklet, track, chi2, ts);
              ++foundCells;
            } else {
              static_assert(false, "Unknown mode!");
            }
          }
        }
      }
      return foundCells;
    };

    for (int cellTopologyId = 0; cellTopologyId < topology.nCells; ++cellTopologyId) {
      const auto& cellTopology = topology.getCell(cellTopologyId);
      if (mTimeFrame->getTracklets()[cellTopology.firstLink].empty() ||
          mTimeFrame->getTracklets()[cellTopology.secondLink].empty()) {
        continue;
      }

      auto& layerCells = mTimeFrame->getCells()[cellTopologyId];
      const int currentLayerTrackletsNum{static_cast<int>(mTimeFrame->getTracklets()[cellTopology.firstLink].size())};
      bounded_vector<int> perTrackletCount(currentLayerTrackletsNum + 1, 0, mMemoryPool.get());
      if (mTaskArena->max_concurrency() <= 1) {
        for (int iTracklet{0}; iTracklet < currentLayerTrackletsNum; ++iTracklet) {
          perTrackletCount[iTracklet] = forTrackletCells(PassMode::OnePass{}, cellTopologyId, layerCells, iTracklet);
        }
        std::exclusive_scan(perTrackletCount.begin(), perTrackletCount.end(), perTrackletCount.begin(), 0);
      } else {
        tbb::parallel_for(0, currentLayerTrackletsNum, [&](const int iTracklet) {
          perTrackletCount[iTracklet] = forTrackletCells(PassMode::TwoPassCount{}, cellTopologyId, layerCells, iTracklet);
        });

        std::exclusive_scan(perTrackletCount.begin(), perTrackletCount.end(), perTrackletCount.begin(), 0);
        auto totalCells{perTrackletCount.back()};
        if (totalCells == 0) {
          auto& lut = mTimeFrame->getCellsLookupTable()[cellTopologyId];
          lut.resize(currentLayerTrackletsNum + 1);
          std::fill(lut.begin(), lut.end(), 0);
          continue;
        }
        layerCells.resize(totalCells);

        tbb::parallel_for(0, currentLayerTrackletsNum, [&](const int iTracklet) {
          int offset = perTrackletCount[iTracklet];
          if (offset == perTrackletCount[iTracklet + 1]) {
            return;
          }
          forTrackletCells(PassMode::TwoPassInsert{}, cellTopologyId, layerCells, iTracklet, offset);
        });
      }

      auto& lut = mTimeFrame->getCellsLookupTable()[cellTopologyId];
      lut.resize(currentLayerTrackletsNum + 1);
      std::copy_n(perTrackletCount.begin(), currentLayerTrackletsNum + 1, lut.begin());

      if (mTimeFrame->hasMCinformation() && mTrkParams[iteration].CreateArtefactLabels) {
        auto& labels = mTimeFrame->getCellsLabel(cellTopologyId);
        labels.reserve(layerCells.size());
        for (const auto& cell : layerCells) {
          MCCompLabel currentLab{mTimeFrame->getTrackletsLabel(cellTopology.firstLink)[cell.getFirstTrackletIndex()]};
          MCCompLabel nextLab{mTimeFrame->getTrackletsLabel(cellTopology.secondLink)[cell.getSecondTrackletIndex()]};
          labels.emplace_back(currentLab == nextLab ? currentLab : MCCompLabel());
        }
      }
    }
  });

  for (int linkId = 0; linkId < topology.nLinks; ++linkId) {
    deepVectorClear(mTimeFrame->getTracklets()[linkId]);
    deepVectorClear(mTimeFrame->getTrackletsLabel(linkId));
  }
}

template <int NLayers>
void TrackerTraits<NLayers>::findCellsNeighbours(const int iteration)
{
  const auto topology = mTimeFrame->getTrackingTopologyView();
  mTaskArena->execute([&] {
    std::vector<bounded_vector<CellNeighbour>> cellsNeighboursByTarget;
    cellsNeighboursByTarget.reserve(topology.nCells);
    for (int cellTopologyId{0}; cellTopologyId < topology.nCells; ++cellTopologyId) {
      deepVectorClear(mTimeFrame->getCellsNeighbours()[cellTopologyId]);
      deepVectorClear(mTimeFrame->getCellsNeighboursTopology()[cellTopologyId]);
      deepVectorClear(mTimeFrame->getCellsNeighboursLUT()[cellTopologyId]);
      cellsNeighboursByTarget.emplace_back(mMemoryPool.get());
    }

    for (int outerLayer{0}; outerLayer < NLayers; ++outerLayer) {
      for (int cellTopologyId{0}; cellTopologyId < topology.nCells; ++cellTopologyId) {
        const auto& cellTopology = topology.getCell(cellTopologyId);
        if (cellTopology.hitLayerMask.last() != outerLayer ||
            mTimeFrame->getCells()[cellTopologyId].empty()) {
          continue;
        }
        const auto successors = topology.getCellsStartingWithLink(cellTopology.secondLink);
        if (!successors.getEntries()) {
          continue;
        }

        tbb::enumerable_thread_specific<bounded_vector<CellNeighbour>> sourceNeighbours([&]() { return bounded_vector<CellNeighbour>{mMemoryPool.get()}; });
        tbb::parallel_for(0, static_cast<int>(mTimeFrame->getCells()[cellTopologyId].size()), [&](const int iCell) {
          auto& localNeighbours = sourceNeighbours.local();
          const auto& currentCellSeed{mTimeFrame->getCells()[cellTopologyId][iCell]};
          const int nextLayerTrackletIndex{currentCellSeed.getSecondTrackletIndex()};
          for (int iSuccessor{0}; iSuccessor < successors.getEntries(); ++iSuccessor) {
            const int nextCellTopologyId = topology.cellsByFirstLink[successors.getFirstEntry() + iSuccessor];
            if (mTimeFrame->getCells()[nextCellTopologyId].empty() ||
                mTimeFrame->getCellsLookupTable()[nextCellTopologyId].empty()) {
              continue;
            }
            const auto& nextCellLUT = mTimeFrame->getCellsLookupTable()[nextCellTopologyId];
            if (nextLayerTrackletIndex + 1 >= static_cast<int>(nextCellLUT.size())) {
              continue;
            }
            const int nextLayerFirstCellIndex{nextCellLUT[nextLayerTrackletIndex]};
            const int nextLayerLastCellIndex{nextCellLUT[nextLayerTrackletIndex + 1]};
            for (int iNextCell{nextLayerFirstCellIndex}; iNextCell < nextLayerLastCellIndex; ++iNextCell) {
              const auto& nextCellSeedRef{mTimeFrame->getCells()[nextCellTopologyId][iNextCell]};
              if (nextCellSeedRef.getFirstTrackletIndex() != nextLayerTrackletIndex || !currentCellSeed.getTimeStamp().isCompatible(nextCellSeedRef.getTimeStamp())) {
                break;
              }

              auto nextCellSeed{mTimeFrame->getCells()[nextCellTopologyId][iNextCell]}; /// copy
              if (!nextCellSeed.rotate(currentCellSeed.getAlpha()) ||
                  !nextCellSeed.propagateTo(currentCellSeed.getX(), getBz())) {
                continue;
              }

              float chi2 = currentCellSeed.getPredictedChi2(nextCellSeed);
              if (chi2 > mTrkParams[iteration].MaxChi2ClusterAttachment) {
                continue;
              }

              const int nextLevel = currentCellSeed.getLevel() + 1;
              localNeighbours.emplace_back(cellTopologyId, iCell, nextCellTopologyId, iNextCell, nextLevel);
            }
          }
        });

        bounded_vector<size_t> count(topology.nCells, 0, mMemoryPool.get());
        for (const auto& localNeighbours : sourceNeighbours) {
          for (const auto& neigh : localNeighbours) {
            ++count[neigh.nextCellTopology];
          }
        }
        for (size_t i{0}; i < topology.nCells; ++i) {
          cellsNeighboursByTarget[i].reserve(count[i]);
        }
        for (const auto& localNeighbours : sourceNeighbours) {
          for (const auto& neigh : localNeighbours) {
            cellsNeighboursByTarget[neigh.nextCellTopology].emplace_back(neigh);
            if (neigh.level > mTimeFrame->getCells()[neigh.nextCellTopology][neigh.nextCell].getLevel()) {
              mTimeFrame->getCells()[neigh.nextCellTopology][neigh.nextCell].setLevel(neigh.level);
            }
          }
        }
      }
    }

    for (int cellTopologyId{0}; cellTopologyId < topology.nCells; ++cellTopologyId) {
      auto& cellsNeighbours = cellsNeighboursByTarget[cellTopologyId];
      if (cellsNeighbours.empty()) {
        continue;
      }

      std::sort(cellsNeighbours.begin(), cellsNeighbours.end(), [](const auto& a, const auto& b) {
        return a.nextCell < b.nextCell;
      });

      auto& cellsNeighbourLUT = mTimeFrame->getCellsNeighboursLUT()[cellTopologyId];
      cellsNeighbourLUT.assign(mTimeFrame->getCells()[cellTopologyId].size(), 0);
      for (const auto& neigh : cellsNeighbours) {
        ++cellsNeighbourLUT[neigh.nextCell];
      }
      std::inclusive_scan(cellsNeighbourLUT.begin(), cellsNeighbourLUT.end(), cellsNeighbourLUT.begin());

      mTimeFrame->getCellsNeighbours()[cellTopologyId].reserve(cellsNeighbours.size());
      mTimeFrame->getCellsNeighboursTopology()[cellTopologyId].reserve(cellsNeighbours.size());
      std::ranges::transform(cellsNeighbours, std::back_inserter(mTimeFrame->getCellsNeighbours()[cellTopologyId]), [](const auto& neigh) { return neigh.cell; });
      std::ranges::transform(cellsNeighbours, std::back_inserter(mTimeFrame->getCellsNeighboursTopology()[cellTopologyId]), [](const auto& neigh) { return neigh.cellTopology; });
    }

    // clean up LUTs
    for (auto& cellLUT : mTimeFrame->getCellsLookupTable()) {
      deepVectorClear(cellLUT);
    }
  });
}

template <int NLayers>
template <typename InputSeed>
void TrackerTraits<NLayers>::processNeighbours(int iteration, int defaultCellTopologyId, int iLevel, const bounded_vector<InputSeed>& currentCellSeed, const bounded_vector<int>& currentCellId, const bounded_vector<int>& currentCellTopologyId, bounded_vector<TrackSeedN>& updatedCellSeeds, bounded_vector<int>& updatedCellsIds, bounded_vector<int>& updatedCellsTopologyIds)
{
  auto propagator = o2::base::Propagator::Instance();

  mTaskArena->execute([&] {
    auto forCellNeighbours = [&](auto Tag, int iCell, int offset = 0) -> int {
      const auto& currentCell{currentCellSeed[iCell]};
      const int cellTopologyId = currentCellTopologyId.empty() ? defaultCellTopologyId : currentCellTopologyId[iCell];

      if constexpr (decltype(Tag)::value != PassMode::TwoPassInsert::value) {
        if (currentCell.getLevel() != iLevel) {
          return 0;
        }
        if (currentCellId.empty()) {
          for (int layer = 0; layer < NLayers; ++layer) {
            const int clusterIndex = currentCell.getCluster(layer);
            if (clusterIndex != constants::UnusedIndex && mTimeFrame->isClusterUsed(layer, clusterIndex)) {
              return 0; /// this we do only on the first iteration, hence the check on currentCellId
            }
          }
        }
      }

      const int cellId = currentCellId.empty() ? iCell : currentCellId[iCell];
      if (cellTopologyId < 0 || mTimeFrame->getCellsNeighboursLUT()[cellTopologyId].empty()) {
        return 0;
      }
      const int startNeighbourId{cellId ? mTimeFrame->getCellsNeighboursLUT()[cellTopologyId][cellId - 1] : 0};
      const int endNeighbourId{mTimeFrame->getCellsNeighboursLUT()[cellTopologyId][cellId]};
      int foundSeeds{0};
      for (int iNeighbourCell{startNeighbourId}; iNeighbourCell < endNeighbourId; ++iNeighbourCell) {
        const int neighbourCellTopologyId = mTimeFrame->getCellsNeighboursTopology()[cellTopologyId][iNeighbourCell];
        const int neighbourCellId = mTimeFrame->getCellsNeighbours()[cellTopologyId][iNeighbourCell];
        const auto& neighbourCell = mTimeFrame->getCells()[neighbourCellTopologyId][neighbourCellId];
        if (neighbourCell.getSecondTrackletIndex() != currentCell.getFirstTrackletIndex()) {
          continue;
        }
        if (!currentCell.getTimeStamp().isCompatible(neighbourCell.getTimeStamp())) {
          continue;
        }
        if (currentCell.getLevel() - 1 != neighbourCell.getLevel()) {
          continue;
        }
        const int neighbourLayer = neighbourCell.getInnerLayer();
        const int neighbourCluster = neighbourCell.getFirstClusterIndex();
        if (mTimeFrame->isClusterUsed(neighbourLayer, neighbourCluster)) {
          continue;
        }

        /// Let's start the fitting procedure
        TrackSeedN seed{currentCell};
        seed.getTimeStamp() = currentCell.getTimeStamp();
        seed.getTimeStamp() += neighbourCell.getTimeStamp();
        const auto& trHit = mTimeFrame->getTrackingFrameInfoOnLayer(neighbourLayer)[neighbourCluster];

        if (!seed.rotate(trHit.alphaTrackingFrame)) {
          continue;
        }

        if (!propagator->propagateToX(seed, trHit.xTrackingFrame, getBz(), o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, mTrkParams[iteration].CorrType)) {
          continue;
        }

        if (mTrkParams[iteration].CorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
          if (!seed.correctForMaterial(mTrkParams[iteration].LayerxX0[neighbourLayer], mTrkParams[iteration].LayerxX0[neighbourLayer] * constants::Radl * constants::Rho, true)) {
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
          seed.getClusters()[neighbourLayer] = neighbourCluster;
          auto mask = seed.getHitLayerMask();
          mask.set(neighbourLayer);
          seed.setHitLayerMask(mask);
          seed.setLevel(neighbourCell.getLevel());
          seed.setFirstTrackletIndex(neighbourCell.getFirstTrackletIndex());
          seed.setSecondTrackletIndex(neighbourCell.getSecondTrackletIndex());
        }

        if constexpr (decltype(Tag)::value == PassMode::OnePass::value) {
          updatedCellSeeds.push_back(seed);
          updatedCellsIds.push_back(neighbourCellId);
          updatedCellsTopologyIds.push_back(neighbourCellTopologyId);
        } else if constexpr (decltype(Tag)::value == PassMode::TwoPassCount::value) {
          ++foundSeeds;
        } else if constexpr (decltype(Tag)::value == PassMode::TwoPassInsert::value) {
          updatedCellSeeds[offset] = seed;
          updatedCellsIds[offset] = neighbourCellId;
          updatedCellsTopologyIds[offset++] = neighbourCellTopologyId;
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
      updatedCellsTopologyIds.resize(totalNeighbours);

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
bool TrackerTraits<NLayers>::finaliseTrackSeed(const TrackSeedN& seed,
                                               TrackITSExt& track,
                                               const int iteration,
                                               const TrackingFrameInfo* const* tfInfos,
                                               const Cluster* const* unsortedClusters,
                                               const o2::base::Propagator* propagator)
{
  const auto& trkParams = mTrkParams[iteration];
  const track::TrackFitContext<NLayers> fitCtx{
    tfInfos, trkParams.LayerxX0.data(), trkParams.NLayers, mBz,
    trkParams.MaxChi2ClusterAttachment, trkParams.MaxChi2NDF,
    propagator, trkParams.CorrType, trkParams.ShiftRefToCluster, trkParams.RepeatRefitOut};
  TrackITSInternal<NLayers> internalTrack;
  if (!track::refitTrackSeed<NLayers>(seed,
                                      internalTrack,
                                      fitCtx,
                                      unsortedClusters,
                                      trkParams.LayerRadii.data(),
                                      trkParams.MinPt.data(),
                                      trkParams.ReseedIfShorter)) {
    return false;
  }
  const auto passesFinalLengthCut = [&trkParams](const TrackITSExt& candidate) {
    LayerMask hitLayerMask{0};
    for (int iLayer{0}; iLayer < trkParams.NLayers; ++iLayer) {
      if (candidate.getClusterIndex(iLayer) != constants::UnusedIndex) {
        hitLayerMask.set(iLayer);
      }
    }
    return track::TrackSeedSelector<NLayers>::getEffectiveTrackLength(hitLayerMask, trkParams.InactiveLayerMask) >= trkParams.MinTrackLength;
  };

  const bool extendTop = trkParams.PassFlags[IterationStep::TrackFollowerTop];
  const bool extendBot = trkParams.PassFlags[IterationStep::TrackFollowerBot];
  if (!extendTop && !extendBot) {
    track = makeTrackITSExt(internalTrack);
    return passesFinalLengthCut(track);
  }

  const int maxHypotheses = std::max(1, trkParams.TrackFollowerMaxHypotheses);
  TrackFollowerScratch scratch{mMemoryPool.get()};
  if (static_cast<int>(scratch.activeHypotheses.size()) < maxHypotheses) {
    scratch.activeHypotheses.resize(maxHypotheses);
  }
  if (static_cast<int>(scratch.nextHypotheses.size()) < maxHypotheses) {
    scratch.nextHypotheses.resize(maxHypotheses);
  }

  const Cluster* clustersPtrs[NLayers]{};
  const unsigned char* usedClustersPtrs[NLayers]{};
  const int* clustersIndexTablesPtrs[NLayers]{};
  const int* rofClustersPtrs[NLayers]{};
  for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
    clustersPtrs[iLayer] = mTimeFrame->getClusters()[iLayer].data();
    usedClustersPtrs[iLayer] = mTimeFrame->getUsedClusters(iLayer).data();
    clustersIndexTablesPtrs[iLayer] = mTimeFrame->getIndexTable(0, iLayer).data();
    rofClustersPtrs[iLayer] = mTimeFrame->getROFrameClusters(iLayer).data();
  }
  const TrackFollowContext<NLayers> followCtx{
    &mTimeFrame->getIndexTableUtils(),
    mTimeFrame->getROFMaskView(),
    mTimeFrame->getROFOverlapTableView(),
    clustersPtrs, usedClustersPtrs, clustersIndexTablesPtrs, rofClustersPtrs,
    trkParams.LayerRadii.data(), trkParams.PhiBins, maxHypotheses,
    trkParams.TrackFollowerNSigmaCutPhi, trkParams.TrackFollowerNSigmaCutZ};

  const auto backup = internalTrack;
  auto best = internalTrack;
  uint32_t bestDiff{0};
  auto followDirection = [&](TrackITSInternal<NLayers>& candidate, bool outward) {
    const TrackExtensionHypothesis<NLayers> startHypothesis{candidate, outward};
    TrackExtensionHypothesis<NLayers> bestHypothesis;
    if (!followTrackExtensionDirection<NLayers>(startHypothesis, fitCtx, followCtx, outward,
                                                scratch.activeHypotheses.data(),
                                                scratch.nextHypotheses.data(),
                                                bestHypothesis)) {
      return false;
    }
    updateTrackFromExtensionHypothesis(bestHypothesis, outward, trkParams.NLayers, candidate);
    return true;
  };
  TrackExtensionBestTrial<NLayers> bestTrial{backup.getPattern(), fitCtx};
  followTrackExtensionBranches(backup, extendTop, extendBot, trkParams.NLayers, followDirection, bestTrial, best, bestDiff);

  track = makeTrackITSExt(best);
  if (bestDiff) {
    track.setExtendedLayerPattern<NLayers>(bestDiff);
  }
  return passesFinalLengthCut(track);
}

template <int NLayers>
void TrackerTraits<NLayers>::findRoads(const int iteration)
{
  bounded_vector<bounded_vector<int>> firstClusters(mTrkParams[iteration].NLayers, bounded_vector<int>(mMemoryPool.get()), mMemoryPool.get());
  firstClusters.resize(mTrkParams[iteration].NLayers);
  const auto propagator = o2::base::Propagator::Instance();
  const TrackingFrameInfo* tfInfos[NLayers]{};
  const Cluster* unsortedClusters[NLayers]{};
  for (int iLayer = 0; iLayer < NLayers; ++iLayer) {
    tfInfos[iLayer] = mTimeFrame->getTrackingFrameInfoOnLayer(iLayer).data();
    unsortedClusters[iLayer] = mTimeFrame->getUnsortedClusters()[iLayer].data();
  }
  const auto topology = mTimeFrame->getTrackingTopologyView();
  for (int startLevel{mTrkParams[iteration].CellsPerRoad()}; startLevel >= mTrkParams[iteration].CellMinimumLevel(); --startLevel) {

    const track::TrackSeedSelector<NLayers> seedFilter{constants::MaxTrackSeedQ2Pt, mTrkParams[iteration].MaxChi2NDF, startLevel, mTrkParams[iteration].MaxHoles, mTrkParams[iteration].getMinSeedingClusters(), mTrkParams[iteration].HoleLayerMask, mTrkParams[iteration].getNonSeedingLayerMask()};

    bounded_vector<TrackSeedN> trackSeeds(mMemoryPool.get());
    for (int startCellTopologyId{0}; startCellTopologyId < topology.nCells; ++startCellTopologyId) {
      const int startLayer = topology.getCell(startCellTopologyId).hitLayerMask.last();
      if (!(mTrkParams[iteration].StartLayerMask.has(startLayer)) || mTimeFrame->getCells()[startCellTopologyId].empty()) {
        continue;
      }

      bounded_vector<int> lastCellId(mMemoryPool.get()), updatedCellId(mMemoryPool.get());
      bounded_vector<int> lastCellTopologyId(mMemoryPool.get()), updatedCellTopologyId(mMemoryPool.get());
      bounded_vector<TrackSeedN> lastCellSeed(mMemoryPool.get()), updatedCellSeed(mMemoryPool.get());

      processNeighbours(iteration, startCellTopologyId, startLevel, mTimeFrame->getCells()[startCellTopologyId], lastCellId, lastCellTopologyId, updatedCellSeed, updatedCellId, updatedCellTopologyId);

      int level = startLevel;
      while (level > 2 && !updatedCellSeed.empty()) {
        lastCellSeed.swap(updatedCellSeed);
        lastCellId.swap(updatedCellId);
        lastCellTopologyId.swap(updatedCellTopologyId);
        deepVectorClear(updatedCellSeed); /// tame the memory peaks
        deepVectorClear(updatedCellId);   /// tame the memory peaks
        deepVectorClear(updatedCellTopologyId);
        processNeighbours(iteration, constants::UnusedIndex, --level, lastCellSeed, lastCellId, lastCellTopologyId, updatedCellSeed, updatedCellId, updatedCellTopologyId);
      }
      deepVectorClear(lastCellId);         /// tame the memory peaks
      deepVectorClear(lastCellTopologyId); /// tame the memory peaks
      deepVectorClear(lastCellSeed);       /// tame the memory peaks

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
      const int nSeeds = static_cast<int>(trackSeeds.size());
      const int nWorkers = std::min(static_cast<int>(mTaskArena->max_concurrency()), nSeeds);
      const int chunkSize = std::min(nSeeds, std::clamp(nSeeds / (16 * nWorkers), 256, 4096));
      std::atomic<int> nextSeed{0};
      std::mutex tracksMutex;
      tbb::parallel_for(0, nWorkers, [&](const int) {
        bounded_vector<TrackITSExt> localTracks(mMemoryPool.get());
        localTracks.reserve(chunkSize);
        while (true) {
          const int firstSeed = nextSeed.fetch_add(chunkSize, std::memory_order_relaxed);
          if (firstSeed >= nSeeds) {
            break;
          }
          const int lastSeed = std::min(firstSeed + chunkSize, nSeeds);
          for (int iSeed{firstSeed}; iSeed < lastSeed; ++iSeed) {
            TrackITSExt temporaryTrack;
            if (finaliseTrackSeed(trackSeeds[iSeed], temporaryTrack, iteration, tfInfos, unsortedClusters, propagator)) {
              localTracks.push_back(temporaryTrack);
            }
          }
          if (!localTracks.empty()) {
            std::lock_guard lock{tracksMutex};
            tracks.insert(tracks.end(), std::make_move_iterator(localTracks.begin()), std::make_move_iterator(localTracks.end()));
            localTracks.clear();
          }
        }
        deepVectorClear(localTracks);
      });

      deepVectorClear(trackSeeds);
    });

    std::sort(tracks.begin(), tracks.end(), [](const auto& a, const auto& b) {
      return track::isBetter(a, b);
    });

    acceptTracks(iteration, tracks, firstClusters);
  }
  markTracks(iteration);
}

template <int NLayers>
void TrackerTraits<NLayers>::acceptTracks(int iteration,
                                          bounded_vector<TrackITSExt>& tracks,
                                          bounded_vector<bounded_vector<int>>& firstClusters)
{
  auto& trks = mTimeFrame->getTracks();
  trks.reserve(trks.size() + tracks.size());
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
    if (nShared - int(isFirstShared && mTrkParams[iteration].AllowSharingFirstCluster) > mTrkParams[iteration].SharedMaxClusters) {
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
    const auto diff = track.getExtendedLayerPattern<NLayers>();
    if (diff) {
      size_t nExtendedClusters = 0;
      for (int iLayer{0}; iLayer < mTrkParams[iteration].NLayers; ++iLayer) {
        nExtendedClusters += static_cast<bool>(diff & (0x1u << iLayer));
      }
      mTimeFrame->addTrackExtensionCounters(1, nExtendedClusters);
    }
    track.clearExtendedLayerPattern();
    trks.emplace_back(track);

    if (mTrkParams[iteration].AllowSharingFirstCluster) {
      firstClusters[firstLayer].push_back(firstCluster);
    }
  }
}

template <int NLayers>
void TrackerTraits<NLayers>::markTracks(int iteration)
{
  if (mTrkParams[iteration].AllowSharingFirstCluster) {
    /// Now we have to set the shared cluster flag
    auto& tracks = mTimeFrame->getTracks();

    bounded_vector<int> fclusSort(tracks.size(), mMemoryPool.get());
    std::iota(fclusSort.begin(), fclusSort.end(), 0);
    std::sort(fclusSort.begin(), fclusSort.end(), [&tracks](int a, int b) {
      return tracks[a].getFirstLayerClusterIndex() < tracks[b].getFirstLayerClusterIndex();
    });

    auto areTracksSelected = [this, iteration](const TrackITSExt& t1, const TrackITSExt& t2) {
      const auto t1FirstLayer{t1.getFirstClusterLayer()}, t2FirstLayer{t2.getFirstClusterLayer()};
      if (t1FirstLayer != t2FirstLayer) {
        return false;
      }
      if (mTimeFrame->getClusterROF(t1FirstLayer, t1.getClusterIndex(t1FirstLayer)) != mTimeFrame->getClusterROF(t2FirstLayer, t2.getClusterIndex(t2FirstLayer))) {
        return false;
      }
      if (!math_utils::isPhiDifferenceBelow(t1.getPhi(), t2.getPhi(), mTrkParams[iteration].SharedClusterMaxDeltaPhi)) {
        return false;
      }
      if (std::abs(t1.getEta() - t2.getEta()) > mTrkParams[iteration].SharedClusterMaxDeltaEta) {
        return false;
      }
      if (mTrkParams[iteration].SharedClusterOppositeSign && t1.getSign() == t2.getSign()) {
        return false;
      }
      return true;
    };

    for (int i{0}; i < static_cast<int>(fclusSort.size()); ++i) {
      auto& track = tracks[fclusSort[i]];
      for (int j{i + 1}; j < static_cast<int>(fclusSort.size()) && tracks[fclusSort[j]].getFirstLayerClusterIndex() == track.getFirstLayerClusterIndex(); ++j) {
        auto& track2 = tracks[fclusSort[j]];
        if (areTracksSelected(track, track2)) {
          track.setSharedClusters();
          track2.setSharedClusters();
        }
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
template void TrackerTraits<7>::processNeighbours<CellSeed>(int, int, int, const bounded_vector<CellSeed>&, const bounded_vector<int>&, const bounded_vector<int>&, bounded_vector<TrackSeed<7>>&, bounded_vector<int>&, bounded_vector<int>&);
template void TrackerTraits<7>::processNeighbours<TrackSeed<7>>(int, int, int, const bounded_vector<TrackSeed<7>>&, const bounded_vector<int>&, const bounded_vector<int>&, bounded_vector<TrackSeed<7>>&, bounded_vector<int>&, bounded_vector<int>&);
// ALICE3 upgrade
#ifdef ENABLE_UPGRADES
template class TrackerTraits<11>;
template void TrackerTraits<11>::processNeighbours<CellSeed>(int, int, int, const bounded_vector<CellSeed>&, const bounded_vector<int>&, const bounded_vector<int>&, bounded_vector<TrackSeed<11>>&, bounded_vector<int>&, bounded_vector<int>&);
template void TrackerTraits<11>::processNeighbours<TrackSeed<11>>(int, int, int, const bounded_vector<TrackSeed<11>>&, const bounded_vector<int>&, const bounded_vector<int>&, bounded_vector<TrackSeed<11>>&, bounded_vector<int>&, bounded_vector<int>&);
#endif

} // namespace o2::its
