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
#include <iterator>
#include <mutex>
#include <ranges>
#include <cmath>
#include <type_traits>
#include <vector>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_scan.h>
#include <oneapi/tbb/parallel_sort.h>

#include "DetectorsBase/Propagator.h"
#include "GPUCommonMath.h"
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/Cell.h"
#include "ITStracking/Constants.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/LayerMask.h"
#include "ITStracking/ROFLookupTables.h"
#include "ITStracking/SlabBumpAllocator.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/TrackFollower.h"
#include "ITStracking/TrackHelpers.h"
#include "ITStracking/Tracklet.h"

namespace o2::its
{

template <int NLayers>
void TrackerTraits<NLayers>::initialiseTimeFrame(const int iteration)
{
  this->mTaskArena->execute([&] {
    mTimeFrame->initialise(mTrkParams[iteration], mTrkParams[iteration].NLayers, iteration);
  });
}

template <int NLayers>
void TrackerTraits<NLayers>::computeLayerTracklets(const int iteration, int iVertex)
{
  const auto topology = mTimeFrame->getTrackingTopologyView();
  const Vertex diamondVert(mTrkParams[iteration].Diamond, mTrkParams[iteration].DiamondCov, 1, 1.f);
  gsl::span<const Vertex> diamondSpan(&diamondVert, 1);

  mTaskArena->execute([&] {
    tbb::parallel_for(0, static_cast<int>(topology.nLinks), [&](const int linkId) {
      mTimeFrame->getTracklets()[linkId].clear();
      mTimeFrame->getTrackletsLabel(linkId).clear();
      auto& lut = mTimeFrame->getTrackletsLookupTable()[linkId];
      std::fill(lut.begin(), lut.end(), 0);
    });

    auto forTracklets = [&](int linkId, int pivotROF, auto&& emit) {
      const auto& link = topology.getLink(linkId);
      if (!mTimeFrame->getROFMaskView().isROFEnabled(link.fromLayer, pivotROF)) {
        return;
      }
      gsl::span<const Vertex> primaryVertices = mTrkParams[iteration].UseDiamond ? diamondSpan : mTimeFrame->getPrimaryVertices(link.fromLayer, pivotROF);
      if (primaryVertices.empty()) {
        return;
      }
      const int startVtx = iVertex >= 0 ? iVertex : 0;
      const int endVtx = iVertex >= 0 ? o2::gpu::CAMath::Min(iVertex + 1, int(primaryVertices.size())) : int(primaryVertices.size());
      if (endVtx <= startVtx || (iVertex + 1) > primaryVertices.size()) {
        return;
      }

      const auto& rofOverlap = mTimeFrame->getROFOverlapTableView().getOverlap(link.fromLayer, link.toLayer, pivotROF);
      if (!rofOverlap.getEntries()) {
        return;
      }

      auto layer0 = mTimeFrame->getClustersOnLayer(pivotROF, link.fromLayer);
      if (layer0.empty()) {
        return;
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
                  const float phi{o2::math_utils::fastATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
                  const float tanL = (currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius);
                  emit(currentSortedIndex, mTimeFrame->getSortedIndex(targetROF, link.toLayer, iNext), tanL, phi, ts);
                }
              }
            }
          }
        }
      }
    };

    if (mTaskArena->max_concurrency() <= 1) {
      for (int linkId{0}; linkId < topology.nLinks; ++linkId) {
        const int fromLayer = topology.getLink(linkId).fromLayer;
        const int endROF = mTimeFrame->getROFOverlapTableView().getLayer(fromLayer).mNROFsTF;
        auto& tracklets = mTimeFrame->getTracklets()[linkId];
        for (int pivotROF{0}; pivotROF < endROF; ++pivotROF) {
          forTracklets(linkId, pivotROF, [&tracklets](auto&&... args) { tracklets.emplace_back(std::forward<decltype(args)>(args)...); });
        }
      }
    } else {
      const int maxConcurrency = std::max(1, mTaskArena->max_concurrency());
      const int nConcurrentSinks = std::min(static_cast<int>(topology.nLinks), maxConcurrency);
      tbb::parallel_for(0, static_cast<int>(topology.nLinks), [&](const int linkId) {
        const int fromLayer = topology.getLink(linkId).fromLayer;
        const int startROF = 0, endROF = mTimeFrame->getROFOverlapTableView().getLayer(fromLayer).mNROFsTF;
        auto& tracklets = mTimeFrame->getTracklets()[linkId];
        const auto key = CapacityEstimator::makeKey(SlabSite::Tracklets, iteration, iVertex + 1, linkId);
        const auto scale = static_cast<double>(mTimeFrame->getClusters()[fromLayer].size());
        const size_t capacity = mTimeFrame->getCapacityEstimator().capacity(key, scale);

        UnorderedSlabSink<Tracklet> sink{{.capacity = capacity, .nThreads = maxConcurrency, .nConcurrentSinks = nConcurrentSinks}, mMemoryPool.get()};
        tbb::parallel_for(startROF, endROF, [&](const int pivotROF) {
          auto& handle = sink.local();
          forTracklets(linkId, pivotROF, [&handle](auto&&... args) { handle.emplace(std::forward<decltype(args)>(args)...); });
        });
        const auto st = sink.stats();
        sink.finalizeUnordered(tracklets);
        mTimeFrame->getCapacityEstimator().update(key, scale, st.emitted, st.capacity, st.overflowed, st.memoryLimited);
      });
    }

    tbb::parallel_for(0, static_cast<int>(topology.nLinks), [&](const int linkId) {
      /// Sort tracklets & remove duplicates
      auto& trkl{mTimeFrame->getTracklets()[linkId]};
      if (mTaskArena->max_concurrency() > 1) {
        tbb::parallel_sort(trkl.begin(), trkl.end());
      } else {
        std::sort(trkl.begin(), trkl.end());
      }
      if (iVertex < 0) { // duplicates can exist simply since we evaluate for all vertices if we do perVertex duplicates cannot exist
        trkl.erase(std::unique(trkl.begin(), trkl.end()), trkl.end());
        trkl.shrink_to_fit();
      }
      auto& lut{mTimeFrame->getTrackletsLookupTable()[linkId]};
      if (!trkl.empty()) {
        const size_t nTracklets{trkl.size()};
        const Tracklet* tkls{trkl.data()};
        tbb::parallel_for(tbb::blocked_range<size_t>(0, nTracklets), [&](const tbb::blocked_range<size_t>& r) {
          size_t begin{r.begin()}, end{r.end()};
          const auto sameRun = [tkls](size_t i, size_t j) { return tkls[i].firstClusterIndex == tkls[j].firstClusterIndex; };
          while (begin > 0 && begin < nTracklets && sameRun(begin, begin - 1)) {
            ++begin;
          }
          while (end > 0 && end < nTracklets && sameRun(end, end - 1)) {
            ++end;
          }
          for (size_t i{begin}; i < end; ++i) {
            ++lut[tkls[i].firstClusterIndex + 1];
          }
        });
        int* data{lut.data()};
        tbb::parallel_scan(
          tbb::blocked_range<size_t>(0, lut.size()), 0,
          [data](const tbb::blocked_range<size_t>& r, int running, bool isFinal) {
            for (size_t i{r.begin()}; i < r.end(); ++i) {
              running += data[i];
              if (isFinal) {
                data[i] = running;
              }
            }
            return running;
          },
          std::plus<int>());
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
  const bool createLabels = mTimeFrame->hasMCinformation() && mTrkParams[iteration].CreateArtefactLabels;

  mTaskArena->execute([&] {
    const int maxConcurrency = std::max(1, mTaskArena->max_concurrency());
    auto clearTopology = [&](const int cellTopologyId) {
      deepVectorClear(mTimeFrame->getCells()[cellTopologyId]);
      deepVectorClear(mTimeFrame->getCellsLookupTable()[cellTopologyId]);
      if (createLabels) {
        deepVectorClear(mTimeFrame->getCellsLabel(cellTopologyId));
      }
    };
    if (maxConcurrency > 1) {
      tbb::parallel_for(0, static_cast<int>(topology.nCells), clearTopology);
    } else {
      for (int cellTopologyId{0}; cellTopologyId < topology.nCells; ++cellTopologyId) {
        clearTopology(cellTopologyId);
      }
    }

    auto forTrackletCells = [&](int cellTopologyId, int iTracklet, auto&& emit) {
      const auto& cellTopology = topology.getCell(cellTopologyId);
      const auto& firstLink = topology.getLink(cellTopology.firstLink);
      const auto& secondLink = topology.getLink(cellTopology.secondLink);
      const Tracklet& currentTracklet{mTimeFrame->getTracklets()[cellTopology.firstLink][iTracklet]};
      const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
      const int nextLayerFirstTrackletIndex{mTimeFrame->getTrackletsLookupTable()[cellTopology.secondLink][nextLayerClusterIndex]};
      const int nextLayerLastTrackletIndex{mTimeFrame->getTrackletsLookupTable()[cellTopology.secondLink][nextLayerClusterIndex + 1]};
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
          const auto& cluster1Glo = mTimeFrame->getUnsortedClusters()[firstLink.fromLayer][clusId[0]];
          const auto& cluster2Glo = mTimeFrame->getUnsortedClusters()[firstLink.toLayer][clusId[1]];
          const auto& cluster3Tf = mTimeFrame->getTrackingFrameInfoOnLayer(secondLink.toLayer)[clusId[2]];
          auto track{o2::its::track::buildTrackSeed(cluster1Glo, cluster2Glo, cluster3Tf, mBz)};

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
            emit(cellTopology.hitLayerMask, clusId[0], clusId[1], clusId[2], iTracklet, iNextTracklet, track, chi2, ts);
          }
        }
      }
    };

    bounded_vector<int> activeTopologies(mMemoryPool.get());
    activeTopologies.reserve(topology.nCells);
    for (int cellTopologyId = 0; cellTopologyId < topology.nCells; ++cellTopologyId) {
      const auto& cellTopology = topology.getCell(cellTopologyId);
      if (!mTimeFrame->getTracklets()[cellTopology.firstLink].empty() &&
          !mTimeFrame->getTracklets()[cellTopology.secondLink].empty()) {
        activeTopologies.push_back(cellTopologyId);
      }
    }

    const int nConcurrentSinks = std::min(maxConcurrency, static_cast<int>(activeTopologies.size()));
    auto processTopology = [&](const int cellTopologyId) {
      const auto& cellTopology = topology.getCell(cellTopologyId);

      auto& layerCells = mTimeFrame->getCells()[cellTopologyId];
      auto& lut = mTimeFrame->getCellsLookupTable()[cellTopologyId];
      const int currentLayerTrackletsNum{static_cast<int>(mTimeFrame->getTracklets()[cellTopology.firstLink].size())};

      const auto key = CapacityEstimator::makeKey(SlabSite::Cells, iteration, 0, cellTopologyId);
      const auto scale = static_cast<double>(currentLayerTrackletsNum);
      if (maxConcurrency > 1) {
        const size_t capacity = mTimeFrame->getCapacityEstimator().capacity(key, scale);

        GroupedSlabSink<CellSeed> sink{{.capacity = capacity, .nThreads = maxConcurrency, .nConcurrentSinks = nConcurrentSinks}, mMemoryPool.get()};
        tbb::parallel_for(0, currentLayerTrackletsNum, [&](const int iTracklet) {
          auto& handle = sink.local();
          handle.beginProducer(iTracklet);
          forTrackletCells(cellTopologyId, iTracklet, [&handle](auto&&... args) { handle.emplace(std::forward<decltype(args)>(args)...); });
        });
        const auto st = sink.stats();
        sink.finalizeGrouped(size_t(currentLayerTrackletsNum), lut, layerCells);
        mTimeFrame->getCapacityEstimator().update(key, scale, st.emitted, st.capacity, st.overflowed, st.memoryLimited);
      } else {
        lut.resize(currentLayerTrackletsNum + 1);
        for (int iTracklet{0}; iTracklet < currentLayerTrackletsNum; ++iTracklet) {
          lut[iTracklet] = static_cast<int>(layerCells.size());
          forTrackletCells(cellTopologyId, iTracklet, [&](auto&&... args) {
            layerCells.emplace_back(std::forward<decltype(args)>(args)...);
          });
        }
        lut.back() = static_cast<int>(layerCells.size());
      }

      if (createLabels) {
        auto& labels = mTimeFrame->getCellsLabel(cellTopologyId);
        labels.reserve(layerCells.size());
        for (const auto& cell : layerCells) {
          MCCompLabel currentLab{mTimeFrame->getTrackletsLabel(cellTopology.firstLink)[cell.getFirstTrackletIndex()]};
          MCCompLabel nextLab{mTimeFrame->getTrackletsLabel(cellTopology.secondLink)[cell.getSecondTrackletIndex()]};
          labels.emplace_back(currentLab == nextLab ? currentLab : MCCompLabel());
        }
      }
    };

    if (maxConcurrency > 1) {
      tbb::parallel_for(0, static_cast<int>(activeTopologies.size()), [&](const int i) {
        processTopology(activeTopologies[i]);
      });
    } else {
      for (const int cellTopologyId : activeTopologies) {
        processTopology(cellTopologyId);
      }
    }

    auto clearTracklets = [&](const int linkId) {
      deepVectorClear(mTimeFrame->getTracklets()[linkId]);
      deepVectorClear(mTimeFrame->getTrackletsLabel(linkId));
    };
    if (maxConcurrency > 1) {
      tbb::parallel_for(0, static_cast<int>(topology.nLinks), clearTracklets);
    } else {
      for (int linkId{0}; linkId < topology.nLinks; ++linkId) {
        clearTracklets(linkId);
      }
    }
  });
}

template <int NLayers>
void TrackerTraits<NLayers>::findCellsNeighbours(const int iteration)
{
  const auto topology = mTimeFrame->getTrackingTopologyView();
  mTaskArena->execute([&] {
    const int maxConcurrency = std::max(1, mTaskArena->max_concurrency());
    auto clearNeighbours = [&](const int cellTopologyId) {
      deepVectorClear(mTimeFrame->getCellsNeighbours()[cellTopologyId]);
      deepVectorClear(mTimeFrame->getCellsNeighboursTopology()[cellTopologyId]);
      deepVectorClear(mTimeFrame->getCellsNeighboursLUT()[cellTopologyId]);
    };
    if (maxConcurrency > 1) {
      tbb::parallel_for(0, static_cast<int>(topology.nCells), clearNeighbours);
    } else {
      for (int cellTopologyId{0}; cellTopologyId < topology.nCells; ++cellTopologyId) {
        clearNeighbours(cellTopologyId);
      }
    }

    auto neighbourLess = [](const CellNeighbour& a, const CellNeighbour& b) {
      return std::tie(a.nextCellTopology, a.nextCell, a.cellTopology, a.cell) <
             std::tie(b.nextCellTopology, b.nextCell, b.cellTopology, b.cell);
    };

    for (int outerLayer{0}; outerLayer < NLayers; ++outerLayer) {
      bounded_vector<int> activeTopologies(mMemoryPool.get());
      activeTopologies.reserve(topology.nCells);
      size_t sourceCellCount{0};
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
        activeTopologies.push_back(cellTopologyId);
        sourceCellCount += mTimeFrame->getCells()[cellTopologyId].size();
      }

      if (activeTopologies.empty()) {
        continue;
      }

      auto forSourceCell = [&](const int cellTopologyId, const int iCell, auto&& emit) {
        const auto& cellTopology = topology.getCell(cellTopologyId);
        const auto successors = topology.getCellsStartingWithLink(cellTopology.secondLink);
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

            float chi2 = currentCellSeed.getPredictedChi2Fast(nextCellSeed);
            if (chi2 > mTrkParams[iteration].MaxChi2ClusterAttachment) {
              continue;
            }

            const int nextLevel = currentCellSeed.getLevel() + 1;
            emit(cellTopologyId, iCell, nextCellTopologyId, iNextCell, nextLevel);
          }
        }
      };

      bounded_vector<CellNeighbour> waveNeighbours{mMemoryPool.get()};
      const auto key = CapacityEstimator::makeKey(SlabSite::Neighbours, iteration, 0, outerLayer);
      const auto scale = static_cast<double>(sourceCellCount);
      if (maxConcurrency > 1) {
        const size_t capacity = mTimeFrame->getCapacityEstimator().capacity(key, scale);
        UnorderedSlabSink<CellNeighbour> sink{{.capacity = capacity, .nThreads = maxConcurrency}, mMemoryPool.get()};
        tbb::parallel_for(0, static_cast<int>(activeTopologies.size()), [&](const int i) {
          const int cellTopologyId = activeTopologies[i];
          tbb::parallel_for(0, static_cast<int>(mTimeFrame->getCells()[cellTopologyId].size()), [&](const int iCell) {
            auto& handle = sink.local();
            forSourceCell(cellTopologyId, iCell, [&handle](auto&&... args) {
              handle.emplace(std::forward<decltype(args)>(args)...);
            });
          });
        });
        const auto st = sink.stats();
        sink.finalizeUnordered(waveNeighbours);
        mTimeFrame->getCapacityEstimator().update(key, scale, st.emitted, st.capacity, st.overflowed, st.memoryLimited);
        tbb::parallel_sort(waveNeighbours.begin(), waveNeighbours.end(), neighbourLess);
      } else {
        for (const int cellTopologyId : activeTopologies) {
          for (int iCell{0}; iCell < static_cast<int>(mTimeFrame->getCells()[cellTopologyId].size()); ++iCell) {
            forSourceCell(cellTopologyId, iCell, [&](auto&&... args) {
              waveNeighbours.emplace_back(std::forward<decltype(args)>(args)...);
            });
          }
        }
        std::sort(waveNeighbours.begin(), waveNeighbours.end(), neighbourLess);
      }

      struct TargetSpan {
        int topologyId;
        size_t begin;
        size_t end;
      };
      bounded_vector<TargetSpan> targetSpans{mMemoryPool.get()};
      targetSpans.reserve(topology.nCells);
      for (int targetTopologyId{0}; targetTopologyId < topology.nCells; ++targetTopologyId) {
        const auto first = std::lower_bound(waveNeighbours.begin(), waveNeighbours.end(), targetTopologyId,
                                            [](const CellNeighbour& neighbour, int id) { return neighbour.nextCellTopology < id; });
        const auto last = std::upper_bound(first, waveNeighbours.end(), targetTopologyId,
                                           [](int id, const CellNeighbour& neighbour) { return id < neighbour.nextCellTopology; });
        if (first != last) {
          targetSpans.push_back({targetTopologyId, static_cast<size_t>(first - waveNeighbours.begin()), static_cast<size_t>(last - waveNeighbours.begin())});
        }
      }

      auto finalizeTarget = [&](const int i) {
        const auto [targetTopologyId, begin, end] = targetSpans[i];
        auto& cellsNeighbourLUT = mTimeFrame->getCellsNeighboursLUT()[targetTopologyId];
        cellsNeighbourLUT.assign(mTimeFrame->getCells()[targetTopologyId].size(), 0);
        for (size_t j{begin}; j < end; ++j) {
          const auto& neighbour = waveNeighbours[j];
          ++cellsNeighbourLUT[neighbour.nextCell];
          auto& targetCell = mTimeFrame->getCells()[targetTopologyId][neighbour.nextCell];
          if (neighbour.level > targetCell.getLevel()) {
            targetCell.setLevel(neighbour.level);
          }
        }
        std::inclusive_scan(cellsNeighbourLUT.begin(), cellsNeighbourLUT.end(), cellsNeighbourLUT.begin());

        auto& cellsNeighbours = mTimeFrame->getCellsNeighbours()[targetTopologyId];
        auto& cellsNeighboursTopology = mTimeFrame->getCellsNeighboursTopology()[targetTopologyId];
        cellsNeighbours.resize(end - begin);
        cellsNeighboursTopology.resize(end - begin);
        for (size_t j{begin}; j < end; ++j) {
          cellsNeighbours[j - begin] = waveNeighbours[j].cell;
          cellsNeighboursTopology[j - begin] = waveNeighbours[j].cellTopology;
        }
      };
      if (maxConcurrency > 1) {
        tbb::parallel_for(0, static_cast<int>(targetSpans.size()), finalizeTarget);
      } else {
        for (int i{0}; i < static_cast<int>(targetSpans.size()); ++i) {
          finalizeTarget(i);
        }
      }
    }

    // clean up LUTs
    auto clearCellLUT = [&](const int cellTopologyId) {
      deepVectorClear(mTimeFrame->getCellsLookupTable()[cellTopologyId]);
    };
    if (maxConcurrency > 1) {
      tbb::parallel_for(0, static_cast<int>(topology.nCells), clearCellLUT);
    } else {
      for (int cellTopologyId{0}; cellTopologyId < topology.nCells; ++cellTopologyId) {
        clearCellLUT(cellTopologyId);
      }
    }
  });
}

template <int NLayers>
template <typename InputSeed>
void TrackerTraits<NLayers>::processNeighbours(int iteration, int defaultCellTopologyId, int iLevel, uint64_t capacityKey, const bounded_vector<InputSeed>& currentSeeds, bounded_vector<RoadSeedN>& updatedSeeds)
{
  constexpr bool IsInitial = std::is_same_v<InputSeed, CellSeed>;
  static_assert(IsInitial || std::is_same_v<InputSeed, RoadSeedN>);
  auto propagator = o2::base::Propagator::Instance();

  mTaskArena->execute([&] {
    auto forCellNeighbours = [&](int iCell, auto&& emit) {
      const auto& inputSeed = currentSeeds[iCell];
      const auto& currentCell = [&]() -> const auto& {
        if constexpr (IsInitial) {
          return inputSeed;
        } else {
          return inputSeed.seed;
        }
      }();
      const int cellTopologyId = [&]() {
        if constexpr (IsInitial) {
          return defaultCellTopologyId;
        } else {
          return inputSeed.cellTopologyId;
        }
      }();
      const int cellId = [&]() {
        if constexpr (IsInitial) {
          return iCell;
        } else {
          return inputSeed.cellId;
        }
      }();

      if (currentCell.getLevel() != iLevel) {
        return;
      }
      if constexpr (IsInitial) {
        for (int layer = 0; layer < NLayers; ++layer) {
          const int clusterIndex = currentCell.getCluster(layer);
          if (clusterIndex != constants::UnusedIndex && mTimeFrame->isClusterUsed(layer, clusterIndex)) {
            return;
          }
        }
      }

      if (cellTopologyId < 0 || mTimeFrame->getCellsNeighboursLUT()[cellTopologyId].empty()) {
        return;
      }
      const int startNeighbourId{cellId ? mTimeFrame->getCellsNeighboursLUT()[cellTopologyId][cellId - 1] : 0};
      const int endNeighbourId{mTimeFrame->getCellsNeighboursLUT()[cellTopologyId][cellId]};
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

        seed.getClusters()[neighbourLayer] = neighbourCluster;
        auto mask = seed.getHitLayerMask();
        mask.set(neighbourLayer);
        seed.setHitLayerMask(mask);
        seed.setLevel(neighbourCell.getLevel());
        seed.setFirstTrackletIndex(neighbourCell.getFirstTrackletIndex());
        seed.setSecondTrackletIndex(neighbourCell.getSecondTrackletIndex());
        emit(std::move(seed), neighbourCellId, neighbourCellTopologyId);
      }
    };

    const int nCells = static_cast<int>(currentSeeds.size());
    if (mTaskArena->max_concurrency() <= 1) {
      for (int iCell{0}; iCell < nCells; ++iCell) {
        forCellNeighbours(iCell, [&](auto&&... args) { updatedSeeds.emplace_back(std::forward<decltype(args)>(args)...); });
      }
    } else {
      const auto scale = static_cast<double>(nCells);
      const size_t capacity = mTimeFrame->getCapacityEstimator().capacity(capacityKey, scale);
      UnorderedSlabSink<RoadSeedN> sink{{.capacity = capacity, .nThreads = mTaskArena->max_concurrency()}, mMemoryPool.get()};

      tbb::parallel_for(0, nCells, [&](const int iCell) {
        auto& handle = sink.local();
        forCellNeighbours(iCell, [&](auto&&... args) { handle.emplace(std::forward<decltype(args)>(args)...); });
      });
      const auto st = sink.stats();
      sink.finalizeUnordered(updatedSeeds);
      mTimeFrame->getCapacityEstimator().update(capacityKey, scale, st.emitted, st.capacity, st.overflowed, st.memoryLimited);
    }
  });
}

template <int NLayers>
bool TrackerTraits<NLayers>::finaliseTrackSeed(const TrackSeedN& seed,
                                               TrackITSExt& track,
                                               const int iteration,
                                               const TrackingFrameInfo* const* tfInfos,
                                               const Cluster* const* unsortedClusters,
                                               const o2::base::Propagator* propagator,
                                               const TrackFollowContext<NLayers>& followCtx,
                                               TrackFollowerScratch& scratch)
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

  if (static_cast<int>(scratch.activeHypotheses.size()) < followCtx.maxHypotheses) {
    scratch.activeHypotheses.resize(followCtx.maxHypotheses);
  }
  if (static_cast<int>(scratch.nextHypotheses.size()) < followCtx.maxHypotheses) {
    scratch.nextHypotheses.resize(followCtx.maxHypotheses);
  }

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
  tbb::enumerable_thread_specific<TrackFollowerScratch> followerScratch{
    [mr = mMemoryPool.get()]() { return TrackFollowerScratch{mr}; }};
  for (int startLevel{mTrkParams[iteration].CellsPerRoad()}; startLevel >= mTrkParams[iteration].CellMinimumLevel(); --startLevel) {

    const track::TrackSeedSelector<NLayers> seedFilter{constants::MaxTrackSeedQ2Pt, mTrkParams[iteration].MaxChi2NDF, startLevel, mTrkParams[iteration].MaxHoles, mTrkParams[iteration].getMinSeedingClusters(), mTrkParams[iteration].HoleLayerMask, mTrkParams[iteration].getNonSeedingLayerMask()};

    bounded_vector<TrackSeedN> trackSeeds(mMemoryPool.get());
    for (int startCellTopologyId{0}; startCellTopologyId < topology.nCells; ++startCellTopologyId) {
      const int startLayer = topology.getCell(startCellTopologyId).hitLayerMask.last();
      if (!(mTrkParams[iteration].StartLayerMask.has(startLayer)) ||
          mTimeFrame->getCells()[startCellTopologyId].empty() ||
          topology.getMaxCellLevel(startCellTopologyId) < startLevel) {
        continue;
      }

      bounded_vector<RoadSeedN> lastSeeds(mMemoryPool.get()), updatedSeeds(mMemoryPool.get());

      auto roadKey = [&](int level) {
        return CapacityEstimator::makeKey(SlabSite::Roads, iteration, CapacityEstimator::makeVariant(startLevel, level), startCellTopologyId);
      };

      processNeighbours(iteration, startCellTopologyId, startLevel, roadKey(startLevel), mTimeFrame->getCells()[startCellTopologyId], updatedSeeds);

      int level = startLevel;
      while (level > 2 && !updatedSeeds.empty()) {
        lastSeeds.swap(updatedSeeds);
        deepVectorClear(updatedSeeds);
        --level;
        processNeighbours(iteration, constants::UnusedIndex, level, roadKey(level), lastSeeds, updatedSeeds);
      }
      deepVectorClear(lastSeeds);

      if (!updatedSeeds.empty()) {
        trackSeeds.reserve(trackSeeds.size() + std::count_if(updatedSeeds.begin(), updatedSeeds.end(), [&](const auto& road) { return seedFilter(road.seed); }));
        for (auto& road : updatedSeeds) {
          if (seedFilter(road.seed)) {
            trackSeeds.emplace_back(std::move(road.seed));
          }
        }
      }
    }

    if (trackSeeds.empty()) {
      continue;
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
      mTrkParams[iteration].LayerRadii.data(), mTrkParams[iteration].PhiBins,
      std::max(1, mTrkParams[iteration].TrackFollowerMaxHypotheses),
      mTrkParams[iteration].TrackFollowerNSigmaCutPhi, mTrkParams[iteration].TrackFollowerNSigmaCutZ};

    bounded_vector<TrackITSExt> tracks(mMemoryPool.get());
    mTaskArena->execute([&] {
      const int nSeeds = static_cast<int>(trackSeeds.size());
      const int maxConcurrency = std::max(1, mTaskArena->max_concurrency());
      const int chunkSize = std::min(nSeeds, std::clamp(nSeeds / (constants::NumberOfConcurrentSeeds * maxConcurrency), constants::MinNumberOfConcurrentSeeds, constants::MaxNumberOfConcurrentSeeds)); // acts as memory bound and minimum work

      // flush local track vector to global vector on reaching chunkSize
      std::mutex tracksMutex;
      auto flushTracks = [&](bounded_vector<TrackITSExt>& localTracks) {
        if (localTracks.empty()) {
          return;
        }
        std::lock_guard lock{tracksMutex};
        tracks.insert(tracks.end(), std::make_move_iterator(localTracks.begin()), std::make_move_iterator(localTracks.end()));
        localTracks.clear();
      };

      // each worker works on its own range
      tbb::parallel_for(tbb::blocked_range<int>(0, nSeeds, chunkSize), [&](const auto& range) {
        bounded_vector<TrackITSExt> localTracks(mMemoryPool.get());
        localTracks.reserve(std::min(chunkSize, static_cast<int>(range.size())));
        auto& scratch = followerScratch.local();
        for (int iSeed{range.begin()}; iSeed < range.end(); ++iSeed) {
          localTracks.emplace_back();
          if (!finaliseTrackSeed(trackSeeds[iSeed], localTracks.back(), iteration, tfInfos, unsortedClusters, propagator, followCtx, scratch)) {
            localTracks.pop_back();
          }
          if (static_cast<int>(localTracks.size()) == chunkSize) {
            flushTracks(localTracks);
          }
        }
        flushTracks(localTracks); // flush remaining
        deepVectorClear(localTracks);
      });

      deepVectorClear(trackSeeds);
    });

    // Sort tracks via indices to avoid moving TrackITSExt objects.
    bounded_vector<int> trackIndices(tracks.size(), mMemoryPool.get());
    std::iota(trackIndices.begin(), trackIndices.end(), 0);
    std::sort(trackIndices.begin(), trackIndices.end(), [&tracks](int a, int b) {
      return track::isBetter(tracks[a], tracks[b]);
    });

    acceptTracks(iteration, tracks, trackIndices, firstClusters);
  }
  markTracks(iteration);
}

template <int NLayers>
void TrackerTraits<NLayers>::acceptTracks(int iteration,
                                          bounded_vector<TrackITSExt>& tracks,
                                          const bounded_vector<int>& trackIndices,
                                          bounded_vector<bounded_vector<int>>& firstClusters)
{
  auto& trks = mTimeFrame->getTracks();
  trks.reserve(trks.size() + tracks.size());
  const float smallestROFHalf = mTimeFrame->getROFOverlapTableView().getClockLayer().mROFLength * 0.5f;
  for (size_t trackId{0}; trackId < trackIndices.size(); ++trackId) {
    auto& track = tracks[trackIndices[trackId]];
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
template void TrackerTraits<7>::processNeighbours<CellSeed>(int, int, int, uint64_t, const bounded_vector<CellSeed>&, bounded_vector<RoadSeed<7>>&);
template void TrackerTraits<7>::processNeighbours<RoadSeed<7>>(int, int, int, uint64_t, const bounded_vector<RoadSeed<7>>&, bounded_vector<RoadSeed<7>>&);
// ALICE3 upgrade
#ifdef ENABLE_UPGRADES
template class TrackerTraits<11>;
template void TrackerTraits<11>::processNeighbours<CellSeed>(int, int, int, uint64_t, const bounded_vector<CellSeed>&, bounded_vector<RoadSeed<11>>&);
template void TrackerTraits<11>::processNeighbours<RoadSeed<11>>(int, int, int, uint64_t, const bounded_vector<RoadSeed<11>>&, bounded_vector<RoadSeed<11>>&);
template class TrackerTraits<13>;
template void TrackerTraits<13>::processNeighbours<CellSeed>(int, int, int, uint64_t, const bounded_vector<CellSeed>&, bounded_vector<RoadSeed<13>>&);
template void TrackerTraits<13>::processNeighbours<RoadSeed<13>>(int, int, int, uint64_t, const bounded_vector<RoadSeed<13>>&, bounded_vector<RoadSeed<13>>&);
#endif

} // namespace o2::its
