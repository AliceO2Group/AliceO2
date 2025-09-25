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

#include <memory>
#include <ranges>
#include <map>
#include <algorithm>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/combinable.h>

#include "ITStracking/VertexerTraits.h"
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "Steer/MCKinematicsReader.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DetectorsRaw/HBFUtils.h"
#include "CommonUtils/TreeStreamRedirector.h"

namespace o2::its
{

template <TrackletMode Mode, bool EvalRun, int nLayers>
static void trackleterKernelHost(
  const gsl::span<const Cluster>& clustersNextLayer,    // 0 2
  const gsl::span<const Cluster>& clustersCurrentLayer, // 1 1
  const gsl::span<uint8_t>& usedClustersNextLayer,      // 0 2
  int* indexTableNext,
  const float phiCut,
  bounded_vector<Tracklet>& tracklets,
  gsl::span<int> foundTracklets,
  const IndexTableUtils<nLayers>& utils,
  const short pivotRof,
  const short targetRof,
  gsl::span<int> rofFoundTrackletsOffsets, // we want to change those, to keep track of the offset in deltaRof>0
  const int maxTrackletsPerCluster = static_cast<int>(2e3))
{
  const int PhiBins{utils.getNphiBins()};
  const int ZBins{utils.getNzBins()};
  // loop on layer1 clusters
  for (int iCurrentLayerClusterIndex = 0; iCurrentLayerClusterIndex < clustersCurrentLayer.size(); ++iCurrentLayerClusterIndex) {
    int storedTracklets{0};
    const Cluster& currentCluster{clustersCurrentLayer[iCurrentLayerClusterIndex]};
    const int4 selectedBinsRect{VertexerTraits<nLayers>::getBinsRect(currentCluster, (int)Mode, 0.f, 50.f, phiCut / 2, utils)};
    if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
      int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};
      if (phiBinsNum < 0) {
        phiBinsNum += PhiBins;
      }
      // loop on phi bins next layer
      for (int iPhiBin{selectedBinsRect.y}, iPhiCount{0}; iPhiCount < phiBinsNum; iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++) {
        const int firstBinIndex{utils.getBinIndex(selectedBinsRect.x, iPhiBin)};
        const int firstRowClusterIndex{indexTableNext[firstBinIndex]};
        const int maxRowClusterIndex{indexTableNext[firstBinIndex + ZBins]};
        // loop on clusters next layer
        for (int iNextLayerClusterIndex{firstRowClusterIndex}; iNextLayerClusterIndex < maxRowClusterIndex && iNextLayerClusterIndex < static_cast<int>(clustersNextLayer.size()); ++iNextLayerClusterIndex) {
          if (usedClustersNextLayer[iNextLayerClusterIndex]) {
            continue;
          }
          const Cluster& nextCluster{clustersNextLayer[iNextLayerClusterIndex]};
          if (o2::gpu::GPUCommonMath::Abs(math_utils::smallestAngleDifference(currentCluster.phi, nextCluster.phi)) < phiCut) {
            if (storedTracklets < maxTrackletsPerCluster) {
              if constexpr (!EvalRun) {
                if constexpr (Mode == TrackletMode::Layer0Layer1) {
                  tracklets[rofFoundTrackletsOffsets[iCurrentLayerClusterIndex] + storedTracklets] = Tracklet{iNextLayerClusterIndex, iCurrentLayerClusterIndex, nextCluster, currentCluster, targetRof, pivotRof};
                } else {
                  tracklets[rofFoundTrackletsOffsets[iCurrentLayerClusterIndex] + storedTracklets] = Tracklet{iCurrentLayerClusterIndex, iNextLayerClusterIndex, currentCluster, nextCluster, pivotRof, targetRof};
                }
              }
              ++storedTracklets;
            }
          }
        }
      }
    }
    if constexpr (EvalRun) {
      foundTracklets[iCurrentLayerClusterIndex] += storedTracklets;
    } else {
      rofFoundTrackletsOffsets[iCurrentLayerClusterIndex] += storedTracklets;
    }
  }
}

static void trackletSelectionKernelHost(
  const gsl::span<const Cluster> clusters0, // 0
  const gsl::span<const Cluster> clusters1, // 1
  gsl::span<unsigned char> usedClusters0,   // Layer 0
  gsl::span<unsigned char> usedClusters2,   // Layer 2
  const gsl::span<const Tracklet>& tracklets01,
  const gsl::span<const Tracklet>& tracklets12,
  bounded_vector<bool>& usedTracklets,
  const gsl::span<int> foundTracklets01,
  const gsl::span<int> foundTracklets12,
  bounded_vector<Line>& lines,
  const gsl::span<const o2::MCCompLabel>& trackletLabels,
  bounded_vector<o2::MCCompLabel>& linesLabels,
  const short targetRofId0,
  const short targetRofId2,
  bool safeWrites = false,
  const float tanLambdaCut = 0.025f,
  const float phiCut = 0.005f,
  const int maxTracklets = static_cast<int>(1e2))
{
  int offset01{0}, offset12{0};
  for (unsigned int iCurrentLayerClusterIndex{0}; iCurrentLayerClusterIndex < clusters1.size(); ++iCurrentLayerClusterIndex) {
    int validTracklets{0};
    for (int iTracklet12{offset12}; iTracklet12 < offset12 + foundTracklets12[iCurrentLayerClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{offset01}; iTracklet01 < offset01 + foundTracklets01[iCurrentLayerClusterIndex]; ++iTracklet01) {
        if (usedTracklets[iTracklet01]) {
          continue;
        }

        const auto& tracklet01{tracklets01[iTracklet01]};
        const auto& tracklet12{tracklets12[iTracklet12]};

        if (tracklet01.rof[0] != targetRofId0 || tracklet12.rof[1] != targetRofId2) {
          continue;
        }

        const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(tracklet01.tanLambda - tracklet12.tanLambda)};
        const float deltaPhi{o2::gpu::GPUCommonMath::Abs(math_utils::smallestAngleDifference(tracklet01.phi, tracklet12.phi))};
        if (deltaTanLambda < tanLambdaCut && deltaPhi < phiCut && validTracklets != maxTracklets) {
          if (safeWrites) {
            __atomic_store_n(&usedClusters0[tracklet01.firstClusterIndex], 1, __ATOMIC_RELAXED);
            __atomic_store_n(&usedClusters2[tracklet12.secondClusterIndex], 1, __ATOMIC_RELAXED);
          } else {
            usedClusters0[tracklet01.firstClusterIndex] = 1;
            usedClusters2[tracklet12.secondClusterIndex] = 1;
          }
          usedTracklets[iTracklet01] = true;
          lines.emplace_back(tracklet01, clusters0.data(), clusters1.data());
          if (!trackletLabels.empty()) {
            linesLabels.emplace_back(trackletLabels[iTracklet01]);
          }
          ++validTracklets;
        }
      }
    }
    offset01 += foundTracklets01[iCurrentLayerClusterIndex];
    offset12 += foundTracklets12[iCurrentLayerClusterIndex];
  }
}

template <int nLayers>
void VertexerTraits<nLayers>::updateVertexingParameters(const std::vector<VertexingParameters>& vrtPar, const TimeFrameGPUParameters& tfPar)
{
  mVrtParams = vrtPar;
  mIndexTableUtils.setTrackingParameters(vrtPar[0]);
  for (auto& par : mVrtParams) {
    par.phiSpan = static_cast<int>(std::ceil(mIndexTableUtils.getNphiBins() * par.phiCut / o2::constants::math::TwoPI));
    par.zSpan = static_cast<int>(std::ceil(par.zCut * mIndexTableUtils.getInverseZCoordinate(0)));
  }
}

// Main functions
template <int nLayers>
void VertexerTraits<nLayers>::computeTracklets(const int iteration)
{
  mTaskArena->execute([&] {
    tbb::parallel_for(0, mTimeFrame->getNrof(), [&](const short pivotRofId) {
      bool skipROF = iteration && (int)mTimeFrame->getPrimaryVertices(pivotRofId).size() > mVrtParams[iteration].vertPerRofThreshold;
      short startROF{std::max((short)0, static_cast<short>(pivotRofId - mVrtParams[iteration].deltaRof))};
      short endROF{std::min(static_cast<short>(mTimeFrame->getNrof()), static_cast<short>(pivotRofId + mVrtParams[iteration].deltaRof + 1))};
      for (auto targetRofId = startROF; targetRofId < endROF; ++targetRofId) {
        trackleterKernelHost<TrackletMode::Layer0Layer1, true>(
          !skipROF ? mTimeFrame->getClustersOnLayer(targetRofId, 0) : gsl::span<Cluster>(), // Clusters to be matched with the next layer in target rof
          !skipROF ? mTimeFrame->getClustersOnLayer(pivotRofId, 1) : gsl::span<Cluster>(),  // Clusters to be matched with the current layer in pivot rof
          mTimeFrame->getUsedClustersROF(targetRofId, 0),                                   // Span of the used clusters in the target rof
          mTimeFrame->getIndexTable(targetRofId, 0).data(),                                 // Index table to access the data on the next layer in target rof
          mVrtParams[iteration].phiCut,
          mTimeFrame->getTracklets()[0],                   // Flat tracklet buffer
          mTimeFrame->getNTrackletsCluster(pivotRofId, 0), // Span of the number of tracklets per each cluster in pivot rof
          mIndexTableUtils,
          pivotRofId,
          targetRofId,
          gsl::span<int>(), // Offset in the tracklet buffer
          mVrtParams[iteration].maxTrackletsPerCluster);
        trackleterKernelHost<TrackletMode::Layer1Layer2, true>(
          !skipROF ? mTimeFrame->getClustersOnLayer(targetRofId, 2) : gsl::span<Cluster>(),
          !skipROF ? mTimeFrame->getClustersOnLayer(pivotRofId, 1) : gsl::span<Cluster>(),
          mTimeFrame->getUsedClustersROF(targetRofId, 2),
          mTimeFrame->getIndexTable(targetRofId, 2).data(),
          mVrtParams[iteration].phiCut,
          mTimeFrame->getTracklets()[1],
          mTimeFrame->getNTrackletsCluster(pivotRofId, 1), // Span of the number of tracklets per each cluster in pivot rof
          mIndexTableUtils,
          pivotRofId,
          targetRofId,
          gsl::span<int>(), // Offset in the tracklet buffer
          mVrtParams[iteration].maxTrackletsPerCluster);
      }
      mTimeFrame->getNTrackletsROF(pivotRofId, 0) = std::accumulate(mTimeFrame->getNTrackletsCluster(pivotRofId, 0).begin(), mTimeFrame->getNTrackletsCluster(pivotRofId, 0).end(), 0);
      mTimeFrame->getNTrackletsROF(pivotRofId, 1) = std::accumulate(mTimeFrame->getNTrackletsCluster(pivotRofId, 1).begin(), mTimeFrame->getNTrackletsCluster(pivotRofId, 1).end(), 0);
    });

    mTimeFrame->computeTrackletsPerROFScans();
    if (auto tot0 = mTimeFrame->getTotalTrackletsTF(0), tot1 = mTimeFrame->getTotalTrackletsTF(1);
        tot0 == 0 || tot1 == 0) {
      return;
    } else {
      mTimeFrame->getTracklets()[0].resize(tot0);
      mTimeFrame->getTracklets()[1].resize(tot1);
    }

    tbb::parallel_for(0, mTimeFrame->getNrof(), [&](const short pivotRofId) {
      bool skipROF = iteration && (int)mTimeFrame->getPrimaryVertices(pivotRofId).size() > mVrtParams[iteration].vertPerRofThreshold;
      short startROF{std::max((short)0, static_cast<short>(pivotRofId - mVrtParams[iteration].deltaRof))};
      short endROF{std::min(static_cast<short>(mTimeFrame->getNrof()), static_cast<short>(pivotRofId + mVrtParams[iteration].deltaRof + 1))};
      auto mobileOffset0 = mTimeFrame->getNTrackletsROF(pivotRofId, 0);
      auto mobileOffset1 = mTimeFrame->getNTrackletsROF(pivotRofId, 1);
      for (auto targetRofId = startROF; targetRofId < endROF; ++targetRofId) {
        trackleterKernelHost<TrackletMode::Layer0Layer1, false>(
          !skipROF ? mTimeFrame->getClustersOnLayer(targetRofId, 0) : gsl::span<Cluster>(),
          !skipROF ? mTimeFrame->getClustersOnLayer(pivotRofId, 1) : gsl::span<Cluster>(),
          mTimeFrame->getUsedClustersROF(targetRofId, 0),
          mTimeFrame->getIndexTable(targetRofId, 0).data(),
          mVrtParams[iteration].phiCut,
          mTimeFrame->getTracklets()[0],
          mTimeFrame->getNTrackletsCluster(pivotRofId, 0),
          mIndexTableUtils,
          pivotRofId,
          targetRofId,
          mTimeFrame->getExclusiveNTrackletsCluster(pivotRofId, 0),
          mVrtParams[iteration].maxTrackletsPerCluster);
        trackleterKernelHost<TrackletMode::Layer1Layer2, false>(
          !skipROF ? mTimeFrame->getClustersOnLayer(targetRofId, 2) : gsl::span<Cluster>(),
          !skipROF ? mTimeFrame->getClustersOnLayer(pivotRofId, 1) : gsl::span<Cluster>(),
          mTimeFrame->getUsedClustersROF(targetRofId, 2),
          mTimeFrame->getIndexTable(targetRofId, 2).data(),
          mVrtParams[iteration].phiCut,
          mTimeFrame->getTracklets()[1],
          mTimeFrame->getNTrackletsCluster(pivotRofId, 1),
          mIndexTableUtils,
          pivotRofId,
          targetRofId,
          mTimeFrame->getExclusiveNTrackletsCluster(pivotRofId, 1),
          mVrtParams[iteration].maxTrackletsPerCluster);
      }
    });
  });

  /// Create tracklets labels for L0-L1, information is as flat as in tracklets vector (no rofId)
  if (mTimeFrame->hasMCinformation()) {
    for (const auto& trk : mTimeFrame->getTracklets()[0]) {
      o2::MCCompLabel label;
      if (!trk.isEmpty()) {
        int sortedId0{mTimeFrame->getSortedIndex(trk.rof[0], 0, trk.firstClusterIndex)};
        int sortedId1{mTimeFrame->getSortedIndex(trk.rof[1], 1, trk.secondClusterIndex)};
        for (const auto& lab0 : mTimeFrame->getClusterLabels(0, mTimeFrame->getClusters()[0][sortedId0].clusterId)) {
          for (const auto& lab1 : mTimeFrame->getClusterLabels(1, mTimeFrame->getClusters()[1][sortedId1].clusterId)) {
            if (lab0 == lab1 && lab0.isValid()) {
              label = lab0;
              break;
            }
          }
          if (label.isValid()) {
            break;
          }
        }
      }
      mTimeFrame->getTrackletsLabel(0).emplace_back(label);
    }
  }

#ifdef VTX_DEBUG
  debugComputeTracklets(iteration);
#endif
}

template <int nLayers>
void VertexerTraits<nLayers>::computeTrackletMatching(const int iteration)
{
  mTaskArena->execute([&] {
    tbb::combinable<int> totalLines{0};
    tbb::parallel_for(
      tbb::blocked_range<short>(0, (short)mTimeFrame->getNrof()),
      [&](const tbb::blocked_range<short>& Rofs) {
        for (short pivotRofId = Rofs.begin(); pivotRofId < Rofs.end(); ++pivotRofId) {
          if (iteration && (int)mTimeFrame->getPrimaryVertices(pivotRofId).size() > mVrtParams[iteration].vertPerRofThreshold) {
            continue;
          }
          if (mTimeFrame->getFoundTracklets(pivotRofId, 0).empty()) {
            continue;
          }
          mTimeFrame->getLines(pivotRofId).reserve(mTimeFrame->getNTrackletsCluster(pivotRofId, 0).size());
          bounded_vector<bool> usedTracklets(mTimeFrame->getFoundTracklets(pivotRofId, 0).size(), false, mMemoryPool.get());
          short startROF{std::max((short)0, static_cast<short>(pivotRofId - mVrtParams[iteration].deltaRof))};
          short endROF{std::min(static_cast<short>(mTimeFrame->getNrof()), static_cast<short>(pivotRofId + mVrtParams[iteration].deltaRof + 1))};

          // needed only if multi-threaded using deltaRof and only at the overlap edges of the ranges
          bool safeWrite = mTaskArena->max_concurrency() > 1 && mVrtParams[iteration].deltaRof != 0 && ((Rofs.begin() - startROF < 0) || (endROF - Rofs.end() > 0));

          for (short targetRofId0 = startROF; targetRofId0 < endROF; ++targetRofId0) {
            for (short targetRofId2 = startROF; targetRofId2 < endROF; ++targetRofId2) {
              if (std::abs(targetRofId0 - targetRofId2) > mVrtParams[iteration].deltaRof) { // do not allow over 3 ROFs
                continue;
              }
              trackletSelectionKernelHost(
                mTimeFrame->getClustersOnLayer(targetRofId0, 0),
                mTimeFrame->getClustersOnLayer(pivotRofId, 1),
                mTimeFrame->getUsedClustersROF(targetRofId0, 0),
                mTimeFrame->getUsedClustersROF(targetRofId2, 2),
                mTimeFrame->getFoundTracklets(pivotRofId, 0),
                mTimeFrame->getFoundTracklets(pivotRofId, 1),
                usedTracklets,
                mTimeFrame->getNTrackletsCluster(pivotRofId, 0),
                mTimeFrame->getNTrackletsCluster(pivotRofId, 1),
                mTimeFrame->getLines(pivotRofId),
                mTimeFrame->getLabelsFoundTracklets(pivotRofId, 0),
                mTimeFrame->getLinesLabel(pivotRofId),
                targetRofId0,
                targetRofId2,
                safeWrite,
                mVrtParams[iteration].tanLambdaCut,
                mVrtParams[iteration].phiCut);
            }
          }
          totalLines.local() += mTimeFrame->getLines(pivotRofId).size();
        }
      });
    mTimeFrame->setNLinesTotal(totalLines.combine(std::plus<int>()));
  });

#ifdef VTX_DEBUG
  debugComputeTrackletMatching(iteration);
#endif

  // from here on we do not use tracklets from L1-2 anymore, so let's free them
  deepVectorClear(mTimeFrame->getTracklets()[1]);
}

template <int nLayers>
void VertexerTraits<nLayers>::computeVertices(const int iteration)
{
  auto nsigmaCut{std::min(mVrtParams[iteration].vertNsigmaCut * mVrtParams[iteration].vertNsigmaCut * (mVrtParams[iteration].vertRadiusSigma * mVrtParams[iteration].vertRadiusSigma + mVrtParams[iteration].trackletSigma * mVrtParams[iteration].trackletSigma), 1.98f)};
  bounded_vector<Vertex> vertices(mMemoryPool.get());
  bounded_vector<std::pair<o2::MCCompLabel, float>> polls(mMemoryPool.get());
  bounded_vector<o2::MCCompLabel> contLabels(mMemoryPool.get());
  bounded_vector<int> noClustersVec(mTimeFrame->getNrof(), 0, mMemoryPool.get());
  for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
    if (iteration && (int)mTimeFrame->getPrimaryVertices(rofId).size() > mVrtParams[iteration].vertPerRofThreshold) {
      continue;
    }
    const int numTracklets{static_cast<int>(mTimeFrame->getLines(rofId).size())};

    bounded_vector<bool> usedTracklets(numTracklets, false, mMemoryPool.get());
    for (int line1{0}; line1 < numTracklets; ++line1) {
      if (usedTracklets[line1]) {
        continue;
      }
      for (int line2{line1 + 1}; line2 < numTracklets; ++line2) {
        if (usedTracklets[line2]) {
          continue;
        }
        auto dca{Line::getDCA(mTimeFrame->getLines(rofId)[line1], mTimeFrame->getLines(rofId)[line2])};
        if (dca < mVrtParams[iteration].pairCut) {
          mTimeFrame->getTrackletClusters(rofId).emplace_back(line1, mTimeFrame->getLines(rofId)[line1], line2, mTimeFrame->getLines(rofId)[line2]);
          std::array<float, 3> tmpVertex{mTimeFrame->getTrackletClusters(rofId).back().getVertex()};
          if (tmpVertex[0] * tmpVertex[0] + tmpVertex[1] * tmpVertex[1] > 4.f) {
            mTimeFrame->getTrackletClusters(rofId).pop_back();
            break;
          }
          usedTracklets[line1] = true;
          usedTracklets[line2] = true;
          for (int tracklet3{0}; tracklet3 < numTracklets; ++tracklet3) {
            if (usedTracklets[tracklet3]) {
              continue;
            }
            if (Line::getDistanceFromPoint(mTimeFrame->getLines(rofId)[tracklet3], tmpVertex) < mVrtParams[iteration].pairCut) {
              mTimeFrame->getTrackletClusters(rofId).back().add(tracklet3, mTimeFrame->getLines(rofId)[tracklet3]);
              usedTracklets[tracklet3] = true;
              tmpVertex = mTimeFrame->getTrackletClusters(rofId).back().getVertex();
            }
          }
          break;
        }
      }
    }
    if (mVrtParams[iteration].allowSingleContribClusters) {
      auto beamLine = Line{{mTimeFrame->getBeamX(), mTimeFrame->getBeamY(), -50.f}, {mTimeFrame->getBeamX(), mTimeFrame->getBeamY(), 50.f}}; // use beam position as contributor
      for (size_t iLine{0}; iLine < numTracklets; ++iLine) {
        if (!usedTracklets[iLine]) {
          auto dca = Line::getDCA(mTimeFrame->getLines(rofId)[iLine], beamLine);
          if (dca < mVrtParams[iteration].pairCut) {
            mTimeFrame->getTrackletClusters(rofId).emplace_back(iLine, mTimeFrame->getLines(rofId)[iLine], -1, beamLine); // beamline must be passed as second line argument
          }
        }
      }
    }

    // Cluster merging
    std::sort(mTimeFrame->getTrackletClusters(rofId).begin(), mTimeFrame->getTrackletClusters(rofId).end(),
              [](ClusterLines& cluster1, ClusterLines& cluster2) { return cluster1.getSize() > cluster2.getSize(); });
    noClustersVec[rofId] = static_cast<int>(mTimeFrame->getTrackletClusters(rofId).size());
    for (int iCluster1{0}; iCluster1 < noClustersVec[rofId]; ++iCluster1) {
      std::array<float, 3> vertex1{mTimeFrame->getTrackletClusters(rofId)[iCluster1].getVertex()};
      std::array<float, 3> vertex2{};
      for (int iCluster2{iCluster1 + 1}; iCluster2 < noClustersVec[rofId]; ++iCluster2) {
        vertex2 = mTimeFrame->getTrackletClusters(rofId)[iCluster2].getVertex();
        if (o2::gpu::GPUCommonMath::Abs(vertex1[2] - vertex2[2]) < mVrtParams[iteration].clusterCut) {
          float distance{(vertex1[0] - vertex2[0]) * (vertex1[0] - vertex2[0]) +
                         (vertex1[1] - vertex2[1]) * (vertex1[1] - vertex2[1]) +
                         (vertex1[2] - vertex2[2]) * (vertex1[2] - vertex2[2])};
          if (distance < mVrtParams[iteration].pairCut * mVrtParams[iteration].pairCut) {
            for (auto label : mTimeFrame->getTrackletClusters(rofId)[iCluster2].getLabels()) {
              mTimeFrame->getTrackletClusters(rofId)[iCluster1].add(label, mTimeFrame->getLines(rofId)[label]);
              vertex1 = mTimeFrame->getTrackletClusters(rofId)[iCluster1].getVertex();
            }
            mTimeFrame->getTrackletClusters(rofId).erase(mTimeFrame->getTrackletClusters(rofId).begin() + iCluster2);
            --iCluster2;
            --noClustersVec[rofId];
          }
        }
      }
    }
  }
  for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
    std::sort(mTimeFrame->getTrackletClusters(rofId).begin(), mTimeFrame->getTrackletClusters(rofId).end(),
              [](const ClusterLines& cluster1, const ClusterLines& cluster2) { return cluster1.getSize() > cluster2.getSize(); }); // ensure clusters are ordered by contributors, so that we can cat after the first.
    bool atLeastOneFound{false};
    for (int iCluster{0}; iCluster < noClustersVec[rofId]; ++iCluster) {
      bool lowMultCandidate{false};
      double beamDistance2{(mTimeFrame->getBeamX() - mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[0]) * (mTimeFrame->getBeamX() - mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[0]) +
                           (mTimeFrame->getBeamY() - mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[1]) * (mTimeFrame->getBeamY() - mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[1])};
      if (atLeastOneFound && (lowMultCandidate = mTimeFrame->getTrackletClusters(rofId)[iCluster].getSize() < mVrtParams[iteration].clusterContributorsCut)) { // We might have pile up with nContr > cut.
        lowMultCandidate &= (beamDistance2 < mVrtParams[iteration].lowMultBeamDistCut * mVrtParams[iteration].lowMultBeamDistCut);
        if (!lowMultCandidate) { // Not the first cluster and not a low multiplicity candidate, we can remove it
          mTimeFrame->getTrackletClusters(rofId).erase(mTimeFrame->getTrackletClusters(rofId).begin() + iCluster);
          noClustersVec[rofId]--;
          continue;
        }
      }

      if (beamDistance2 < nsigmaCut && o2::gpu::GPUCommonMath::Abs(mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[2]) < mVrtParams[iteration].maxZPositionAllowed) {
        atLeastOneFound = true;
        auto& vertex = vertices.emplace_back(o2::math_utils::Point3D<float>(mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[0],
                                                                            mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[1],
                                                                            mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[2]),
                                             mTimeFrame->getTrackletClusters(rofId)[iCluster].getRMS2(),          // Symm matrix. Diagonal: RMS2 components,
                                                                                                                  // off-diagonal: square mean of projections on planes.
                                             mTimeFrame->getTrackletClusters(rofId)[iCluster].getSize(),          // Contributors
                                             mTimeFrame->getTrackletClusters(rofId)[iCluster].getAvgDistance2()); // In place of chi2

        if (iteration) {
          vertex.setFlags(Vertex::UPCMode);
        }
        vertex.setTimeStamp(mTimeFrame->getTrackletClusters(rofId)[iCluster].getROF());
        if (mTimeFrame->hasMCinformation()) {
          bounded_vector<o2::MCCompLabel> labels(mMemoryPool.get());
          for (auto& index : mTimeFrame->getTrackletClusters(rofId)[iCluster].getLabels()) {
            labels.push_back(mTimeFrame->getLinesLabel(rofId)[index]); // then we can use nContributors from vertices to get the labels
          }
          polls.push_back(computeMain(labels));
          if (mVrtParams[iteration].outputContLabels) {
            contLabels.insert(contLabels.end(), labels.begin(), labels.end());
          }
        }
      }
    }
    if (!iteration) {
      mTimeFrame->addPrimaryVertices(vertices, iteration);
      if (mTimeFrame->hasMCinformation()) {
        mTimeFrame->addPrimaryVerticesLabels(polls);
        if (mVrtParams[iteration].outputContLabels) {
          mTimeFrame->addPrimaryVerticesContributorLabels(contLabels);
        }
      }
    } else {
      mTimeFrame->addPrimaryVerticesInROF(vertices, rofId, iteration);
      if (mTimeFrame->hasMCinformation()) {
        mTimeFrame->addPrimaryVerticesLabelsInROF(polls, rofId);
        if (mVrtParams[iteration].outputContLabels) {
          mTimeFrame->addPrimaryVerticesContributorLabelsInROF(contLabels, rofId);
        }
      }
    }
    if (vertices.empty() && !(iteration && (int)mTimeFrame->getPrimaryVertices(rofId).size() > mVrtParams[iteration].vertPerRofThreshold)) {
      mTimeFrame->getNoVertexROF()++;
    }
    vertices.clear();
    polls.clear();
  }

#ifdef VTX_DEBUG
  debugComputeVertices(iteration);
#endif
}

template <int nLayers>
void VertexerTraits<nLayers>::addTruthSeedingVertices()
{
  LOGP(info, "Using truth seeds as vertices; will skip computations");
  mTimeFrame->resetRofPV();
  const auto dc = o2::steer::DigitizationContext::loadFromFile("collisioncontext.root");
  const auto irs = dc->getEventRecords();
  int64_t roFrameBiasInBC = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance().roFrameBiasInBC;
  int64_t roFrameLengthInBC = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance().roFrameLengthInBC;
  o2::steer::MCKinematicsReader mcReader(dc);
  struct VertInfo {
    bounded_vector<Vertex> vertices;
    bounded_vector<int> srcs;
    bounded_vector<int> events;
  };
  std::map<int, VertInfo> vertices;
  for (int iSrc{0}; iSrc < mcReader.getNSources(); ++iSrc) {
    auto eveId2colId = dc->getCollisionIndicesForSource(iSrc);
    for (int iEve{0}; iEve < mcReader.getNEvents(iSrc); ++iEve) {
      const auto& ir = irs[eveId2colId[iEve]];
      if (!ir.isDummy()) { // do we need this, is this for diffractive events?
        const auto& eve = mcReader.getMCEventHeader(iSrc, iEve);
        int rofId = ((ir - raw::HBFUtils::Instance().getFirstSampledTFIR()).toLong() - roFrameBiasInBC) / roFrameLengthInBC;
        if (!vertices.contains(rofId)) {
          vertices[rofId] = {
            .vertices = bounded_vector<Vertex>(mMemoryPool.get()),
            .srcs = bounded_vector<int>(mMemoryPool.get()),
            .events = bounded_vector<int>(mMemoryPool.get()),
          };
        }
        Vertex vert;
        vert.setTimeStamp(rofId);
        vert.setNContributors(std::ranges::count_if(mcReader.getTracks(iSrc, iEve), [](const auto& trk) {
          return trk.isPrimary() && trk.GetPt() > 0.2 && std::abs(trk.GetEta()) < 1.3;
        }));
        vert.setXYZ((float)eve.GetX(), (float)eve.GetY(), (float)eve.GetZ());
        vert.setChi2(1);
        constexpr float cov = 50e-9;
        vert.setCov(cov, cov, cov, cov, cov, cov);
        vertices[rofId].vertices.push_back(vert);
        vertices[rofId].srcs.push_back(iSrc);
        vertices[rofId].events.push_back(iEve);
      }
    }
  }
  size_t nVerts{0};
  for (int iROF{0}; iROF < mTimeFrame->getNrof(); ++iROF) {
    bounded_vector<Vertex> verts(mMemoryPool.get());
    bounded_vector<std::pair<o2::MCCompLabel, float>> polls(mMemoryPool.get());
    if (vertices.contains(iROF)) {
      const auto& vertInfo = vertices[iROF];
      verts = vertInfo.vertices;
      nVerts += verts.size();
      for (size_t i{0}; i < verts.size(); ++i) {
        o2::MCCompLabel lbl(o2::MCCompLabel::maxTrackID(), vertInfo.events[i], vertInfo.srcs[i], false);
        polls.emplace_back(lbl, 1.f);
      }
    } else {
      mTimeFrame->getNoVertexROF()++;
    }
    mTimeFrame->addPrimaryVertices(verts, 0);
    mTimeFrame->addPrimaryVerticesLabels(polls);
  }
  LOGP(info, "Found {}/{} ROFs with {} vertices -> <NV>={:.2f}", vertices.size(), mTimeFrame->getNrof(), nVerts, (float)nVerts / (float)vertices.size());
}

template <int nLayers>
void VertexerTraits<nLayers>::setNThreads(int n, std::shared_ptr<tbb::task_arena>& arena)
{
#if defined(VTX_DEBUG)
  LOGP(info, "Vertexer with debug output forcing single thread");
  mTaskArena = std::make_shared<tbb::task_arena>(1);
#else
  if (arena == nullptr) {
    mTaskArena = std::make_shared<tbb::task_arena>(std::abs(n));
    LOGP(info, "Setting seeding vertexer with {} threads.", n);
  } else {
    mTaskArena = arena;
    LOGP(info, "Attaching vertexer to calling thread's arena");
  }
#endif
}

template <int nLayers>
void VertexerTraits<nLayers>::debugComputeTracklets(int iteration)
{
  auto stream = new utils::TreeStreamRedirector("artefacts_tf.root", "recreate");
  LOGP(info, "writing debug output for computeTracklets");
  for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
    const auto& strk0 = mTimeFrame->getFoundTracklets(rofId, 0);
    std::vector<Tracklet> trk0(strk0.begin(), strk0.end());
    const auto& strk1 = mTimeFrame->getFoundTracklets(rofId, 1);
    std::vector<Tracklet> trk1(strk1.begin(), strk1.end());
    (*stream) << "tracklets"
              << "Tracklets0=" << trk0
              << "Tracklets1=" << trk1
              << "iteration=" << iteration
              << "\n";
  }
  stream->Close();
  delete stream;
}

template <int nLayers>
void VertexerTraits<nLayers>::debugComputeTrackletMatching(int iteration)
{
  auto stream = new utils::TreeStreamRedirector("artefacts_tf.root", "update");
  LOGP(info, "writing debug output for computeTrackletMatching");
  for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
    (*stream) << "lines"
              << "Lines=" << toSTDVector(mTimeFrame->getLines(rofId))
              << "NTrackletCluster01=" << mTimeFrame->getNTrackletsCluster(rofId, 0)
              << "NTrackletCluster12=" << mTimeFrame->getNTrackletsCluster(rofId, 1)
              << "iteration=" << iteration
              << "\n";
  }

  if (mTimeFrame->hasMCinformation()) {
    LOGP(info, "\tdumping also MC information");
    const auto dc = o2::steer::DigitizationContext::loadFromFile("collisioncontext.root");
    const auto irs = dc->getEventRecords();
    int64_t roFrameBiasInBC = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance().roFrameBiasInBC;
    int64_t roFrameLengthInBC = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance().roFrameLengthInBC;
    o2::steer::MCKinematicsReader mcReader(dc);

    std::map<int, int> eve2BcInROF, bcInRofNEve;
    for (int iSrc{0}; iSrc < mcReader.getNSources(); ++iSrc) {
      auto eveId2colId = dc->getCollisionIndicesForSource(iSrc);
      for (int iEve{0}; iEve < mcReader.getNEvents(iSrc); ++iEve) {
        const auto& ir = irs[eveId2colId[iEve]];
        if (!ir.isDummy()) { // do we need this, is this for diffractive events?
          const auto& eve = mcReader.getMCEventHeader(iSrc, iEve);
          const int bcInROF = ((ir - raw::HBFUtils::Instance().getFirstSampledTFIR()).toLong() - roFrameBiasInBC) % roFrameLengthInBC;
          eve2BcInROF[iEve] = bcInROF;
          ++bcInRofNEve[bcInROF];
        }
      }
    }

    std::unordered_map<int, int> bcROFNTracklets01, bcROFNTracklets12;
    std::vector<std::vector<int>> tracklet01BC, tracklet12BC;
    for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
      { // 0-1
        const auto& tracklet01 = mTimeFrame->getFoundTracklets(rofId, 0);
        const auto& lbls01 = mTimeFrame->getLabelsFoundTracklets(rofId, 0);
        auto& trkls01 = tracklet01BC.emplace_back();
        for (int iTrklt{0}; iTrklt < (int)tracklet01.size(); ++iTrklt) {
          const auto& tracklet = tracklet01[iTrklt];
          const auto& lbl = lbls01[iTrklt];
          if (lbl.isCorrect()) {
            ++bcROFNTracklets01[eve2BcInROF[lbl.getEventID()]];
            trkls01.push_back(eve2BcInROF[lbl.getEventID()]);
          } else {
            trkls01.push_back(-1);
          }
        }
      }
      { // 1-2 computed on the fly!
        const auto& tracklet12 = mTimeFrame->getFoundTracklets(rofId, 1);
        auto& trkls12 = tracklet12BC.emplace_back();
        for (int iTrklt{0}; iTrklt < (int)tracklet12.size(); ++iTrklt) {
          const auto& tracklet = tracklet12[iTrklt];
          o2::MCCompLabel label;

          int sortedId1{mTimeFrame->getSortedIndex(tracklet.rof[0], 1, tracklet.firstClusterIndex)};
          int sortedId2{mTimeFrame->getSortedIndex(tracklet.rof[1], 2, tracklet.secondClusterIndex)};
          for (const auto& lab1 : mTimeFrame->getClusterLabels(1, mTimeFrame->getClusters()[1][sortedId1].clusterId)) {
            for (const auto& lab2 : mTimeFrame->getClusterLabels(2, mTimeFrame->getClusters()[2][sortedId2].clusterId)) {
              if (lab1 == lab2 && lab1.isValid()) {
                label = lab1;
                break;
              }
            }
            if (label.isValid()) {
              break;
            }
          }

          if (label.isCorrect()) {
            ++bcROFNTracklets12[eve2BcInROF[label.getEventID()]];
            trkls12.push_back(eve2BcInROF[label.getEventID()]);
          } else {
            trkls12.push_back(-1);
          }
        }
      }
    }
    LOGP(info, "\tdumping ntracklets/RofBC ({})", bcInRofNEve.size());
    for (const auto& [bcInRof, neve] : bcInRofNEve) {
      (*stream) << "ntracklets"
                << "bcInROF=" << bcInRof
                << "ntrkl01=" << bcROFNTracklets01[bcInRof]
                << "ntrkl12=" << bcROFNTracklets12[bcInRof]
                << "neve=" << neve
                << "iteration=" << iteration
                << "\n";
    }

    std::unordered_map<int, int> bcROFNLines;
    for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
      const auto& lines = mTimeFrame->getLines(rofId);
      const auto& lbls = mTimeFrame->getLinesLabel(rofId);
      for (int iLine{0}; iLine < (int)lines.size(); ++iLine) {
        const auto& line = lines[iLine];
        const auto& lbl = lbls[iLine];
        if (lbl.isCorrect()) {
          ++bcROFNLines[eve2BcInROF[lbl.getEventID()]];
        }
      }
    }

    LOGP(info, "\tdumping nlines/RofBC");
    for (const auto& [bcInRof, neve] : bcInRofNEve) {
      (*stream) << "nlines"
                << "bcInROF=" << bcInRof
                << "nline=" << bcROFNLines[bcInRof]
                << "neve=" << neve
                << "iteration=" << iteration
                << "\n";
    }
  }
  stream->Close();
  delete stream;
}

template <int nLayers>
void VertexerTraits<nLayers>::debugComputeVertices(int iteration)
{
  auto stream = new utils::TreeStreamRedirector("artefacts_tf.root", "update");
  LOGP(info, "writing debug output for computeVertices");
  for (auto rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
    (*stream) << "clusterlines"
              << "clines_post=" << toSTDVector(mTimeFrame->getTrackletClusters(rofId))
              << "iteration=" << iteration
              << "\n";
  }

  if (mTimeFrame->hasMCinformation()) {
    LOGP(info, "\tdumping also MC information");
    const auto dc = o2::steer::DigitizationContext::loadFromFile("collisioncontext.root");
    const auto irs = dc->getEventRecords();
    int64_t roFrameBiasInBC = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance().roFrameBiasInBC;
    int64_t roFrameLengthInBC = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance().roFrameLengthInBC;
    o2::steer::MCKinematicsReader mcReader(dc);

    std::map<int, int> eve2BcInROF, bcInRofNEve;
    for (int iSrc{0}; iSrc < mcReader.getNSources(); ++iSrc) {
      auto eveId2colId = dc->getCollisionIndicesForSource(iSrc);
      for (int iEve{0}; iEve < mcReader.getNEvents(iSrc); ++iEve) {
        const auto& ir = irs[eveId2colId[iEve]];
        if (!ir.isDummy()) { // do we need this, is this for diffractive events?
          const auto& eve = mcReader.getMCEventHeader(iSrc, iEve);
          const int bcInROF = ((ir - raw::HBFUtils::Instance().getFirstSampledTFIR()).toLong() - roFrameBiasInBC) % roFrameLengthInBC;
          eve2BcInROF[iEve] = bcInROF;
          ++bcInRofNEve[bcInROF];
        }
      }
    }

    std::unordered_map<int, int> bcROFNVtx;
    std::unordered_map<int, float> bcROFNPur;
    std::unordered_map<o2::MCCompLabel, size_t> uniqueVertices;
    for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
      const auto& pvs = mTimeFrame->getPrimaryVertices(rofId);
      const auto& lblspv = mTimeFrame->getPrimaryVerticesMCRecInfo(rofId);
      for (int i{0}; i < (int)pvs.size(); ++i) {
        const auto& pv = pvs[i];
        const auto& [lbl, pur] = lblspv[i];
        if (lbl.isCorrect()) {
          ++uniqueVertices[lbl];
          ++bcROFNVtx[eve2BcInROF[lbl.getEventID()]];
          bcROFNPur[eve2BcInROF[lbl.getEventID()]] += pur;
        }
      }
    }

    std::unordered_map<int, int> bcROFNUVtx, bcROFNCVtx;
    for (const auto& [k, _] : eve2BcInROF) {
      bcROFNUVtx[k] = bcROFNCVtx[k] = 0;
    }

    for (const auto& [lbl, c] : uniqueVertices) {
      if (c <= 1) {
        ++bcROFNUVtx[eve2BcInROF[lbl.getEventID()]];
      } else {
        ++bcROFNCVtx[eve2BcInROF[lbl.getEventID()]];
      }
    }

    LOGP(info, "\tdumping nvtx/RofBC");
    for (const auto& [bcInRof, neve] : bcInRofNEve) {
      (*stream) << "nvtx"
                << "bcInROF=" << bcInRof
                << "nvtx=" << bcROFNVtx[bcInRof]   // all vertices
                << "nuvtx=" << bcROFNUVtx[bcInRof] // unique vertices
                << "ncvtx=" << bcROFNCVtx[bcInRof] // cloned vertices
                << "npur=" << bcROFNPur[bcInRof]
                << "neve=" << neve
                << "iteration=" << iteration
                << "\n";
    }

    // check dist of clones
    std::unordered_map<o2::MCCompLabel, std::vector<Vertex>> cVtx;
    for (int rofId{0}; rofId < mTimeFrame->getNrof(); ++rofId) {
      const auto& pvs = mTimeFrame->getPrimaryVertices(rofId);
      const auto& lblspv = mTimeFrame->getPrimaryVerticesMCRecInfo(rofId);
      for (int i{0}; i < (int)pvs.size(); ++i) {
        const auto& pv = pvs[i];
        const auto& [lbl, pur] = lblspv[i];
        if (lbl.isCorrect() && uniqueVertices.contains(lbl) && uniqueVertices[lbl] > 1) {
          if (!cVtx.contains(lbl)) {
            cVtx[lbl] = std::vector<Vertex>();
          }
          cVtx[lbl].push_back(pv);
        }
      }
    }

    for (auto& [_, vertices] : cVtx) {
      std::sort(vertices.begin(), vertices.end(), [](const Vertex& a, const Vertex& b) { return a.getNContributors() > b.getNContributors(); });
      for (int i{0}; i < (int)vertices.size(); ++i) {
        const auto vtx = vertices[i];
        (*stream) << "cvtx"
                  << "vertex=" << vtx
                  << "i=" << i
                  << "dx=" << vertices[0].getX() - vtx.getX()
                  << "dy=" << vertices[0].getY() - vtx.getY()
                  << "dz=" << vertices[0].getZ() - vtx.getZ()
                  << "drof=" << vertices[0].getTimeStamp().getTimeStamp() - vtx.getTimeStamp().getTimeStamp()
                  << "dnc=" << vertices[0].getNContributors() - vtx.getNContributors()
                  << "iteration=" << iteration
                  << "\n";
      }
    }
  }
  stream->Close();
  delete stream;
}

template class VertexerTraits<7>;
} // namespace o2::its
