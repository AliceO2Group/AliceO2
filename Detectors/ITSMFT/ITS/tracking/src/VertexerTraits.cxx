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
#include <algorithm>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/combinable.h>

#include "ITStracking/VertexerTraits.h"
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/Tracklet.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "Steer/MCKinematicsReader.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "DetectorsRaw/HBFUtils.h"

namespace o2::its
{

template <TrackletMode Mode, bool EvalRun, int NLayers>
static void trackleterKernelHost(
  const gsl::span<const Cluster>& clustersNextLayer,    // 0 2
  const gsl::span<const Cluster>& clustersCurrentLayer, // 1 1
  const gsl::span<uint8_t>& usedClustersNextLayer,      // 0 2
  int* indexTableNext,
  const float phiCut,
  bounded_vector<Tracklet>& tracklets,
  gsl::span<int> foundTracklets,
  const IndexTableUtils<NLayers>& utils,
  const TimeEstBC& timErr,
  gsl::span<int> rofFoundTrackletsOffsets, // we want to change those, to keep track of the offset in deltaRof>0
  const int maxTrackletsPerCluster = static_cast<int>(2e3))
{
  const int PhiBins{utils.getNphiBins()};
  const int ZBins{utils.getNzBins()};
  // loop on layer1 clusters
  for (int iCurrentLayerClusterIndex = 0; iCurrentLayerClusterIndex < clustersCurrentLayer.size(); ++iCurrentLayerClusterIndex) {
    int storedTracklets{0};
    const Cluster& currentCluster{clustersCurrentLayer[iCurrentLayerClusterIndex]};
    const int4 selectedBinsRect{VertexerTraits<NLayers>::getBinsRect(currentCluster, (int)Mode, 0.f, 50.f, phiCut / 2, utils)};
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
                  tracklets[rofFoundTrackletsOffsets[iCurrentLayerClusterIndex] + storedTracklets] = Tracklet{iNextLayerClusterIndex, iCurrentLayerClusterIndex, nextCluster, currentCluster, timErr};
                } else {
                  tracklets[rofFoundTrackletsOffsets[iCurrentLayerClusterIndex] + storedTracklets] = Tracklet{iCurrentLayerClusterIndex, iNextLayerClusterIndex, currentCluster, nextCluster, timErr};
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
  const TimeEstBC& targetRofTime0,
  const TimeEstBC& targetRofTime2,
  const float tanLambdaCut = 0.025f,
  const float phiCut = 0.005f,
  const int maxTracklets = 100)
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
        if (!tracklet01.getTimeStamp().isCompatible(targetRofTime0) ||
            !tracklet12.getTimeStamp().isCompatible(targetRofTime2) ||
            !tracklet01.getTimeStamp().isCompatible(tracklet12.getTimeStamp())) {
          continue;
        }

        const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(tracklet01.tanLambda - tracklet12.tanLambda)};
        const float deltaPhi{o2::gpu::GPUCommonMath::Abs(math_utils::smallestAngleDifference(tracklet01.phi, tracklet12.phi))};
        if (deltaTanLambda < tanLambdaCut && deltaPhi < phiCut && validTracklets != maxTracklets) {
          usedClusters0[tracklet01.firstClusterIndex] = 1;
          usedClusters2[tracklet12.secondClusterIndex] = 1;
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

template <int NLayers>
void VertexerTraits<NLayers>::updateVertexingParameters(const std::vector<VertexingParameters>& vrtPar)
{
  mVrtParams = vrtPar;
  mIndexTableUtils.setTrackingParameters(vrtPar[0]);
  for (auto& par : mVrtParams) {
    par.phiSpan = static_cast<int>(std::ceil(mIndexTableUtils.getNphiBins() * par.phiCut / o2::constants::math::TwoPI));
    par.zSpan = static_cast<int>(std::ceil(par.zCut * mIndexTableUtils.getInverseZCoordinate(0)));
  }
}

// Main functions
template <int NLayers>
void VertexerTraits<NLayers>::computeTracklets(const int iteration)
{
  mTaskArena->execute([&] {
    tbb::parallel_for(0, mTimeFrame->getNrof(1), [&](const short pivotRofId) {
      const auto& rofRange01 = mTimeFrame->getROFOverlapTableView().getOverlap(1, 0, pivotRofId);
      for (auto targetRofId = rofRange01.getFirstEntry(); targetRofId < rofRange01.getEntriesBound(); ++targetRofId) {
        const auto timeErr = mTimeFrame->getROFOverlapTableView().getTimeStamp(0, targetRofId, 1, pivotRofId);
        trackleterKernelHost<TrackletMode::Layer0Layer1, true>(
          mTimeFrame->getClustersOnLayer(targetRofId, 0),   // Clusters to be matched with the next layer in target rof
          mTimeFrame->getClustersOnLayer(pivotRofId, 1),    // Clusters to be matched with the current layer in pivot rof
          mTimeFrame->getUsedClustersROF(targetRofId, 0),   // Span of the used clusters in the target rof
          mTimeFrame->getIndexTable(targetRofId, 0).data(), // Index table to access the data on the next layer in target rof
          mVrtParams[iteration].phiCut,
          mTimeFrame->getTracklets()[0],                   // Flat tracklet buffer
          mTimeFrame->getNTrackletsCluster(pivotRofId, 0), // Span of the number of tracklets per each cluster in pivot rof
          mIndexTableUtils,
          timeErr,
          gsl::span<int>(), // Offset in the tracklet buffer
          mVrtParams[iteration].maxTrackletsPerCluster);
      }
      const auto& rofRange12 = mTimeFrame->getROFOverlapTableView().getOverlap(1, 2, pivotRofId);
      for (auto targetRofId = rofRange12.getFirstEntry(); targetRofId < rofRange12.getEntriesBound(); ++targetRofId) {
        const auto timeErr = mTimeFrame->getROFOverlapTableView().getTimeStamp(2, targetRofId, 1, pivotRofId);
        trackleterKernelHost<TrackletMode::Layer1Layer2, true>(
          mTimeFrame->getClustersOnLayer(targetRofId, 2),
          mTimeFrame->getClustersOnLayer(pivotRofId, 1),
          mTimeFrame->getUsedClustersROF(targetRofId, 2),
          mTimeFrame->getIndexTable(targetRofId, 2).data(),
          mVrtParams[iteration].phiCut,
          mTimeFrame->getTracklets()[1],
          mTimeFrame->getNTrackletsCluster(pivotRofId, 1), // Span of the number of tracklets per each cluster in pivot rof
          mIndexTableUtils,
          timeErr,
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

    tbb::parallel_for(0, mTimeFrame->getNrof(1), [&](const short pivotRofId) {
      const auto& rofRange01 = mTimeFrame->getROFOverlapTableView().getOverlap(1, 0, pivotRofId);
      for (auto targetRofId = rofRange01.getFirstEntry(); targetRofId < rofRange01.getEntriesBound(); ++targetRofId) {
        const auto timeErr = mTimeFrame->getROFOverlapTableView().getTimeStamp(0, targetRofId, 1, pivotRofId);
        trackleterKernelHost<TrackletMode::Layer0Layer1, false>(
          mTimeFrame->getClustersOnLayer(targetRofId, 0),
          mTimeFrame->getClustersOnLayer(pivotRofId, 1),
          mTimeFrame->getUsedClustersROF(targetRofId, 0),
          mTimeFrame->getIndexTable(targetRofId, 0).data(),
          mVrtParams[iteration].phiCut,
          mTimeFrame->getTracklets()[0],
          mTimeFrame->getNTrackletsCluster(pivotRofId, 0),
          mIndexTableUtils,
          timeErr,
          mTimeFrame->getExclusiveNTrackletsCluster(pivotRofId, 0),
          mVrtParams[iteration].maxTrackletsPerCluster);
      }
      const auto& rofRange12 = mTimeFrame->getROFOverlapTableView().getOverlap(1, 2, pivotRofId);
      for (auto targetRofId = rofRange12.getFirstEntry(); targetRofId < rofRange12.getEntriesBound(); ++targetRofId) {
        const auto timeErr = mTimeFrame->getROFOverlapTableView().getTimeStamp(2, targetRofId, 1, pivotRofId);
        trackleterKernelHost<TrackletMode::Layer1Layer2, false>(
          mTimeFrame->getClustersOnLayer(targetRofId, 2),
          mTimeFrame->getClustersOnLayer(pivotRofId, 1),
          mTimeFrame->getUsedClustersROF(targetRofId, 2),
          mTimeFrame->getIndexTable(targetRofId, 2).data(),
          mVrtParams[iteration].phiCut,
          mTimeFrame->getTracklets()[1],
          mTimeFrame->getNTrackletsCluster(pivotRofId, 1),
          mIndexTableUtils,
          timeErr,
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
        // FIXME: !!!!!!!
        // int sortedId0{mTimeFrame->getSortedIndex(trk.rof[0], 0, trk.firstClusterIndex)};
        // int sortedId1{mTimeFrame->getSortedIndex(trk.rof[1], 1, trk.secondClusterIndex)};
        int sortedId0{0};
        int sortedId1{0};
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
}

template <int NLayers>
void VertexerTraits<NLayers>::computeTrackletMatching(const int iteration)
{
  mTaskArena->execute([&] {
    tbb::combinable<int> totalLines{0};
    tbb::parallel_for(
      tbb::blocked_range<short>(0, (short)mTimeFrame->getNrof(1)),
      [&](const tbb::blocked_range<short>& Rofs) {
        for (short pivotRofId = Rofs.begin(); pivotRofId < Rofs.end(); ++pivotRofId) {
          if (mTimeFrame->getFoundTracklets(pivotRofId, 0).empty()) {
            continue;
          }
          mTimeFrame->getLines(pivotRofId).reserve(mTimeFrame->getNTrackletsCluster(pivotRofId, 0).size());
          bounded_vector<bool> usedTracklets(mTimeFrame->getFoundTracklets(pivotRofId, 0).size(), false, mMemoryPool.get());

          const auto& rofRange01 = mTimeFrame->getROFOverlapTableView().getOverlap(1, 0, pivotRofId);
          const auto& rofRange12 = mTimeFrame->getROFOverlapTableView().getOverlap(1, 2, pivotRofId);
          for (uint32_t targetRofId0 = rofRange01.getFirstEntry(); targetRofId0 < rofRange01.getEntriesBound(); ++targetRofId0) {
            const auto targetRofTime0 = mTimeFrame->getROFOverlapTableView().getLayer(0).getROFTimeBounds(targetRofId0);
            for (uint32_t targetRofId2 = rofRange12.getFirstEntry(); targetRofId2 < rofRange12.getEntriesBound(); ++targetRofId2) {
              const auto targetRofTime2 = mTimeFrame->getROFOverlapTableView().getLayer(2).getROFTimeBounds(targetRofId2);
              if (!(mTimeFrame->getROFOverlapTableView().doROFsOverlap(0, targetRofId0, 2, targetRofId2))) {
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
                targetRofTime0,
                targetRofTime2,
                mVrtParams[iteration].tanLambdaCut,
                mVrtParams[iteration].phiCut);
            }
          }
          totalLines.local() += mTimeFrame->getLines(pivotRofId).size();
        }
      });
    mTimeFrame->setNLinesTotal(totalLines.combine(std::plus<int>()));
  });

  // from here on we do not use tracklets from L1-2 anymore, so let's free them
  deepVectorClear(mTimeFrame->getTracklets()[1]);
}

template <int NLayers>
void VertexerTraits<NLayers>::computeVertices(const int iteration)
{
  auto nsigmaCut{std::min(mVrtParams[iteration].vertNsigmaCut * mVrtParams[iteration].vertNsigmaCut * (mVrtParams[iteration].vertRadiusSigma * mVrtParams[iteration].vertRadiusSigma + mVrtParams[iteration].trackletSigma * mVrtParams[iteration].trackletSigma), 1.98f)};
  bounded_vector<int> noClustersVec(mTimeFrame->getNrof(1), 0, mMemoryPool.get());
  for (int rofId{0}; rofId < mTimeFrame->getNrof(1); ++rofId) {
    const int numTracklets{static_cast<int>(mTimeFrame->getLines(rofId).size())};
    bounded_vector<bool> usedTracklets(numTracklets, false, mMemoryPool.get());
    for (int iLine1{0}; iLine1 < numTracklets; ++iLine1) {
      if (usedTracklets[iLine1]) {
        continue;
      }
      const auto& line1 = mTimeFrame->getLines(rofId)[iLine1];
      for (int iLine2{iLine1 + 1}; iLine2 < numTracklets; ++iLine2) {
        if (usedTracklets[iLine2]) {
          continue;
        }
        const auto& line2 = mTimeFrame->getLines(rofId)[iLine2];
        if (!line1.mTime.isCompatible(line2.mTime)) {
          continue;
        }
        auto dca{Line::getDCA(line1, line2)};
        if (dca < mVrtParams[iteration].pairCut) {
          mTimeFrame->getTrackletClusters(rofId).emplace_back(iLine1, line1, iLine2, line2);
          if (mTimeFrame->getTrackletClusters(rofId).back().getR2() > 4.f) {
            mTimeFrame->getTrackletClusters(rofId).pop_back();
            break;
          }
          usedTracklets[iLine1] = true;
          usedTracklets[iLine2] = true;
          for (int iLine3{0}; iLine3 < numTracklets; ++iLine3) {
            if (usedTracklets[iLine3]) {
              continue;
            }
            const auto& line3 = mTimeFrame->getLines(rofId)[iLine3];
            if (!line3.mTime.isCompatible(mTimeFrame->getTrackletClusters(rofId).back().getTimeStamp())) {
              continue;
            }
            if (Line::getDistanceFromPoint(line3, mTimeFrame->getTrackletClusters(rofId).back().getVertex()) < mVrtParams[iteration].pairCut) {
              mTimeFrame->getTrackletClusters(rofId).back().add(iLine3, line3);
              usedTracklets[iLine3] = true;
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
        if (mTimeFrame->getTrackletClusters(rofId)[iCluster1].getTimeStamp().isCompatible(mTimeFrame->getTrackletClusters(rofId)[iCluster2].getTimeStamp())) {
          vertex2 = mTimeFrame->getTrackletClusters(rofId)[iCluster2].getVertex();
          if (o2::gpu::GPUCommonMath::Abs(vertex1[2] - vertex2[2]) < mVrtParams[iteration].clusterCut) {
            float distance{((vertex1[0] - vertex2[0]) * (vertex1[0] - vertex2[0])) +
                           ((vertex1[1] - vertex2[1]) * (vertex1[1] - vertex2[1])) +
                           ((vertex1[2] - vertex2[2]) * (vertex1[2] - vertex2[2]))};
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
  }
  for (int rofId{0}; rofId < mTimeFrame->getNrof(1); ++rofId) {
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
        Vertex vertex{o2::math_utils::Point3D<float>(mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[0],
                                                     mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[1],
                                                     mTimeFrame->getTrackletClusters(rofId)[iCluster].getVertex()[2]),
                      mTimeFrame->getTrackletClusters(rofId)[iCluster].getRMS2(),          // Symm matrix. Diagonal: RMS2 components,
                                                                                           // off-diagonal: square mean of projections on planes.
                      (ushort)mTimeFrame->getTrackletClusters(rofId)[iCluster].getSize(),  // Contributors
                      mTimeFrame->getTrackletClusters(rofId)[iCluster].getAvgDistance2()}; // In place of chi2

        if (iteration) {
          vertex.setFlags(Vertex::UPCMode);
        }
        vertex.setTimeStamp(mTimeFrame->getTrackletClusters(rofId)[iCluster].getTimeStamp());
        mTimeFrame->addPrimaryVertex(vertex);
        if (mTimeFrame->hasMCinformation()) {
          bounded_vector<o2::MCCompLabel> labels(mMemoryPool.get());
          for (auto& index : mTimeFrame->getTrackletClusters(rofId)[iCluster].getLabels()) {
            labels.push_back(mTimeFrame->getLinesLabel(rofId)[index]); // then we can use nContributors from vertices to get the labels
          }
          mTimeFrame->addPrimaryVertexLabel(computeMain(labels));
        }
      }
    }
  }
}

template <int NLayers>
void VertexerTraits<NLayers>::addTruthSeedingVertices()
{
  LOGP(info, "Using truth seeds as vertices; will skip computations");
  const auto dc = o2::steer::DigitizationContext::loadFromFile("collisioncontext.root");
  const auto irs = dc->getEventRecords();
  int64_t roFrameBiasInBC = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance().getROFBiasInBC(1);
  int64_t roFrameLengthInBC = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance().getROFLengthInBC(1);
  o2::steer::MCKinematicsReader mcReader(dc);
  const int iSrc = 0; // take only events from collision generator
  auto eveId2colId = dc->getCollisionIndicesForSource(iSrc);
  for (int iEve{0}; iEve < mcReader.getNEvents(iSrc); ++iEve) {
    const auto& ir = irs[eveId2colId[iEve]];
    if (!ir.isDummy()) { // do we need this, is this for diffractive events?
      const auto& eve = mcReader.getMCEventHeader(iSrc, iEve);
      auto bc = (ir - raw::HBFUtils::Instance().getFirstSampledTFIR()).toLong() - roFrameBiasInBC;
      if (bc < 0) { // event happened before TF
        continue;
      }
      Vertex vert;
      vert.getTimeStamp().setTimeStamp(bc);
      vert.getTimeStamp().setTimeStampError(roFrameLengthInBC / 2);
      // set minimum to 1 sometimes for diffractive events there is nothing acceptance
      vert.setNContributors(std::max(1L, std::ranges::count_if(mcReader.getTracks(iSrc, iEve), [](const auto& trk) {
                                       return trk.isPrimary() && trk.GetPt() > 0.05 && std::abs(trk.GetEta()) < 1.1;
                                     })));
      vert.setXYZ((float)eve.GetX(), (float)eve.GetY(), (float)eve.GetZ());
      vert.setChi2(1); // not used as constraint
      constexpr float cov = 25e-4;
      vert.setSigmaX(cov);
      vert.setSigmaY(cov);
      vert.setSigmaZ(cov);
      mTimeFrame->addPrimaryVertex(vert);
      o2::MCCompLabel mcLbl(o2::MCCompLabel::maxTrackID(), iEve, iSrc, false);
      VertexLabel lbl(mcLbl, 1.0);
      mTimeFrame->addPrimaryVertexLabel(lbl);
    }
    mcReader.releaseTracksForSourceAndEvent(iSrc, iEve);
  }
  LOGP(info, "Imposed {} pv collisions from mc-truth", mTimeFrame->getPrimaryVertices().size());
}

template <int NLayers>
void VertexerTraits<NLayers>::setNThreads(int n, std::shared_ptr<tbb::task_arena>& arena)
{
  if (arena == nullptr) {
    mTaskArena = std::make_shared<tbb::task_arena>(std::abs(n));
    LOGP(info, "Setting seeding vertexer with {} threads.", n);
  } else {
    mTaskArena = arena;
    LOGP(info, "Attaching vertexer to calling thread's arena");
  }
}

template class VertexerTraits<7>;
} // namespace o2::its
