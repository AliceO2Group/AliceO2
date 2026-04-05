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

#include <algorithm>
#include <memory>
#include <ranges>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/combinable.h>

#include "ITStracking/VertexerTraits.h"
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/Tracklet.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "SimulationDataFormat/O2DatabasePDG.h"
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
  gsl::span<int> rofFoundTrackletsOffsets,
  const int globalOffsetNextLayer = 0,
  const int globalOffsetCurrentLayer = 0,
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
                  tracklets[rofFoundTrackletsOffsets[iCurrentLayerClusterIndex] + storedTracklets] = Tracklet{globalOffsetNextLayer + iNextLayerClusterIndex, globalOffsetCurrentLayer + iCurrentLayerClusterIndex, nextCluster, currentCluster, timErr};
                } else {
                  tracklets[rofFoundTrackletsOffsets[iCurrentLayerClusterIndex] + storedTracklets] = Tracklet{globalOffsetCurrentLayer + iCurrentLayerClusterIndex, globalOffsetNextLayer + iNextLayerClusterIndex, currentCluster, nextCluster, timErr};
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
  const Cluster* clusters0,               // global layer 0 clusters
  const Cluster* clusters1,               // global layer 1 clusters
  gsl::span<unsigned char> usedClusters0, // global layer 0 used clusters
  gsl::span<unsigned char> usedClusters2, // global layer 2 used clusters
  const gsl::span<const Tracklet>& tracklets01,
  const gsl::span<const Tracklet>& tracklets12,
  bounded_vector<bool>& usedTracklets,
  const gsl::span<int> foundTracklets01,
  const gsl::span<int> foundTracklets12,
  bounded_vector<Line>& lines,
  const gsl::span<const o2::MCCompLabel>& trackletLabels,
  bounded_vector<o2::MCCompLabel>& linesLabels,
  const int nLayer1Clusters,
  const float tanLambdaCut = 0.025f,
  const float phiCut = 0.005f,
  const int maxTracklets = 100)
{
  int offset01{0}, offset12{0};
  for (int iCurrentLayerClusterIndex{0}; iCurrentLayerClusterIndex < nLayer1Clusters; ++iCurrentLayerClusterIndex) {
    int validTracklets{0};
    for (int iTracklet12{offset12}; iTracklet12 < offset12 + foundTracklets12[iCurrentLayerClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{offset01}; iTracklet01 < offset01 + foundTracklets01[iCurrentLayerClusterIndex]; ++iTracklet01) {
        if (usedTracklets[iTracklet01]) {
          continue;
        }

        const auto& tracklet01{tracklets01[iTracklet01]};
        const auto& tracklet12{tracklets12[iTracklet12]};
        if (!tracklet01.getTimeStamp().isCompatible(tracklet12.getTimeStamp())) {
          continue;
        }

        const float deltaTanLambda{o2::gpu::GPUCommonMath::Abs(tracklet01.tanLambda - tracklet12.tanLambda)};
        const float deltaPhi{o2::gpu::GPUCommonMath::Abs(math_utils::smallestAngleDifference(tracklet01.phi, tracklet12.phi))};
        if (deltaTanLambda < tanLambdaCut && deltaPhi < phiCut && validTracklets != maxTracklets) {
          usedClusters0[tracklet01.firstClusterIndex] = 1;
          usedClusters2[tracklet12.secondClusterIndex] = 1;
          usedTracklets[iTracklet01] = true;
          lines.emplace_back(tracklet01, clusters0, clusters1);
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
      bool skipROF = !mTimeFrame->getROFMaskView().isROFEnabled(1, pivotRofId);
      const auto& rofRange01 = mTimeFrame->getROFOverlapTableView().getOverlap(1, 0, pivotRofId);
      for (auto targetRofId = rofRange01.getFirstEntry(); targetRofId < rofRange01.getEntriesBound(); ++targetRofId) {
        const auto timeErr = mTimeFrame->getROFOverlapTableView().getTimeStamp(0, targetRofId, 1, pivotRofId);
        trackleterKernelHost<TrackletMode::Layer0Layer1, true>(
          !skipROF ? mTimeFrame->getClustersOnLayer(targetRofId, 0) : gsl::span<Cluster>(), // Clusters to be matched with the next layer in target rof
          !skipROF ? mTimeFrame->getClustersOnLayer(pivotRofId, 1) : gsl::span<Cluster>(),  // Clusters to be matched with the current layer in pivot rof
          mTimeFrame->getUsedClustersROF(targetRofId, 0),                                   // Span of the used clusters in the target rof
          mTimeFrame->getIndexTable(targetRofId, 0).data(),                                 // Index table to access the data on the next layer in target rof
          mVrtParams[iteration].phiCut,
          mTimeFrame->getTracklets()[0],                   // Flat tracklet buffer
          mTimeFrame->getNTrackletsCluster(pivotRofId, 0), // Span of the number of tracklets per each cluster in pivot rof
          mIndexTableUtils,
          timeErr,
          gsl::span<int>(), // Offset in the tracklet buffer
          0,
          0,
          mVrtParams[iteration].maxTrackletsPerCluster);
      }
      const auto& rofRange12 = mTimeFrame->getROFOverlapTableView().getOverlap(1, 2, pivotRofId);
      for (auto targetRofId = rofRange12.getFirstEntry(); targetRofId < rofRange12.getEntriesBound(); ++targetRofId) {
        const auto timeErr = mTimeFrame->getROFOverlapTableView().getTimeStamp(2, targetRofId, 1, pivotRofId);
        trackleterKernelHost<TrackletMode::Layer1Layer2, true>(
          !skipROF ? mTimeFrame->getClustersOnLayer(targetRofId, 2) : gsl::span<Cluster>(),
          !skipROF ? mTimeFrame->getClustersOnLayer(pivotRofId, 1) : gsl::span<Cluster>(),
          mTimeFrame->getUsedClustersROF(targetRofId, 2),
          mTimeFrame->getIndexTable(targetRofId, 2).data(),
          mVrtParams[iteration].phiCut,
          mTimeFrame->getTracklets()[1],
          mTimeFrame->getNTrackletsCluster(pivotRofId, 1), // Span of the number of tracklets per each cluster in pivot rof
          mIndexTableUtils,
          timeErr,
          gsl::span<int>(), // Offset in the tracklet buffer
          0,
          0,
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
      bool skipROF = !mTimeFrame->getROFMaskView().isROFEnabled(1, pivotRofId);
      const int globalOffsetPivot = mTimeFrame->getSortedStartIndex(pivotRofId, 1);
      const auto& rofRange01 = mTimeFrame->getROFOverlapTableView().getOverlap(1, 0, pivotRofId);
      for (auto targetRofId = rofRange01.getFirstEntry(); targetRofId < rofRange01.getEntriesBound(); ++targetRofId) {
        const auto timeErr = mTimeFrame->getROFOverlapTableView().getTimeStamp(0, targetRofId, 1, pivotRofId);
        trackleterKernelHost<TrackletMode::Layer0Layer1, false>(
          !skipROF ? mTimeFrame->getClustersOnLayer(targetRofId, 0) : gsl::span<Cluster>(),
          !skipROF ? mTimeFrame->getClustersOnLayer(pivotRofId, 1) : gsl::span<Cluster>(),
          mTimeFrame->getUsedClustersROF(targetRofId, 0),
          mTimeFrame->getIndexTable(targetRofId, 0).data(),
          mVrtParams[iteration].phiCut,
          mTimeFrame->getTracklets()[0],
          mTimeFrame->getNTrackletsCluster(pivotRofId, 0),
          mIndexTableUtils,
          timeErr,
          mTimeFrame->getExclusiveNTrackletsCluster(pivotRofId, 0),
          mTimeFrame->getSortedStartIndex(targetRofId, 0),
          globalOffsetPivot,
          mVrtParams[iteration].maxTrackletsPerCluster);
      }
      const auto& rofRange12 = mTimeFrame->getROFOverlapTableView().getOverlap(1, 2, pivotRofId);
      for (auto targetRofId = rofRange12.getFirstEntry(); targetRofId < rofRange12.getEntriesBound(); ++targetRofId) {
        const auto timeErr = mTimeFrame->getROFOverlapTableView().getTimeStamp(2, targetRofId, 1, pivotRofId);
        trackleterKernelHost<TrackletMode::Layer1Layer2, false>(
          !skipROF ? mTimeFrame->getClustersOnLayer(targetRofId, 2) : gsl::span<Cluster>(),
          !skipROF ? mTimeFrame->getClustersOnLayer(pivotRofId, 1) : gsl::span<Cluster>(),
          mTimeFrame->getUsedClustersROF(targetRofId, 2),
          mTimeFrame->getIndexTable(targetRofId, 2).data(),
          mVrtParams[iteration].phiCut,
          mTimeFrame->getTracklets()[1],
          mTimeFrame->getNTrackletsCluster(pivotRofId, 1),
          mIndexTableUtils,
          timeErr,
          mTimeFrame->getExclusiveNTrackletsCluster(pivotRofId, 1),
          mTimeFrame->getSortedStartIndex(targetRofId, 2),
          globalOffsetPivot,
          mVrtParams[iteration].maxTrackletsPerCluster);
      }
    });
  });

  /// Create tracklets labels for L0-L1, information is as flat as in tracklets vector (no rofId)
  if (mTimeFrame->hasMCinformation()) {
    for (const auto& trk : mTimeFrame->getTracklets()[0]) {
      o2::MCCompLabel label;
      if (!trk.isEmpty()) {
        int sortedId0{trk.firstClusterIndex};
        int sortedId1{trk.secondClusterIndex};
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
          trackletSelectionKernelHost(
            mTimeFrame->getClusters()[0].data(),
            mTimeFrame->getClusters()[1].data(),
            mTimeFrame->getUsedClusters(0),
            mTimeFrame->getUsedClusters(2),
            mTimeFrame->getFoundTracklets(pivotRofId, 0),
            mTimeFrame->getFoundTracklets(pivotRofId, 1),
            usedTracklets,
            mTimeFrame->getNTrackletsCluster(pivotRofId, 0),
            mTimeFrame->getNTrackletsCluster(pivotRofId, 1),
            mTimeFrame->getLines(pivotRofId),
            mTimeFrame->getLabelsFoundTracklets(pivotRofId, 0),
            mTimeFrame->getLinesLabel(pivotRofId),
            static_cast<int>(mTimeFrame->getClustersOnLayer(pivotRofId, 1).size()),
            mVrtParams[iteration].tanLambdaCut,
            mVrtParams[iteration].phiCut);
          auto& lines = mTimeFrame->getLines(pivotRofId);
          totalLines.local() += lines.size();
          std::stable_sort(lines.begin(), lines.end(), [](const Line& a, const Line& b) {
            // sort by lower edge and secondly prefer wider windows
            if (a.mTime.lower() != b.mTime.lower()) {
              return a.mTime.lower() < b.mTime.lower();
            }
            return a.mTime.upper() > b.mTime.upper();
          });
        }
      });
    mTimeFrame->setNLinesTotal(totalLines.combine(std::plus<int>()));
  });

  // from here on we do not use tracklets anymore, so let's free them
  deepVectorClear(mTimeFrame->getTracklets());
}

template <int NLayers>
void VertexerTraits<NLayers>::computeVertices(const int iteration)
{
  const auto nsigmaCut{std::min(mVrtParams[iteration].vertNsigmaCut * mVrtParams[iteration].vertNsigmaCut * (mVrtParams[iteration].vertRadiusSigma * mVrtParams[iteration].vertRadiusSigma + mVrtParams[iteration].trackletSigma * mVrtParams[iteration].trackletSigma), 1.98f)};
  const auto pairCut2{mVrtParams[iteration].pairCut * mVrtParams[iteration].pairCut};
  const int nRofs = mTimeFrame->getNrof(1);
  const bool hasMC = mTimeFrame->hasMCinformation();
  std::vector<std::vector<Vertex>> rofVertices(nRofs);
  std::vector<std::vector<VertexLabel>> rofLabels(nRofs);

  const auto processROF = [&](const int rofId) {
    auto& lines = mTimeFrame->getLines(rofId);
    const int nLines{static_cast<int>(lines.size())};
    bounded_vector<uint8_t> usedTracklets(nLines, 0, mMemoryPool.get());
    auto& clusters = mTimeFrame->getTrackletClusters(rofId);

    for (int iLine1{0}; iLine1 < nLines; ++iLine1) {
      if (usedTracklets[iLine1]) {
        continue;
      }
      const auto& line1 = lines[iLine1];
      for (int iLine2{iLine1 + 1}; iLine2 < nLines; ++iLine2) {
        if (usedTracklets[iLine2]) {
          continue;
        }
        const auto& line2 = lines[iLine2];
        if (!line1.mTime.isCompatible(line2.mTime)) {
          continue;
        }
        auto dca2{Line::getDCA2(line1, line2)};
        if (dca2 < pairCut2) {
          auto& cluster = clusters.emplace_back(iLine1, line1, iLine2, line2);
          if (!cluster.isValid() || cluster.getR2() > 4.f) {
            clusters.pop_back();
            continue;
          }

          usedTracklets[iLine1] = 1;
          usedTracklets[iLine2] = 1;
          for (int iLine3{0}; iLine3 < nLines; ++iLine3) {
            if (usedTracklets[iLine3]) {
              continue;
            }
            const auto& line3 = lines[iLine3];
            if (!line3.mTime.isCompatible(cluster.getTimeStamp())) {
              continue;
            }
            const auto distance2 = Line::getDistance2FromPoint(line3, cluster.getVertex());
            if (distance2 < pairCut2) {
              cluster.add(iLine3, line3);
              usedTracklets[iLine3] = 1;
            }
          }
          break;
        }
      }
    }

    // Cluster merging
    std::sort(clusters.begin(), clusters.end(),
              [](ClusterLines& cluster1, ClusterLines& cluster2) { return cluster1.getSize() > cluster2.getSize(); });
    int nClusters = static_cast<int>(clusters.size());
    for (int iCluster1{0}; iCluster1 < nClusters; ++iCluster1) {
      std::array<float, 3> vertex1{clusters[iCluster1].getVertex()};
      std::array<float, 3> vertex2{};
      for (int iCluster2{iCluster1 + 1}; iCluster2 < nClusters; ++iCluster2) {
        if (clusters[iCluster1].getTimeStamp().isCompatible(clusters[iCluster2].getTimeStamp())) {
          vertex2 = clusters[iCluster2].getVertex();
          if (o2::gpu::GPUCommonMath::Abs(vertex1[2] - vertex2[2]) < mVrtParams[iteration].clusterCut) {
            float distance{((vertex1[0] - vertex2[0]) * (vertex1[0] - vertex2[0])) +
                           ((vertex1[1] - vertex2[1]) * (vertex1[1] - vertex2[1])) +
                           ((vertex1[2] - vertex2[2]) * (vertex1[2] - vertex2[2]))};
            if (distance < mVrtParams[iteration].pairCut * mVrtParams[iteration].pairCut) {
              for (auto label : clusters[iCluster2].getLabels()) {
                clusters[iCluster1].add(label, lines[label]);
                vertex1 = clusters[iCluster1].getVertex();
              }
              clusters.erase(clusters.begin() + iCluster2);
              --iCluster2;
              --nClusters;
            }
          }
        }
      }
    }

    // Vertex filtering
    std::sort(clusters.begin(), clusters.end(),
              [](const ClusterLines& cluster1, const ClusterLines& cluster2) { return cluster1.getSize() > cluster2.getSize(); });
    bool atLeastOneFound{false};
    for (int iCluster{0}; iCluster < nClusters; ++iCluster) {
      bool lowMultCandidate{false};
      double beamDistance2{(mTimeFrame->getBeamX() - clusters[iCluster].getVertex()[0]) * (mTimeFrame->getBeamX() - clusters[iCluster].getVertex()[0]) +
                           (mTimeFrame->getBeamY() - clusters[iCluster].getVertex()[1]) * (mTimeFrame->getBeamY() - clusters[iCluster].getVertex()[1])};
      if (atLeastOneFound && (lowMultCandidate = clusters[iCluster].getSize() < mVrtParams[iteration].clusterContributorsCut)) {
        lowMultCandidate &= (beamDistance2 < mVrtParams[iteration].lowMultBeamDistCut * mVrtParams[iteration].lowMultBeamDistCut);
        if (!lowMultCandidate) {
          clusters.erase(clusters.begin() + iCluster);
          nClusters--;
          continue;
        }
      }

      if (beamDistance2 < nsigmaCut && o2::gpu::GPUCommonMath::Abs(clusters[iCluster].getVertex()[2]) < mVrtParams[iteration].maxZPositionAllowed) {
        atLeastOneFound = true;
        Vertex vertex{clusters[iCluster].getVertex().data(),
                      clusters[iCluster].getRMS2(),
                      (ushort)clusters[iCluster].getSize(),
                      clusters[iCluster].getAvgDistance2()};

        if (iteration) {
          vertex.setFlags(Vertex::UPCMode);
        }
        vertex.setTimeStamp(clusters[iCluster].getTimeStamp());
        rofVertices[rofId].push_back(vertex);
        if (hasMC) {
          bounded_vector<o2::MCCompLabel> labels(mMemoryPool.get());
          for (auto& index : clusters[iCluster].getLabels()) {
            labels.push_back(mTimeFrame->getLinesLabel(rofId)[index]);
          }
          rofLabels[rofId].push_back(computeMain(labels));
        }
      }
    }
  };

  if (mTaskArena->max_concurrency() <= 1) {
    for (int rofId{0}; rofId < nRofs; ++rofId) {
      processROF(rofId);
    }
  } else {
    mTaskArena->execute([&] {
      tbb::parallel_for(0, nRofs, [&](const int rofId) {
        processROF(rofId);
      });
    });
  }
  // add vertices, these anyways get sorted afterward
  for (int rofId{0}; rofId < nRofs; ++rofId) {
    for (auto& vertex : rofVertices[rofId]) {
      mTimeFrame->addPrimaryVertex(vertex);
    }
    if (hasMC) {
      for (auto& label : rofLabels[rofId]) {
        mTimeFrame->addPrimaryVertexLabel(label);
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
                                       if (!trk.isPrimary() || trk.GetPt() < 0.05 || std::abs(trk.GetEta()) > 1.1) {
                                         return false;
                                       }
                                       return o2::O2DatabasePDG::Instance()->GetParticle(trk.GetPdgCode())->Charge() != 0;
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
  }
}

template class VertexerTraits<7>;
} // namespace o2::its
