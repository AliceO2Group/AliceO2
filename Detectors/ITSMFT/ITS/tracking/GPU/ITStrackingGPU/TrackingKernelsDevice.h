// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
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
/// \file TrackingKernelsDevice.h
/// \brief Bodies of the ITS tracking kernels, shared by the CUDA/HIP kernels and other GPU backends.
///

#ifndef ITSTRACKINGGPU_TRACKINGKERNELSDEVICE_H_
#define ITSTRACKINGGPU_TRACKINGKERNELSDEVICE_H_

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "ITSMFTTracking/Constants.h"
#include "ITSMFTTracking/MathUtils.h"
#include "ITSMFTTracking/ROFLookupTables.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/TrackingTopology.h"
#include "ITStrackingGPU/DeviceSpan.h"

namespace o2::its::gpu
{

template <int NLayers>
GPUdi() void computeLayerTrackletsMultiROF(
  const int iThread,
  const int nThreads,
  const IndexTableUtils<NLayers>* utils,
  const typename ROFMaskTable<NLayers>::View rofMask,
  const int linkId,
  const typename TrackingTopology<NLayers>::View topology,
  const typename ROFOverlapTable<NLayers>::View rofOverlaps,
  const typename ROFVertexLookupTable<NLayers>::View vertexLUT,
  const Vertex* vertices,
  const bool useDiamond,
  const int vertexId,
  const Cluster** clusters,
  const int** ROFClusters,
  const unsigned char** usedClusters,
  const int** indexTables,
  Tracklet** tracklets,
  int* outputCounter,
  const int outputCapacity,
  const bool selectUPCVertices,
  const float NSigmaCut,
  const float phiCut,
  const float resolutionPV,
  const float* minRs,
  const float* maxRs,
  const float positionResolution,
  const float meanDeltaR,
  const float MSAngle)
{
  const auto link = topology.getLink(linkId);
  const int fromLayer = link.fromLayer;
  const int toLayer = link.toLayer;
  const float minR = minRs[toLayer];
  const float maxR = maxRs[toLayer];
  const int phiBins{utils->getNphiBins()};
  const int zBins{utils->getNzBins()};
  const int tableSize{phiBins * zBins + 1};
  const int totalROFs0 = rofOverlaps.getLayer(fromLayer).mNROFsTF;
  const int totalROFs1 = rofOverlaps.getLayer(toLayer).mNROFsTF;
  if (totalROFs0 <= 0) {
    return;
  }

  const int* const rofOffsets = ROFClusters[fromLayer];
  const int totalClusters = rofOffsets[totalROFs0];
  for (int currentSortedIndex = iThread;
       currentSortedIndex < totalClusters;
       currentSortedIndex += nThreads) {
    // last ROF whose first cluster is at or before this one
    int lo{0}, hi{totalROFs0 - 1};
    while (lo < hi) {
      const int mid{(lo + hi + 1) >> 1};
      if (rofOffsets[mid] <= currentSortedIndex) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    const unsigned int pivotROF = static_cast<unsigned int>(lo);
    if (!rofMask.isROFEnabled(fromLayer, pivotROF)) {
      continue;
    }

    gpuSpan<const Vertex> primaryVertices;
    if (useDiamond) {
      primaryVertices = gpuSpan<const Vertex>(vertices, 1);
    } else {
      const auto& pvs = vertexLUT.getVertices(fromLayer, pivotROF);
      primaryVertices = gpuSpan<const Vertex>(&vertices[pvs.getFirstEntry()], pvs.getEntries());
    }
    if (primaryVertices.empty()) {
      continue;
    }
    const auto startVtx{vertexId >= 0 ? vertexId : 0};
    const auto endVtx{vertexId >= 0 ? o2::gpu::CAMath::Min(vertexId + 1, static_cast<int>(primaryVertices.size())) : static_cast<int>(primaryVertices.size())};
    if (endVtx <= startVtx || (vertexId + 1) > primaryVertices.size()) {
      continue;
    }

    const auto& rofOverlap = rofOverlaps.getOverlap(fromLayer, toLayer, pivotROF);
    if (!rofOverlap.getEntries()) {
      continue;
    }

    {
      const auto& currentCluster{clusters[fromLayer][currentSortedIndex]};
      if (usedClusters[fromLayer][currentCluster.clusterId]) {
        continue;
      }

      const float inverseR0{1.f / currentCluster.radius};
      for (int iV{startVtx}; iV < endVtx; ++iV) {
        auto& primaryVertex{primaryVertices[iV]};
        if (!vertexLUT.isVertexCompatible(fromLayer, pivotROF, primaryVertex)) {
          continue;
        }
        if (primaryVertex.isFlagSet(Vertex::Flags::UPCMode) != selectUPCVertices) {
          continue;
        }

        const float resolution = o2::gpu::CAMath::Sqrt(math_utils::Sq(resolutionPV) / primaryVertex.getNContributors() + math_utils::Sq(positionResolution));
        const float tanLambda{(currentCluster.zCoordinate - primaryVertex.getZ()) * inverseR0};
        const float zAtRmin{tanLambda * (minR - currentCluster.radius) + currentCluster.zCoordinate};
        const float zAtRmax{tanLambda * (maxR - currentCluster.radius) + currentCluster.zCoordinate};
        const float sqInverseDeltaZ0{1.f / (math_utils::Sq(currentCluster.zCoordinate - primaryVertex.getZ()) + constants::Tolerance)}; /// protecting from overflows adding the detector resolution
        const float sigmaZ{o2::gpu::CAMath::Sqrt(math_utils::Sq(resolution) * math_utils::Sq(tanLambda) * ((math_utils::Sq(inverseR0) + sqInverseDeltaZ0) * math_utils::Sq(meanDeltaR) + 1.f) + math_utils::Sq(meanDeltaR * MSAngle))};
        const int4 selectedBinsRect{o2::its::getBinsRect(currentCluster, toLayer, zAtRmin, zAtRmax, sigmaZ * NSigmaCut, phiCut, *utils)};
        if (selectedBinsRect.x < 0) {
          continue;
        }
        int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};

        if (phiBinsNum < 0) {
          phiBinsNum += phiBins;
        }

        for (short targetROF = rofOverlap.getFirstEntry(); targetROF < rofOverlap.getEntriesBound(); ++targetROF) {
          if (!rofMask.isROFEnabled(toLayer, targetROF)) {
            continue;
          }
          auto clustersNextLayer = getClustersOnLayer(targetROF, totalROFs1, toLayer, ROFClusters, clusters);
          if (clustersNextLayer.empty()) {
            continue;
          }
          const auto ts = rofOverlaps.getTimeStamp(fromLayer, pivotROF, toLayer, targetROF);
          if (!ts.isCompatible(primaryVertex.getTimeStamp())) {
            continue;
          }
          for (int iPhiCount{0}; iPhiCount < phiBinsNum; iPhiCount++) {
            int iPhiBin = (selectedBinsRect.y + iPhiCount) % phiBins;
            const int firstBinIndex{utils->getBinIndex(selectedBinsRect.x, iPhiBin)};
            const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
            const int firstRowClusterIndex = indexTables[toLayer][(targetROF)*tableSize + firstBinIndex];
            const int maxRowClusterIndex = indexTables[toLayer][(targetROF)*tableSize + maxBinIndex];
            for (int nextClusterIndex{firstRowClusterIndex}; nextClusterIndex < maxRowClusterIndex; ++nextClusterIndex) {
              if (nextClusterIndex >= clustersNextLayer.size()) {
                break;
              }
              const Cluster& nextCluster{clustersNextLayer[nextClusterIndex]};
              if (usedClusters[toLayer][nextCluster.clusterId]) {
                continue;
              }
              const float deltaPhi{o2::gpu::CAMath::Abs(currentCluster.phi - nextCluster.phi)};
              const float deltaZ{o2::gpu::CAMath::Abs(tanLambda * (nextCluster.radius - currentCluster.radius) + currentCluster.zCoordinate - nextCluster.zCoordinate)};
              if (deltaZ / sigmaZ < NSigmaCut && (deltaPhi < phiCut || o2::gpu::CAMath::Abs(deltaPhi - o2::constants::math::TwoPI) < phiCut)) {
                const float phi{o2::math_utils::fastATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
                const float tanL{(currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius)};
                const int nextSortedIndex{ROFClusters[toLayer][targetROF] + nextClusterIndex};
                const int outputIndex = o2::gpu::CAMath::AtomicAdd(outputCounter, 1); // the optimizer turns this into a wave ballot vote
                if (outputIndex < outputCapacity) {
                  tracklets[linkId][outputIndex] = Tracklet{currentSortedIndex, nextSortedIndex, tanL, phi, ts};
                }
              }
            }
          }
        }
      }
    }
  }
}

} // namespace o2::its::gpu

#endif // ITSTRACKINGGPU_TRACKINGKERNELSDEVICE_H_
