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

#ifndef ITSTRACKINGGPU_TRACKINGKERNELS_H_
#define ITSTRACKINGGPU_TRACKINGKERNELS_H_

#include <array>
#include <gsl/gsl>

#include "ITSMFTTracking/BoundedAllocator.h"
#include "ITSMFTTracking/CapacityEstimator.h"
#include "ITSMFTTracking/ROFLookupTables.h"
#include "ITStracking/TrackingTopology.h"
#include "ITStracking/TrackExtensionHypothesis.h"
#include "ITStrackingGPU/Utils.h"
#include "DetectorsBase/Propagator.h"

namespace o2::its
{
class CellSeed;
struct CellNeighbour;
template <int>
class TrackSeed;
class TrackingFrameInfo;
class Tracklet;
template <int>
class IndexTableUtils;
class Cluster;
class TrackITSExt;
class ExternalAllocator;

template <int NLayers>
struct TrackingKernels {
  static int computeTrackletsInROFsHandler(const IndexTableUtils<NLayers>* utils,
                                           const typename ROFMaskTable<NLayers>::View& rofMask,
                                           const int linkId,
                                           const int fromLayer,
                                           const int toLayer,
                                           const typename ROFOverlapTable<NLayers>::View& rofOverlaps,
                                           const typename ROFVertexLookupTable<NLayers>::View& vertexLUT,
                                           const int vertexId,
                                           const Vertex* vertices,
                                           const Cluster** clusters,
                                           const std::vector<unsigned int>& nClusters,
                                           const int** ROFClusters,
                                           const unsigned char** usedClusters,
                                           const int** clustersIndexTables,
                                           Tracklet** tracklets,
                                           gsl::span<Tracklet*> spanTracklets,
                                           gsl::span<int> nTracklets,
                                           const int capacity,
                                           gsl::span<int*> trackletsLUTsHost,
                                           const bool selectUPCVertices,
                                           const float NSigmaCut,
                                           const typename TrackingTopology<NLayers>::View topology,
                                           o2::itsmft::tracking::bounded_vector<float>& linkPhiCuts,
                                           const float resolutionPV,
                                           std::array<float, NLayers>& minR,
                                           std::array<float, NLayers>& maxR,
                                           o2::itsmft::tracking::bounded_vector<float>& resolutions,
                                           std::vector<float>& radii,
                                           o2::itsmft::tracking::bounded_vector<float>& linkMSAngles,
                                           o2::its::ExternalAllocator* alloc,
                                           gpu::Streams& streams);

  static int computeCellsHandler(const Cluster** sortedClusters,
                                 const Cluster** unsortedClusters,
                                 const TrackingFrameInfo** tfInfo,
                                 Tracklet** tracklets,
                                 int** trackletsLUT,
                                 const int nTracklets,
                                 const int cellTopologyId,
                                 const typename TrackingTopology<NLayers>::View topology,
                                 CellSeed* cells,
                                 const int capacity,
                                 int* cellsLUTsHost,
                                 const float bz,
                                 const float maxChi2ClusterAttachment,
                                 const float cellDeltaTanLambdaSigma,
                                 const float nSigmaCut,
                                 const float* layerxX0,
                                 o2::its::ExternalAllocator* alloc,
                                 gpu::Streams& streams);

  static void computeCellNeighboursHandler(CellSeed** cellsLayersDevice,
                                           int** cellsLUTs,
                                           CellNeighbour* cellNeighbours,
                                           int* outputCounter,
                                           const int capacity,
                                           const int sourceCellTopologyId,
                                           const int targetCellTopologyId,
                                           const float maxChi2ClusterAttachment,
                                           const float bz,
                                           const unsigned int nCells,
                                           o2::its::ExternalAllocator* alloc,
                                           gpu::Stream& stream);

  static void processNeighboursHandler(const int startLevel,
                                       const int startCellTopologyId,
                                       CellSeed** allCellSeeds,
                                       CellSeed* currentCellSeeds,
                                       const int* currentCellTopologyIds,
                                       const int* currentCellIds,
                                       const int* nCells,
                                       const unsigned char** usedClusters,
                                       CellNeighbour** neighbours,
                                       int** neighboursDeviceLUTs,
                                       const TrackingFrameInfo** foundTrackingFrameInfo,
                                       TrackSeed<NLayers>* seedsDevice,
                                       const int seedsCapacity,
                                       int& seedsCursor,
                                       o2::itsmft::tracking::CapacityEstimator& estimator,
                                       const int iteration,
                                       const float bz,
                                       const float MaxChi2ClusterAttachment,
                                       const float maxChi2NDF,
                                       const int maxHoles,
                                       const int minSeedingClusters,
                                       const LayerMask holeLayerMask,
                                       const LayerMask nonSeedingLayerMask,
                                       const float* layerxX0,
                                       const o2::base::Propagator* propagator,
                                       const o2::base::PropagatorF::MatCorrType matCorrType,
                                       o2::its::ExternalAllocator* alloc);

  static int computeTrackSeedHandler(TrackSeed<NLayers>* trackSeeds,
                                     const TrackingFrameInfo** foundTrackingFrameInfo,
                                     const Cluster** unsortedClusters,
                                     const IndexTableUtils<NLayers>* utils,
                                     const typename ROFMaskTable<NLayers>::View& rofMask,
                                     const typename ROFOverlapTable<NLayers>::View& rofOverlaps,
                                     const Cluster** clusters,
                                     const unsigned char** usedClusters,
                                     const int** clustersIndexTables,
                                     const int** ROFClusters,
                                     o2::its::TrackITSExt* tracks,
                                     int* trackIndices,
                                     int* trackSeedIndices,
                                     int* outputCounter,
                                     const int trackCapacity,
                                     TrackExtensionHypothesis<NLayers>* activeHypotheses,
                                     TrackExtensionHypothesis<NLayers>* nextHypotheses,
                                     const float* layerRadii,
                                     const float* minPts,
                                     const float* layerxX0,
                                     const unsigned int nSeeds,
                                     const float Bz,
                                     const float maxChi2ClusterAttachment,
                                     const float maxChi2NDF,
                                     const int reseedIfShorter,
                                     const bool repeatRefitOut,
                                     const bool shiftRefToCluster,
                                     const int nLayers,
                                     const int phiBins,
                                     const int maxHypotheses,
                                     const bool extendTop,
                                     const bool extendBot,
                                     const float nSigmaCutPhi,
                                     const float nSigmaCutZ,
                                     const o2::base::Propagator* propagator,
                                     const o2::base::PropagatorF::MatCorrType matCorrType,
                                     o2::its::ExternalAllocator* alloc);
};

void resetOutputCounterHandler(int* outputCounter, gpu::Stream& stream);

int finalizeCellNeighboursHandler(CellNeighbour* cellNeighbours,
                                  int* neighboursLUT,
                                  const int nTargetCells,
                                  const int capacity,
                                  o2::its::ExternalAllocator* alloc,
                                  gpu::Stream& stream);

} // namespace o2::its
#endif // ITSTRACKINGGPU_TRACKINGKERNELS_H_
