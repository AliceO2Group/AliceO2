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

#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/ROFLookupTables.h"
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
void countTrackletsInROFsHandler(const IndexTableUtils<NLayers>* utils,
                                 const typename ROFMaskTable<NLayers>::View& rofMask,
                                 const int transitionId,
                                 const int fromLayer,
                                 const int toLayer,
                                 const typename ROFOverlapTable<NLayers>::View& rofOverlaps,
                                 const typename ROFVertexLookupTable<NLayers>::View& vertexLUT,
                                 const int vertexId,
                                 const Vertex* vertices,
                                 const int* rofPV,
                                 const Cluster** clusters,
                                 std::vector<unsigned int> nClusters,
                                 const int** ROFClusters,
                                 const unsigned char** usedClusters,
                                 const int** clustersIndexTables,
                                 int** trackletsLUTs,
                                 gsl::span<int*> trackletsLUTsHost,
                                 const bool selectUPCVertices,
                                 const float NSigmaCut,
                                 const typename TrackingTopology<NLayers>::View topology,
                                 bounded_vector<float>& transitionPhiCuts,
                                 const float resolutionPV,
                                 std::array<float, NLayers>& minR,
                                 std::array<float, NLayers>& maxR,
                                 bounded_vector<float>& resolutions,
                                 std::vector<float>& radii,
                                 bounded_vector<float>& transitionMSAngles,
                                 o2::its::ExternalAllocator* alloc,
                                 gpu::Streams& streams);

template <int NLayers>
void computeTrackletsInROFsHandler(const IndexTableUtils<NLayers>* utils,
                                   const typename ROFMaskTable<NLayers>::View& rofMask,
                                   const int transitionId,
                                   const int fromLayer,
                                   const int toLayer,
                                   const typename ROFOverlapTable<NLayers>::View& rofOverlaps,
                                   const typename ROFVertexLookupTable<NLayers>::View& vertexLUT,
                                   const int vertexId,
                                   const Vertex* vertices,
                                   const int* rofPV,
                                   const Cluster** clusters,
                                   std::vector<unsigned int> nClusters,
                                   const int** ROFClusters,
                                   const unsigned char** usedClusters,
                                   const int** clustersIndexTables,
                                   Tracklet** tracklets,
                                   gsl::span<Tracklet*> spanTracklets,
                                   gsl::span<int> nTracklets,
                                   int** trackletsLUTs,
                                   gsl::span<int*> trackletsLUTsHost,
                                   const bool selectUPCVertices,
                                   const float NSigmaCut,
                                   const typename TrackingTopology<NLayers>::View topology,
                                   bounded_vector<float>& transitionPhiCuts,
                                   const float resolutionPV,
                                   std::array<float, NLayers>& minR,
                                   std::array<float, NLayers>& maxR,
                                   bounded_vector<float>& resolutions,
                                   std::vector<float>& radii,
                                   bounded_vector<float>& transitionMSAngles,
                                   o2::its::ExternalAllocator* alloc,
                                   gpu::Streams& streams);

template <int NLayers>
void countCellsHandler(const Cluster** sortedClusters,
                       const Cluster** unsortedClusters,
                       const TrackingFrameInfo** tfInfo,
                       Tracklet** tracklets,
                       int** trackletsLUT,
                       const int nTracklets,
                       const int cellTopologyId,
                       const typename TrackingTopology<NLayers>::View topology,
                       CellSeed* cells,
                       int** cellsLUTsDeviceArray,
                       int* cellsLUTsHost,
                       const float bz,
                       const float maxChi2ClusterAttachment,
                       const float cellDeltaTanLambdaSigma,
                       const float nSigmaCut,
                       const std::vector<float>& layerxX0Host,
                       o2::its::ExternalAllocator* alloc,
                       gpu::Streams& streams);

template <int NLayers>
void computeCellsHandler(const Cluster** sortedClusters,
                         const Cluster** unsortedClusters,
                         const TrackingFrameInfo** tfInfo,
                         Tracklet** tracklets,
                         int** trackletsLUT,
                         const int nTracklets,
                         const int cellTopologyId,
                         const typename TrackingTopology<NLayers>::View topology,
                         CellSeed* cells,
                         int** cellsLUTsDeviceArray,
                         int* cellsLUTsHost,
                         const float bz,
                         const float maxChi2ClusterAttachment,
                         const float cellDeltaTanLambdaSigma,
                         const float nSigmaCut,
                         const std::vector<float>& layerxX0Host,
                         gpu::Streams& streams);

template <int NLayers>
void countCellNeighboursHandler(CellSeed** cellsLayersDevice,
                                int* neighboursCursor,
                                int** cellsLUTs,
                                const int sourceCellTopologyId,
                                const int targetCellTopologyId,
                                const float maxChi2ClusterAttachment,
                                const float bz,
                                const unsigned int nCells,
                                gpu::Stream& stream);

void scanCellNeighboursHandler(int* neighboursCursor,
                               int* neighboursLUT,
                               const unsigned int nCells,
                               o2::its::ExternalAllocator* alloc,
                               gpu::Stream& stream);

template <int NLayers>
void computeCellNeighboursHandler(CellSeed** cellsLayersDevice,
                                  int* neighboursCursor,
                                  int** cellsLUTs,
                                  CellNeighbour* cellNeighbours,
                                  const int sourceCellTopologyId,
                                  const int targetCellTopologyId,
                                  const float maxChi2ClusterAttachment,
                                  const float bz,
                                  const unsigned int nCells,
                                  gpu::Stream& stream);

int filterCellNeighboursHandler(gpuPair<int, int>*,
                                int*,
                                unsigned int,
                                gpu::Stream&,
                                o2::its::ExternalAllocator* = nullptr);

template <int NLayers>
void processNeighboursHandler(const int startLevel,
                              const int defaultCellTopologyId,
                              CellSeed** allCellSeeds,
                              CellSeed* currentCellSeeds,
                              const int* currentCellTopologyIds,
                              const int* currentCellIds,
                              const int* nCells,
                              const unsigned char** usedClusters,
                              CellNeighbour** neighbours,
                              int** neighboursDeviceLUTs,
                              const TrackingFrameInfo** foundTrackingFrameInfo,
                              bounded_vector<TrackSeed<NLayers>>& seedsHost,
                              const float bz,
                              const float MaxChi2ClusterAttachment,
                              const float maxChi2NDF,
                              const int maxHoles,
                              const int minTrackLength,
                              const LayerMask holeLayerMask,
                              const std::vector<float>& layerxX0Host,
                              const o2::base::Propagator* propagator,
                              const o2::base::PropagatorF::MatCorrType matCorrType,
                              o2::its::ExternalAllocator* alloc);

template <int NLayers>
void countTrackSeedHandler(TrackSeed<NLayers>* trackSeeds,
                           const TrackingFrameInfo** foundTrackingFrameInfo,
                           const Cluster** unsortedClusters,
                           int* seedLUT,
                           const std::vector<float>& layerRadiiHost,
                           const std::vector<float>& minPtsHost,
                           const std::vector<float>& layerxX0Host,
                           const unsigned int nSeeds,
                           const float Bz,
                           const float maxChi2ClusterAttachment,
                           const float maxChi2NDF,
                           const int reseedIfShorter,
                           const bool repeatRefitOut,
                           const bool shiftRefToCluster,
                           const o2::base::Propagator* propagator,
                           const o2::base::PropagatorF::MatCorrType matCorrType,
                           o2::its::ExternalAllocator* alloc);

template <int NLayers>
void computeTrackSeedHandler(TrackSeed<NLayers>* trackSeeds,
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
                             const int* seedLUT,
                             TrackExtensionHypothesis<NLayers>* activeHypotheses,
                             TrackExtensionHypothesis<NLayers>* nextHypotheses,
                             const std::vector<float>& layerRadiiHost,
                             const std::vector<float>& minPtsHost,
                             const std::vector<float>& layerxX0Host,
                             const unsigned int nSeeds,
                             const unsigned int nTracks,
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

} // namespace o2::its
#endif // ITSTRACKINGGPU_TRACKINGKERNELS_H_
