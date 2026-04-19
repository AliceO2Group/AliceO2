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

#include <gsl/gsl>

#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/ROFLookupTables.h"
#include "ITStracking/Definitions.h"
#include "ITStrackingGPU/Utils.h"
#include "DetectorsBase/Propagator.h"
#include "GPUCommonDef.h"

namespace o2::its
{
class CellSeed;
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
                                 const int layer,
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
                                 const int iteration,
                                 const float NSigmaCut,
                                 bounded_vector<float>& phiCuts,
                                 const float resolutionPV,
                                 std::array<float, NLayers>& minR,
                                 std::array<float, NLayers>& maxR,
                                 bounded_vector<float>& resolutions,
                                 std::vector<float>& radii,
                                 bounded_vector<float>& mulScatAng,
                                 o2::its::ExternalAllocator* alloc,
                                 gpu::Streams& streams);

template <int NLayers>
void computeTrackletsInROFsHandler(const IndexTableUtils<NLayers>* utils,
                                   const typename ROFMaskTable<NLayers>::View& rofMask,
                                   const int layer,
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
                                   const int iteration,
                                   const float NSigmaCut,
                                   bounded_vector<float>& phiCuts,
                                   const float resolutionPV,
                                   std::array<float, NLayers>& minR,
                                   std::array<float, NLayers>& maxR,
                                   bounded_vector<float>& resolutions,
                                   std::vector<float>& radii,
                                   bounded_vector<float>& mulScatAng,
                                   o2::its::ExternalAllocator* alloc,
                                   gpu::Streams& streams);

template <int NLayers>
void countCellsHandler(const Cluster** sortedClusters,
                       const Cluster** unsortedClusters,
                       const TrackingFrameInfo** tfInfo,
                       Tracklet** tracklets,
                       int** trackletsLUT,
                       const int nTracklets,
                       const int layer,
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
                         const int layer,
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
                                int* neighboursLUTs,
                                int** cellsLUTs,
                                gpuPair<int, int>* cellNeighbours,
                                int* neighboursIndexTable,
                                const Tracklet** tracklets,
                                const float maxChi2ClusterAttachment,
                                const float bz,
                                const int layerIndex,
                                const unsigned int nCells,
                                const unsigned int nCellsNext,
                                const int maxCellNeighbours,
                                o2::its::ExternalAllocator* alloc,
                                gpu::Stream& stream);

template <int NLayers>
void computeCellNeighboursHandler(CellSeed** cellsLayersDevice,
                                  int* neighboursLUTs,
                                  int** cellsLUTs,
                                  gpuPair<int, int>* cellNeighbours,
                                  int* neighboursIndexTable,
                                  const Tracklet** tracklets,
                                  const float maxChi2ClusterAttachment,
                                  const float bz,
                                  const int layerIndex,
                                  const unsigned int nCells,
                                  const unsigned int nCellsNext,
                                  const int maxCellNeighbours,
                                  gpu::Stream& stream);

int filterCellNeighboursHandler(gpuPair<int, int>*,
                                int*,
                                unsigned int,
                                gpu::Stream&,
                                o2::its::ExternalAllocator* = nullptr);

template <int NLayers>
void processNeighboursHandler(const int startLayer,
                              const int startLevel,
                              CellSeed** allCellSeeds,
                              CellSeed* currentCellSeeds,
                              std::array<int, NLayers - 2>& nCells,
                              const unsigned char** usedClusters,
                              std::array<int*, NLayers - 2>& neighbours,
                              gsl::span<int*> neighboursDeviceLUTs,
                              const TrackingFrameInfo** foundTrackingFrameInfo,
                              bounded_vector<TrackSeed<NLayers>>& seedsHost,
                              const float bz,
                              const float MaxChi2ClusterAttachment,
                              const float maxChi2NDF,
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
                           const int startLevel,
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
                             o2::its::TrackITSExt* tracks,
                             const int* seedLUT,
                             const std::vector<float>& layerRadiiHost,
                             const std::vector<float>& minPtsHost,
                             const std::vector<float>& layerxX0Host,
                             const unsigned int nSeeds,
                             const unsigned int nTracks,
                             const float Bz,
                             const int startLevel,
                             const float maxChi2ClusterAttachment,
                             const float maxChi2NDF,
                             const int reseedIfShorter,
                             const bool repeatRefitOut,
                             const bool shiftRefToCluster,
                             const o2::base::Propagator* propagator,
                             const o2::base::PropagatorF::MatCorrType matCorrType,
                             o2::its::ExternalAllocator* alloc);

} // namespace o2::its
#endif // ITSTRACKINGGPU_TRACKINGKERNELS_H_
