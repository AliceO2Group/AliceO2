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

#include "DetectorsBase/Propagator.h"
#include "GPUCommonDef.h"

namespace o2::its
{
class CellSeed;
class ExternalAllocator;
namespace gpu
{

#ifdef GPUCA_GPUCODE // GPUg() global kernels must only when compiled by GPU compiler

GPUdi() int4 getEmptyBinsRect()
{
  return int4{0, 0, 0, 0};
}

GPUd() bool fitTrack(TrackITSExt& track,
                     int start,
                     int end,
                     int step,
                     float chi2clcut,
                     float chi2ndfcut,
                     float maxQoverPt,
                     int nCl,
                     float Bz,
                     TrackingFrameInfo** tfInfos,
                     const o2::base::Propagator* prop,
                     o2::base::PropagatorF::MatCorrType matCorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE);

template <int nLayers = 7>
GPUg() void fitTrackSeedsKernel(
  CellSeed* trackSeeds,
  const TrackingFrameInfo** foundTrackingFrameInfo,
  o2::its::TrackITSExt* tracks,
  const float* minPts,
  const unsigned int nSeeds,
  const float Bz,
  const int startLevel,
  float maxChi2ClusterAttachment,
  float maxChi2NDF,
  const o2::base::Propagator* propagator,
  const o2::base::PropagatorF::MatCorrType matCorrType = o2::base::PropagatorF::MatCorrType::USEMatCorrLUT);
#endif
} // namespace gpu

template <int nLayers = 7>
void countTrackletsInROFsHandler(const IndexTableUtils* utils,
                                 const uint8_t* multMask,
                                 const int startROF,
                                 const int endROF,
                                 const int maxROF,
                                 const int deltaROF,
                                 const int vertexId,
                                 const Vertex* vertices,
                                 const int* rofPV,
                                 const int nVertices,
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
                                 std::array<float, nLayers>& minR,
                                 std::array<float, nLayers>& maxR,
                                 bounded_vector<float>& resolutions,
                                 std::vector<float>& radii,
                                 bounded_vector<float>& mulScatAng,
                                 o2::its::ExternalAllocator* alloc,
                                 const int nBlocks,
                                 const int nThreads,
                                 gpu::Streams& streams);

template <int nLayers = 7>
void computeTrackletsInROFsHandler(const IndexTableUtils* utils,
                                   const uint8_t* multMask,
                                   const int startROF,
                                   const int endROF,
                                   const int maxROF,
                                   const int deltaROF,
                                   const int vertexId,
                                   const Vertex* vertices,
                                   const int* rofPV,
                                   const int nVertices,
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
                                   std::array<float, nLayers>& minR,
                                   std::array<float, nLayers>& maxR,
                                   bounded_vector<float>& resolutions,
                                   std::vector<float>& radii,
                                   bounded_vector<float>& mulScatAng,
                                   o2::its::ExternalAllocator* alloc,
                                   const int nBlocks,
                                   const int nThreads,
                                   gpu::Streams& streams);

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
                       const int deltaROF,
                       const float bz,
                       const float maxChi2ClusterAttachment,
                       const float cellDeltaTanLambdaSigma,
                       const float nSigmaCut,
                       o2::its::ExternalAllocator* alloc,
                       const int nBlocks,
                       const int nThreads,
                       gpu::Streams& streams);

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
                         const int deltaROF,
                         const float bz,
                         const float maxChi2ClusterAttachment,
                         const float cellDeltaTanLambdaSigma,
                         const float nSigmaCut,
                         const int nBlocks,
                         const int nThreads,
                         gpu::Streams& streams);

void countCellNeighboursHandler(CellSeed** cellsLayersDevice,
                                int* neighboursLUTs,
                                int** cellsLUTs,
                                gpuPair<int, int>* cellNeighbours,
                                int* neighboursIndexTable,
                                const Tracklet** tracklets,
                                const int deltaROF,
                                const float maxChi2ClusterAttachment,
                                const float bz,
                                const int layerIndex,
                                const unsigned int nCells,
                                const unsigned int nCellsNext,
                                const int maxCellNeighbours,
                                o2::its::ExternalAllocator* alloc,
                                const int nBlocks,
                                const int nThreads,
                                gpu::Stream& stream);

void computeCellNeighboursHandler(CellSeed** cellsLayersDevice,
                                  int* neighboursLUTs,
                                  int** cellsLUTs,
                                  gpuPair<int, int>* cellNeighbours,
                                  int* neighboursIndexTable,
                                  const Tracklet** tracklets,
                                  const int deltaROF,
                                  const float maxChi2ClusterAttachment,
                                  const float bz,
                                  const int layerIndex,
                                  const unsigned int nCells,
                                  const unsigned int nCellsNext,
                                  const int maxCellNeighbours,
                                  const int nBlocks,
                                  const int nThreads,
                                  gpu::Stream& stream);

int filterCellNeighboursHandler(gpuPair<int, int>*,
                                int*,
                                unsigned int,
                                gpu::Stream&,
                                o2::its::ExternalAllocator* = nullptr);

template <int nLayers = 7>
void processNeighboursHandler(const int startLayer,
                              const int startLevel,
                              CellSeed** allCellSeeds,
                              CellSeed* currentCellSeeds,
                              std::array<int, nLayers - 2>& nCells,
                              const unsigned char** usedClusters,
                              std::array<int*, nLayers - 2>& neighbours,
                              gsl::span<int*> neighboursDeviceLUTs,
                              const TrackingFrameInfo** foundTrackingFrameInfo,
                              bounded_vector<CellSeed>& seedsHost,
                              const float bz,
                              const float MaxChi2ClusterAttachment,
                              const float maxChi2NDF,
                              const o2::base::Propagator* propagator,
                              const o2::base::PropagatorF::MatCorrType matCorrType,
                              o2::its::ExternalAllocator* alloc,
                              const int nBlocks,
                              const int nThreads);

void trackSeedHandler(CellSeed* trackSeeds,
                      const TrackingFrameInfo** foundTrackingFrameInfo,
                      o2::its::TrackITSExt* tracks,
                      std::vector<float>& minPtsHost,
                      const unsigned int nSeeds,
                      const float Bz,
                      const int startLevel,
                      float maxChi2ClusterAttachment,
                      float maxChi2NDF,
                      const o2::base::Propagator* propagator,
                      const o2::base::PropagatorF::MatCorrType matCorrType,
                      const int nBlocks,
                      const int nThreads);
} // namespace o2::its
#endif // ITSTRACKINGGPU_TRACKINGKERNELS_H_
