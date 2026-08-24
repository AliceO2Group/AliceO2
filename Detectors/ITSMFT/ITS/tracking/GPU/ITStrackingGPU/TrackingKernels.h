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
#include "ITStrackingGPU/ClusterLinesGPU.h"
#include "DetectorsBase/Propagator.h"

namespace o2::its
{
using o2::itsmft::tracking::bounded_vector;
using o2::itsmft::tracking::CapacityEstimator;

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
                                           const bool vtxMode,
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
                                           bounded_vector<float>& linkPhiCuts,
                                           const float resolutionPV,
                                           const float* minRs,
                                           const float* maxRs,
                                           bounded_vector<float>& resolutions,
                                           std::vector<float>& radii,
                                           bounded_vector<float>& linkMSAngles,
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
                                 const float cellDeltaPhiCut,
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
                                       CapacityEstimator& estimator,
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

  static void sortClustersHandler(const Cluster* unsorted,
                                  Cluster* sorted,
                                  const int* clusterOffsets,
                                  int* indexTable,
                                  const IndexTableUtils<NLayers>* utils,
                                  const typename ROFMaskTable<NLayers>::View& rofMask,
                                  float beamX, float beamY,
                                  int zBins, int phiBins, int nRofs, int nClustersLayer, int iLayer,
                                  float* minRadiusLayer, float* maxRadiusLayer,
                                  int* keys,
                                  int* perm,
                                  o2::its::ExternalAllocator* alloc,
                                  gpu::Stream& stream);

  static void registerClusterOwnershipHandler(const CellSeed* cellsLayersDevice,
                                              const int nCells,
                                              unsigned long long** clusterOwnersDeviceArray,
                                              gpu::Stream& stream);

  static void linearizeCellsToLinesHandler(const int nCells,
                                           const CellSeed* cells,
                                           const unsigned long long* const* clusterOwners,
                                           const int* rofFramesClustersL1,
                                           const int nRofsL1,
                                           const int ownedClustersCut,
                                           gpu::GPULine* lines,
                                           int* lineRof,
                                           int* lineClusters,
                                           int* lineSlots,
                                           const float beamX,
                                           const float beamY,
                                           const float maxZ,
                                           const float minPt,
                                           float* linesZs,
                                           o2::its::TimeEstBC* lineTimes,
                                           float* lineChi2,
                                           float* linePt,
                                           o2::its::ExternalAllocator* alloc,
                                           gpu::Stream& stream);

  static void sortLinesHandler(const int nLines,
                               const int nRofs,
                               const gpu::LineProjSoA soa,
                               const gpu::LineProjSoA sortedSoa,
                               const int* lineRof,
                               int* rofOffsets,
                               o2::its::ExternalAllocator* alloc,
                               gpu::Stream& stream);

  static void scanDensityHandler(const int nLines,
                                 const gpu::LineProjSoA sortedSoa,
                                 const int* rofOffsets,
                                 int* density,
                                 gpu::LineWindow* win,
                                 const float zWindow,
                                 gpu::Stream& stream);

  static void findPeaksHandler(const int nLines,
                               const int nRofs,
                               const gpu::LineProjSoA sortedSoa,
                               const int* rofOffsets,
                               const int* density,
                               const gpu::LineWindow* win,
                               uint8_t* isPeak,
                               const int* densityFine,
                               const gpu::LineWindow* winFine,
                               const int fineMinDensity,
                               uint8_t* isPeakFine,
                               int* peakScan,
                               int* peakLineIdx,
                               int* peakOffsets,
                               o2::its::ExternalAllocator* alloc,
                               gpu::Stream& stream);

  static void fitPeaksHandler(const int* nPeaksDevice,
                              const int* peakLineIdx,
                              const gpu::LineWindow* win,
                              const gpu::LineProjSoA sortedSoa,
                              const gpu::GPULine* lines,
                              const float* lineChi2,
                              const float* linePt,
                              const float goodLineChi2Cut,
                              const float goodLinePtCut,
                              const float pairCut2,
                              const float nSigmaCut,
                              const int minContributors,
                              const float beamX,
                              const float beamY,
                              const uint8_t* isPeakFine,
                              const float fineMaxDrift,
                              gpu::VertexCand* cands,
                              gpu::Stream& stream);

  static void dedupVertexCandidatesHandler(const int* nPeaksDevice,
                                           const int* peakLineIdx,
                                           const int* peakOffsets,
                                           const gpu::LineProjSoA sortedSoa,
                                           const float duplicateZCut,
                                           const float duplicateZScale,
                                           gpu::VertexCand* cands,
                                           gpu::Stream& stream);
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
