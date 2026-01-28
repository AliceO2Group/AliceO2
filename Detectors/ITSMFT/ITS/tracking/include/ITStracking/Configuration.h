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
/// \file Configuration.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_CONFIGURATION_H_
#define TRACKINGITSU_INCLUDE_CONFIGURATION_H_

#ifndef GPUCA_GPUCODE_DEVICE
#include <array>
#include <limits>
#include <vector>
#include <cmath>
#endif

#include "DetectorsBase/Propagator.h"
#include "ITStracking/Constants.h"

namespace o2::its
{

struct TrackingParameters {
  int CellMinimumLevel() const noexcept { return MinTrackLength - constants::ClustersPerCell + 1; }
  int NeighboursPerRoad() const noexcept { return NLayers - 3; }
  int CellsPerRoad() const noexcept { return NLayers - 2; }
  int TrackletsPerRoad() const noexcept { return NLayers - 1; }
  std::string asString() const;

  int NLayers = 7;
  int DeltaROF = 0;
  std::vector<float> LayerZ = {16.333f + 1, 16.333f + 1, 16.333f + 1, 42.140f + 1, 42.140f + 1, 73.745f + 1, 73.745f + 1};
  std::vector<float> LayerRadii = {2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
  std::vector<float> LayerxX0 = {5.e-3f, 5.e-3f, 5.e-3f, 1.e-2f, 1.e-2f, 1.e-2f, 1.e-2f};
  std::vector<float> LayerResolution = {5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f};
  std::vector<float> SystErrorY2 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  std::vector<float> SystErrorZ2 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int ZBins{256};
  int PhiBins{128};
  int nROFsPerIterations = -1;
  bool UseDiamond = false;
  float Diamond[3] = {0.f, 0.f, 0.f};

  /// General parameters
  bool AllowSharingFirstCluster = false;
  int ClusterSharing = 0;
  int MinTrackLength = 7;
  float NSigmaCut = 5;
  float PVres = 1.e-2f;
  /// Trackleting cuts
  float TrackletMinPt = 0.3f;
  float TrackletsPerClusterLimit = 2.f;
  /// Cell finding cuts
  float CellDeltaTanLambdaSigma = 0.007f;
  float CellsPerClusterLimit = 2.f;
  /// Fitter parameters
  o2::base::PropagatorImpl<float>::MatCorrType CorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE;
  float MaxChi2ClusterAttachment = 60.f;
  float MaxChi2NDF = 30.f;
  int ReseedIfShorter = 6; // reseed for the final fit track with the length shorter than this
  std::vector<float> MinPt = {0.f, 0.f, 0.f, 0.f};
  uint16_t StartLayerMask = 0x7F;
  bool RepeatRefitOut = false;   // repeat outward refit using inward refit as a seed
  bool ShiftRefToCluster = true; // TrackFit: after update shift the linearization reference to cluster
  bool FindShortTracks = false;
  bool PerPrimaryVertexProcessing = false;
  bool SaveTimeBenchmarks = false;
  bool DoUPCIteration = false;
  bool FataliseUponFailure = true;
  /// Cluster attachment
  bool UseTrackFollower = false;
  bool UseTrackFollowerTop = false;
  bool UseTrackFollowerBot = false;
  bool UseTrackFollowerMix = false;
  float TrackFollowerNSigmaCutZ = 1.f;
  float TrackFollowerNSigmaCutPhi = 1.f;

  bool createArtefactLabels{false};

  bool PrintMemory = false; // print allocator usage in epilog report
  size_t MaxMemory = std::numeric_limits<size_t>::max();
  bool DropTFUponFailure = false;
};

struct VertexingParameters {
  std::string asString() const;

  int nIterations = 1;         // Number of vertexing passes to perform
  int vertPerRofThreshold = 0; // Maximum number of vertices per ROF to trigger second a round
  bool allowSingleContribClusters = false;
  std::vector<float> LayerZ = {16.333f + 1, 16.333f + 1, 16.333f + 1, 42.140f + 1, 42.140f + 1, 73.745f + 1, 73.745f + 1};
  std::vector<float> LayerRadii = {2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
  int ZBins{1};
  int PhiBins{128};
  int deltaRof = 0;
  float zCut = 0.002f;
  float phiCut = 0.005f;
  float pairCut = 0.04f;
  float clusterCut = 0.8f;
  float histPairCut = 0.04f;
  float tanLambdaCut = 0.002f;     // tanLambda = deltaZ/deltaR
  float lowMultBeamDistCut = 0.1f; // XY cut for low-multiplicity pile up
  int vertNsigmaCut = 6;           // N sigma cut for vertex XY
  float vertRadiusSigma = 0.33f;   // sigma of vertex XY
  float trackletSigma = 0.01f;     // tracklet to vertex sigma
  float maxZPositionAllowed = 25.f;
  int clusterContributorsCut = 16;
  int maxTrackletsPerCluster = 2e3;
  int phiSpan = -1;
  int zSpan = -1;
  bool SaveTimeBenchmarks = false;

  bool useTruthSeeding = false; // overwrite found vertices with MC events
  bool outputContLabels = false;

  int nThreads = 1;
  bool PrintMemory = false; // print allocator usage in epilog report
  size_t MaxMemory = std::numeric_limits<size_t>::max();
  bool DropTFUponFailure = false;
};

struct TimeFrameGPUParameters {
  std::string asString() const;

  size_t tmpCUBBufferSize = 1e5; // In average in pp events there are required 4096 bytes
  size_t maxTrackletsPerCluster = 1e2;
  size_t clustersPerLayerCapacity = 2.5e5;
  size_t clustersPerROfCapacity = 1.5e3;
  size_t validatedTrackletsCapacity = 1e3;
  size_t cellsLUTsize = validatedTrackletsCapacity;
  size_t maxNeighboursSize = 1e2;
  size_t neighboursLUTsize = maxNeighboursSize;
  size_t maxRoadPerRofSize = 1e3; // pp!
  size_t maxLinesCapacity = 1e2;
  size_t maxVerticesCapacity = 5e4;
  size_t nMaxROFs = 1e3;
  size_t nTimeFrameChunks = 3;
  size_t nROFsPerChunk = 768; // pp defaults
  int maxGPUMemoryGB = -1;
};

namespace TrackingMode
{
enum Type : int8_t {
  Unset = -1, // Special value to leave a default in case we want to override via Configurable Params
  Sync = 0,
  Async = 1,
  Cosmics = 2,
  Off = 3,
};

Type fromString(std::string_view str);
std::string toString(Type mode);

std::vector<TrackingParameters> getTrackingParameters(Type mode);
std::vector<VertexingParameters> getVertexingParameters(Type mode);

}; // namespace TrackingMode

} // namespace o2::its

#endif /* TRACKINGITSU_INCLUDE_CONFIGURATION_H_ */
