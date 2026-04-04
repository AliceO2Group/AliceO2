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

#include <cstdint>
#ifndef GPUCA_GPUCODE_DEVICE
#include <array>
#include <limits>
#include <string>
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
  std::vector<uint32_t> AddTimeError = {0, 0, 0, 0, 0, 0, 0};
  std::vector<float> LayerZ = {16.333f + 1, 16.333f + 1, 16.333f + 1, 42.140f + 1, 42.140f + 1, 73.745f + 1, 73.745f + 1};
  std::vector<float> LayerRadii = {2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
  std::vector<float> LayerxX0 = {5.e-3f, 5.e-3f, 5.e-3f, 1.e-2f, 1.e-2f, 1.e-2f, 1.e-2f};
  std::vector<float> LayerResolution = {5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f};
  std::vector<float> SystErrorY2 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  std::vector<float> SystErrorZ2 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int ZBins{256};
  int PhiBins{128};
  bool UseDiamond = false;
  float Diamond[3] = {0.f, 0.f, 0.f};
  float DiamondCov[6] = {25.e-6f, 0.f, 0.f, 25.e-6f, 0.f, 36.f};

  /// General parameters
  bool AllowSharingFirstCluster = false;
  int ClusterSharing = 0;
  int MinTrackLength = 7;
  float NSigmaCut = 5;
  float PVres = 1.e-2f;
  /// Trackleting cuts
  float TrackletMinPt = 0.3f;
  /// Cell finding cuts
  float CellDeltaTanLambdaSigma = 0.007f;
  /// Fitter parameters
  o2::base::PropagatorImpl<float>::MatCorrType CorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE;
  float MaxChi2ClusterAttachment = 60.f;
  float MaxChi2NDF = 30.f;
  int ReseedIfShorter = 6; // reseed for the final fit track with the length shorter than this
  std::vector<float> MinPt = {0.f, 0.f, 0.f, 0.f};
  uint16_t StartLayerMask = 0x7F;
  bool RepeatRefitOut = false;   // repeat outward refit using inward refit as a seed
  bool ShiftRefToCluster = true; // TrackFit: after update shift the linearization reference to cluster
  bool PerPrimaryVertexProcessing = false;
  bool SaveTimeBenchmarks = false;
  bool DoUPCIteration = false;
  bool FataliseUponFailure = true;

  bool createArtefactLabels{false};

  bool PrintMemory = false; // print allocator usage in epilog report
  size_t MaxMemory = std::numeric_limits<size_t>::max();
  bool DropTFUponFailure = false;
};

struct VertexingParameters {
  std::string asString() const;

  int nIterations = 1; // Number of vertexing passes to perform
  std::vector<float> LayerZ = {16.333f + 1, 16.333f + 1, 16.333f + 1, 42.140f + 1, 42.140f + 1, 73.745f + 1, 73.745f + 1};
  std::vector<float> LayerRadii = {2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
  int ZBins{1};
  int PhiBins{128};
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

  int nThreads = 1;
  bool PrintMemory = false; // print allocator usage in epilog report
  size_t MaxMemory = std::numeric_limits<size_t>::max();
  bool DropTFUponFailure = false;
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
