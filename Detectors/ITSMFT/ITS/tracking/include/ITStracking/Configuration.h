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
#include <limits>
#include <string>
#include <vector>
#endif

#include "CommonUtils/EnumFlags.h"
#include "DetectorsBase/Propagator.h"
#include "ITStracking/Constants.h"
#include "ITStracking/LayerMask.h"

namespace o2::its
{

// Steering of dedicated steps in an iteration
enum class IterationStep : uint16_t {
  FirstPass = 0,
  RebuildClusterLUT,
  UseUPCMask,
  SelectUPCVertices,
  ResetVertices,
  SkipROFsAboveThreshold,
  MarkVerticesAsUPC,
  TrackFollowerTop,
  TrackFollowerBot,
};
using IterationSteps = o2::utils::EnumFlags<IterationStep>;

struct TrackingParameters {
  LayerMask getActiveLayerMask() const noexcept
  {
    return LayerMask::span(0, NLayers - 1) & ~InactiveLayerMask;
  }

  LayerMask getSeedingLayerMask() const noexcept
  {
    const auto activeLayers = getActiveLayerMask();
    return SeedingLayers.empty() ? activeLayers : (SeedingLayers & activeLayers);
  }

  LayerMask getNonSeedingLayerMask() const noexcept
  {
    return ~(getSeedingLayerMask());
  }

  int getNSeedingLayers() const noexcept
  {
    return getSeedingLayerMask().count();
  }

  int getMinSeedingClusters() const noexcept
  {
    const int minClusters = MinTrackLength - (MaxHoles > 0 ? MaxHoles : 0);
    const int minClustersWithCells = minClusters > constants::ClustersPerCell ? minClusters : constants::ClustersPerCell;
    const int nSeedingLayers = getNSeedingLayers();
    return minClustersWithCells < nSeedingLayers ? minClustersWithCells : nSeedingLayers;
  }

  int CellMinimumLevel() const noexcept
  {
    return getMinSeedingClusters() - constants::ClustersPerCell + 1;
  }
  int NeighboursPerRoad() const noexcept { return getNSeedingLayers() - 3; }
  int CellsPerRoad() const noexcept { return getNSeedingLayers() - 2; }
  int TrackletsPerRoad() const noexcept { return getNSeedingLayers() - 1; }
  std::string asString() const;

  IterationSteps PassFlags{IterationStep::FirstPass, IterationStep::RebuildClusterLUT};
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
  int MinTrackLength = 7;
  int MaxHoles = 0;
  LayerMask HoleLayerMask = 0;
  LayerMask InactiveLayerMask = 0;
  LayerMask SeedingLayers = 0;
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
  LayerMask StartLayerMask = 0x7F;
  bool RepeatRefitOut = false;   // repeat outward refit using inward refit as a seed
  bool ShiftRefToCluster = true; // TrackFit: after update shift the linearization reference to cluster
  bool PerPrimaryVertexProcessing = false;
  bool SaveTimeBenchmarks = false;
  bool DoUPCIteration = false;
  bool FataliseUponFailure = true;
  bool CreateArtefactLabels{false};
  float TrackFollowerNSigmaCutZ = 1.f;
  float TrackFollowerNSigmaCutPhi = 1.f;
  int TrackFollowerMaxHypotheses = 1;
  bool PrintMemory = false; // print allocator usage in epilog report
  size_t MaxMemory = std::numeric_limits<size_t>::max();
  bool DropTFUponFailure = false;

  // Selections on tracks sharing clusters
  bool AllowSharingFirstCluster = false;
  float SharedClusterMaxDeltaPhi = 0.05f; // For tracks sharing clusters, maximum allowed delta phi at the cluster position
  float SharedClusterMaxDeltaEta = 0.03f; // For tracks sharing clusters, maximum allowed delta eta at the cluster position
  bool SharedClusterOppositeSign = false; // For tracks sharing clusters, require opposite sign of the tracklets
  int SharedMaxClusters = 0;              // Maximal allowed shared clusters (excluding first cluster)
};

struct VertexingParameters {
  std::string asString() const;

  IterationSteps PassFlags{IterationStep::FirstPass, IterationStep::ResetVertices};
  std::vector<float> LayerZ = {16.333f + 1, 16.333f + 1, 16.333f + 1, 42.140f + 1, 42.140f + 1, 73.745f + 1, 73.745f + 1};
  std::vector<float> LayerRadii = {2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
  int vertPerRofThreshold = 0; // Maximum number of vertices per ROF to trigger second a round
  int ZBins = 1;
  int PhiBins = 128;
  float zCut = -1.f;
  float phiCut = -1.f;
  float pairCut = -1.f;
  float clusterCut = -1.f;
  float coarseZWindow = -1.f;
  float seedDedupZCut = -1.f;
  float refitDedupZCut = -1.f;
  float duplicateZCut = -1.f;
  float finalSelectionZCut = -1.f;
  float duplicateDistance2Cut = -1.f;
  float tanLambdaCut = -1.f;
  float NSigmaCut = -1;
  float maxZPositionAllowed = -1.f;
  int clusterContributorsCut = -1;
  int suppressLowMultDebris = -1;
  int seedMemberRadiusTime = -1;
  int seedMemberRadiusZ = -1;
  int maxTrackletsPerCluster = -1;
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
