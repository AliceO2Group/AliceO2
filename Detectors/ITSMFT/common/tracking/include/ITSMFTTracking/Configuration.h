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
/// \brief Shared CA tracking configuration for ITS and MFT
///

#ifndef ALICEO2_ITSMFT_TRACKING_CONFIGURATION_H_
#define ALICEO2_ITSMFT_TRACKING_CONFIGURATION_H_

#include <cstddef>
#include <cstdint>

#ifndef GPUCA_GPUCODE_DEVICE
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#endif

#include "CommonUtils/EnumFlags.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTTracking/LayerMask.h"
#include "ITSMFTTracking/TrackingConfigParam.h"

namespace o2::itsmft
{

inline constexpr int ClustersPerCell = 3;

// Dedicated steps in an iteration.
enum class IterationStep : uint16_t {
  FirstPass = 0,
  RebuildClusterLUT = 1,
  UseUPCMask = 2,
  SelectUPCVertices = 3,
};
using IterationSteps = o2::utils::EnumFlags<IterationStep>;

// Time-frame execution policy, invariant across tracking passes. Thread
// scheduling remains in the workflow's resolved TrackerOptions.
struct TrackingExecutionPolicy {
  size_t MaxMemory = std::numeric_limits<size_t>::max();
  bool DropTFUponFailure = false;
};

// Parameters that may change from one tracking pass to the next.
struct IterationParameters {
  tracking::LayerMask getActiveLayerMask() const noexcept
  {
    return tracking::LayerMask::span(0, NLayers - 1) & ~InactiveLayerMask;
  }

  tracking::LayerMask getSeedingLayerMask() const noexcept
  {
    const auto activeLayers = getActiveLayerMask();
    return SeedingLayers.empty() ? activeLayers : (SeedingLayers & activeLayers);
  }

  int getNSeedingLayers() const noexcept
  {
    return getSeedingLayerMask().count();
  }

  int getMinSeedingClusters() const noexcept
  {
    const int minClusters = MinTrackLength - (MaxHoles > 0 ? MaxHoles : 0);
    const int minClustersWithCells = minClusters > ClustersPerCell ? minClusters : ClustersPerCell;
    const int nSeedingLayers = getNSeedingLayers();
    return minClustersWithCells < nSeedingLayers ? minClustersWithCells : nSeedingLayers;
  }

  int CellMinimumLevel() const noexcept
  {
    return getMinSeedingClusters() - ClustersPerCell + 1;
  }
  IterationSteps PassFlags{IterationStep::FirstPass, IterationStep::RebuildClusterLUT};
  int NLayers = tracking::ITSNLayers;
  bool UseDiamond = false;
  float Diamond[3] = {0.f, 0.f, 0.f};
  float DiamondCov[6] = {25.e-6f, 0.f, 0.f, 25.e-6f, 0.f, 36.f};

  /// General parameters
  int MinTrackLength = 7;
  int MaxHoles = 0;
  // Positional static-graph surfaces disabled for this tracking pass.
  tracking::LayerMask InactiveLayerMask = 0;
  // Positional layers used to build tracklets, cells, and roads. Empty means all active layers.
  tracking::LayerMask SeedingLayers = 0;
  float NSigmaCut = 5;
  float PVres = 1.e-2f;
  /// Trackleting cuts
  float TrackletMinPt = 0.3f;
  /// Fitter parameters
  float MaxChi2ClusterAttachment = 60.f;
  float MaxChi2NDF = 30.f;
  std::vector<float> MinPt = {0.f, 0.f, 0.f, 0.f};
  tracking::LayerMask StartLayerMask = 0x7F;
  bool RepeatRefitOut = false;   // Repeat outward refit using inward refit as a seed.
  bool ShiftRefToCluster = true; // Shift the linearization reference to the cluster after an update.
  bool PerPrimaryVertexProcessing = false;
  bool CreateArtefactLabels{false};
  // Track-sharing selections.
  bool AllowSharingFirstCluster = false;
  float SharedClusterMaxDeltaPhi = 0.05f; // Maximum delta phi at a shared cluster.
  float SharedClusterMaxDeltaEta = 0.03f; // Maximum delta eta at a shared cluster.
  bool SharedClusterOppositeSign = false; // Require opposite-sign tracklets.
  int SharedMaxClusters = 0;              // Maximum shared clusters, excluding the first.
};

// Detector inputs accepted by the configuration interface. Tracker consumes
// these once to construct DetectorConfiguration; they are not retained in the
// per-iteration configuration.
struct DetectorParameters {
  std::vector<uint32_t> AddTimeError = {0, 0, 0, 0, 0, 0, 0};
  std::vector<float> LayerResolution = {5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f};
  std::vector<float> SystError2Row = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; // Systematic row error squared per layer (ALPIDE X).
  std::vector<float> SystError2Col = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; // Systematic column error squared per layer (ALPIDE Z).
  int ColBins{256};                                                       // ITS: ZBins
  int RowBins{128};                                                       // ITS: PhiBins
};

// Single-pass host defaults/input bundle. Production plans store detector
// inputs and execution policy once, separately from the iteration records.
struct TrackingParameters : IterationParameters, DetectorParameters, TrackingExecutionPolicy {
};

struct TrackingPlan {
  DetectorParameters detector;
  TrackingExecutionPolicy execution;
  std::vector<IterationParameters> iterations;
};

namespace TrackingMode
{
enum Type : int8_t {
  Unset = -1,
  Sync = 0,
  Async = 1,
  Cosmics = 2,
  Off = 3,
};

Type fromString(std::string_view str);
std::string toString(Type mode);
TrackingPlan getTrackingPlan(o2::detectors::DetID::ID detId, Type mode);

} // namespace TrackingMode

} // namespace o2::itsmft

#endif /* ALICEO2_ITSMFT_TRACKING_CONFIGURATION_H_ */
