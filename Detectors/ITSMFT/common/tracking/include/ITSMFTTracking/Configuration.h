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

#include <cstdint>

#ifndef GPUCA_GPUCODE
#include <gsl/span>
#include "ITSMFTTracking/SurfaceDescriptor.h"
#endif

#ifndef GPUCA_GPUCODE_DEVICE
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#endif

#include "CommonUtils/EnumFlags.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "GPUCommonMath.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/LayerMask.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "ITSMFTTracking/ITSTrackingConfigParam.h"

namespace o2::itsmft
{

inline constexpr int ClustersPerCell = 3;

// Dedicated steps in an iteration.
enum class IterationStep : uint16_t {
  FirstPass = 0,
  RebuildClusterLUT = 1,
  UseUPCMask = 2,
  SelectUPCVertices = 3,
  // Reserved for legacy vertexing/follower configurations; the common
  // tracker does not implement these steps.
  ResetVertices = 4,
  SkipROFsAboveThreshold = 5,
  MarkVerticesAsUPC = 6,
  TrackFollowerTop = 7,
  TrackFollowerBot = 8,
};
using IterationSteps = o2::utils::EnumFlags<IterationStep>;

static_assert(sizeof(IterationStep) == sizeof(uint16_t));
static_assert(sizeof(IterationSteps) == sizeof(uint16_t));
static_assert(static_cast<uint16_t>(IterationStep::FirstPass) == 0);
static_assert(static_cast<uint16_t>(IterationStep::RebuildClusterLUT) == 1);
static_assert(static_cast<uint16_t>(IterationStep::UseUPCMask) == 2);
static_assert(static_cast<uint16_t>(IterationStep::SelectUPCVertices) == 3);
static_assert(static_cast<uint16_t>(IterationStep::ResetVertices) == 4);
static_assert(static_cast<uint16_t>(IterationStep::SkipROFsAboveThreshold) == 5);
static_assert(static_cast<uint16_t>(IterationStep::MarkVerticesAsUPC) == 6);
static_assert(static_cast<uint16_t>(IterationStep::TrackFollowerTop) == 7);
static_assert(static_cast<uint16_t>(IterationStep::TrackFollowerBot) == 8);

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

  tracking::LayerMask getNonSeedingLayerMask() const noexcept
  {
    return tracking::LayerMask::span(0, NLayers - 1) & ~getSeedingLayerMask();
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
  int NeighboursPerRoad() const noexcept { return getNSeedingLayers() - 3; }
  int CellsPerRoad() const noexcept { return getNSeedingLayers() - 2; }
  int TrackletsPerRoad() const noexcept { return getNSeedingLayers() - 1; }
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
  // Common tracking applies nominal descriptor material; NONE disables external providers only.
  o2::base::PropagatorImpl<float>::MatCorrType CorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE;
  float MaxChi2ClusterAttachment = 60.f;
  float MaxChi2NDF = 30.f;
  int ReseedIfShorter = 6; // Reseed final fit tracks shorter than this.
  std::vector<float> MinPt = {0.f, 0.f, 0.f, 0.f};
  tracking::LayerMask StartLayerMask = 0x7F;
  bool RepeatRefitOut = false;   // Repeat outward refit using inward refit as a seed.
  bool ShiftRefToCluster = true; // Shift the linearization reference to the cluster after an update.
  bool PerPrimaryVertexProcessing = false;
  bool DoUPCIteration = false;
  bool CreateArtefactLabels{false};
  // Reserved compatibility storage; top/bottom followers are unused by the common tracker.
  float TrackFollowerNSigmaCutZ = 1.f;
  float TrackFollowerNSigmaCutPhi = 1.f;
  int TrackFollowerMaxHypotheses = 1;

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
  std::vector<float> LayerZ{tracking::kITSLookupZHalfExtent.begin(), tracking::kITSLookupZHalfExtent.end()};
  std::vector<float> LayerColHalfExtent{}; // Legacy PhiZ helper extent (cm); production lookup uses descriptor chartRange.
  float IndexRowMin{0.f};                  // Reserved legacy bound; production phi lookup starts at 0.
  float IndexRowMax{0.f};                  // Reserved legacy bound; production phi lookup ends at TwoPI.
  std::vector<float> LayerRadii = {2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
  std::vector<float> LayerResolution = {5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f};
  std::vector<float> SystError2Row = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; // Systematic row error squared per layer (ALPIDE X).
  std::vector<float> SystError2Col = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; // Systematic column error squared per layer (ALPIDE Z).
  int ColBins{256};                                                       // ITS: ZBins
  int RowBins{128};                                                       // ITS: PhiBins
};

// Single-pass host defaults/input bundle. Production plans store detector
// inputs and execution policy once, separately from the iteration records.
struct TrackingParameters : IterationParameters, DetectorParameters, TrackingExecutionPolicy {
  std::string asString() const;
};

struct TrackingPlan {
  DetectorParameters detector;
  TrackingExecutionPolicy execution;
  std::vector<IterationParameters> iterations;
};

#ifndef GPUCA_GPUCODE

inline bool isRecognizedMatCorrType(o2::base::PropagatorF::MatCorrType corrType) noexcept
{
  return corrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE ||
         corrType == o2::base::PropagatorF::MatCorrType::USEMatCorrTGeo ||
         corrType == o2::base::PropagatorF::MatCorrType::USEMatCorrLUT;
}

struct AttachHitConfigView {
  tracking::SurfaceCatalogView catalog;
  o2::base::PropagatorF::MatCorrType corrType{o2::base::PropagatorF::MatCorrType::USEMatCorrNONE};

  bool isValid(size_t expectedLayers) const noexcept
  {
    if (catalog.nSurfaces < expectedLayers || !catalog.surfaces || !isRecognizedMatCorrType(corrType)) {
      return false;
    }
    for (size_t layer = 0; layer < expectedLayers; ++layer) {
      const auto& material = catalog.surfaces[layer].material;
      if (!o2::gpu::GPUCommonMath::Finite(material.xOverX0) || material.xOverX0 < 0.f ||
          !o2::gpu::GPUCommonMath::Finite(material.arealDensityGPerCm2) || material.arealDensityGPerCm2 < 0.f) {
        return false;
      }
    }
    return true;
  }
};

inline AttachHitConfigView bindAttachHitConfig(tracking::SurfaceCatalogView catalog,
                                               const IterationParameters& params) noexcept
{
  return {catalog, params.CorrType};
}

namespace tracking
{

enum class MaterialCorrectionModeSupport : uint8_t {
  Supported,
  Unsupported,
  InvalidMode,
  InvalidSurfaceKind
};

inline MaterialCorrectionModeSupport materialCorrectionModeSupport(
  SurfaceKind kind, o2::base::PropagatorF::MatCorrType corrType) noexcept
{
  if (!isRecognizedMatCorrType(corrType)) {
    return MaterialCorrectionModeSupport::InvalidMode;
  }
  if (kind != SurfaceKind::Cylinder && kind != SurfaceKind::Disk) {
    return MaterialCorrectionModeSupport::InvalidSurfaceKind;
  }
  if (corrType != o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
    return MaterialCorrectionModeSupport::Unsupported;
  }
  return MaterialCorrectionModeSupport::Supported;
}

} // namespace tracking

#endif

/// Reset tracking parameters to detector geometry defaults.
void resetDetectorDefaults(TrackingParameters& params, o2::detectors::DetID::ID detId);

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
// Field-independent validation of common-CA public aliases.
void validateCommonCAOptions(detectors::DetID::ID detId);
TrackingPlan getTrackingPlan(o2::detectors::DetID::ID detId, Type mode);

} // namespace TrackingMode

struct VertexingParameters {
  std::string asString() const;

  IterationSteps PassFlags{IterationStep::FirstPass, IterationStep::ResetVertices};
  std::vector<float> LayerZ = {16.333f + 1, 16.333f + 1, 16.333f + 1, 42.140f + 1, 42.140f + 1, 73.745f + 1, 73.745f + 1};
  std::vector<float> LayerRadii = {2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
  int vertPerRofThreshold = 0; // Vertices per ROF that trigger a second round.
  int ColBins = 1;
  int RowBins = 128;
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

  bool useTruthSeeding = false; // Replace found vertices with MC events.

  int nThreads = 1;
  bool PrintMemory = false; // Print allocator usage in the epilog report.
  size_t MaxMemory = std::numeric_limits<size_t>::max();
  bool DropTFUponFailure = false;
};

} // namespace o2::itsmft

namespace o2::itsmft::tracking
{

/// MFT uses o2::itsmft::TrackerParamConfig; ITS keeps its legacy parameter type.
template <o2::detectors::DetID::ID DetId>
struct TrackerParamRef;

template <>
struct TrackerParamRef<o2::detectors::DetID::MFT> {
  using Type = o2::itsmft::TrackerParamConfig<o2::detectors::DetID::MFT>;
  static const Type& get() { return Type::Instance(); }
  static constexpr int nLayers() { return Type::getNLayers(); }
};

template <>
struct TrackerParamRef<o2::detectors::DetID::ITS> {
  using Type = o2::its::TrackerParamConfig;
  static const Type& get() { return Type::Instance(); }
  static constexpr int nLayers() { return ITSNLayers; }
};

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_CONFIGURATION_H_ */
