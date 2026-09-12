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

#ifndef ALICEO2_ITSMFT_TRACKING_CONFIG_PARAM_H_
#define ALICEO2_ITSMFT_TRACKING_CONFIG_PARAM_H_

#include <array>
#include <limits>
#include <string>
#include <string_view>

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "DetectorsCommonDataFormats/DetID.h"

namespace o2::itsmft::tracking
{
/// ITS CA layer count.
constexpr int ITSNLayers = 7;
/// MFT CA half-disk layer count.
constexpr int MFTNLayers = 10;
/// Maximum CA iterations.
constexpr int MaxIter = 4;
/// Minimum accepted CA track length for the detector presets.
constexpr int kCAMinTrackLength = 4;
inline constexpr std::array<float, ITSNLayers> kITSLookupZHalfExtent{
  16.333f + 1.f, 16.333f + 1.f, 16.333f + 1.f,
  42.140f + 1.f, 42.140f + 1.f, 73.745f + 1.f, 73.745f + 1.f};
} // namespace o2::itsmft::tracking

namespace o2::itsmft
{

/// Minimal configuration for opt-in ITS common-CA tracking.
/// It does not use the registered name "ITSCATrackerParam", which belongs to the
/// legacy o2::its::TrackerParamConfig.
/// Implemented workflow controls plus reserved diagnostic aliases; unsupported
/// overrides are rejected by common-CA option validation. Defaults preserve the detector tracking
/// baseline for both supported modes.
///
/// diamondPos, pvRes, and useDiamond define the static vertex/beam constraint
/// consumed by the shared TrackerTraits.
struct ITSCommonCATrackerParam : public o2::conf::ConfigurableParamHelper<ITSCommonCATrackerParam> {
  bool dropTFUponFailure = false;
  bool printMemory = false; // Reserved alias: true is rejected (no memory report).
  size_t maxMemory = std::numeric_limits<size_t>::max();
  bool saveTimeBenchmarks = false; // Reserved alias: true is rejected (no benchmark writer).
  bool useDiamond = false;
  float diamondPos[3] = {0.f, 0.f, 0.f}; // Diamond vertex position when useDiamond is set.
  float pvRes = -1.f;                    // Diamond-vertex PV resolution; <=0 keeps the default.
  uint16_t holeLayerMask = 0;            // Detector layers that may be absent from accepted tracks.

  /// Number of tbb::task_arena threads for the ITS common-CA tracker.
  /// This dedicated field is separate from the legacy ITS configuration.
  /// Must be > 0; validated where consumed because ConfigurableParam
  /// structs cannot reject construction.
  int nThreads = 1;

  O2ParamDef(ITSCommonCATrackerParam, "ITSCommonCATrackerParam");
};

template <int N>
struct TrackerParamConfig : public o2::conf::ConfigurableParamHelper<TrackerParamConfig<N>> {
  static constexpr std::string_view getParamName()
  {
    return "MFTCATrackerParam";
  }

  static constexpr int MinTrackLength = tracking::kCAMinTrackLength;
  static constexpr int MaxTrackLength = tracking::MFTNLayers;
  static constexpr int getNLayers() { return tracking::MFTNLayers; }

  std::string materialModel = "nominal";                                                          // Implemented provider: nominal descriptor material.
  bool useMatCorrTGeo = false;                                                                    // Legacy alias: true requests unsupported TGeo and is rejected.
  bool useFastMaterial = true;                                                                    // Legacy alias: true selects nominal; false requests unsupported LUT.
  int addTimeError[getNLayers()] = {0};                                                           // Tracking window width in BC.
  int minTrackLgtIter[o2::itsmft::tracking::MaxIter] = {};                                        // Async minimum track length per iteration; <=0 keeps preset.
  uint32_t startLayerMask[o2::itsmft::tracking::MaxIter] = {};                                    // Per-pass starts; 0 keeps the preset, bits must name detector layers.
  int maxHolesIter[o2::itsmft::tracking::MaxIter] = {};                                           // Maximum missing internal layers per iteration.
  uint16_t holeLayerMask = 0;                                                                     // Detector layers that may be absent from accepted tracks.
  float minPtIterLgt[o2::itsmft::tracking::MaxIter * (MaxTrackLength - MinTrackLength + 1)] = {}; // Async minimum pT by track length; <=0 keeps preset.
  float sysErr2Row[getNLayers()] = {0};                                                           // Systematic sensor-row variance for candidate windows (cm^2).
  float sysErr2Col[getNLayers()] = {0};                                                           // Systematic sensor-column variance for candidate windows (cm^2).
  float maxChi2ClusterAttachment = -1.f;
  float maxChi2NDF = -1.f;
  float nSigmaCut = -1.f;
  float deltaTanLres = -1.f; // Reserved alias: overrides are rejected (no consumer).
  float minPt = -1.f;
  float pvRes = -1.f;
  int LUTbinsU = 64;                       // Radial bins in the MFT PhiR index (radius in cm).
  int LUTbinsV = 128;                      // Phi bins in the MFT PhiR index (angle in radians).
  float diamondPos[3] = {0.f, 0.f, 0.f};   // Diamond vertex for MFT seeds (cm).
  bool useDiamond = true;                  // Compatibility constraint: MFT requires true.
  bool perPrimaryVertexProcessing = false; // Compatibility constraint: MFT requires false.
  bool saveTimeBenchmarks = false;         // Reserved alias: true is rejected (no benchmark writer).
  bool overrideBeamEstimation = false;     // Reserved alias: true is rejected (no MFT beam estimation).
  int trackingMode = -1;                   // -1: use --tracking-mode; 0: sync, 1: async, 2: cosmics, 3: off.
  bool doUPCIteration = false;             // Reserved alias: true is rejected (no MFT UPC preset).
  int nIterations = -1;                    // -1 uses all mode preset passes; otherwise a positive limit no larger than the preset.
  int reseedIfShorter = 6;                 // Reserved while reseeding is developed; currently diagnosed as ineffective.
  bool shiftRefToCluster{true};            // Shift the linearization reference to the cluster after update.
  bool repeatRefitOut{false};              // Repeat outward refit using the inward refit as a seed.
  bool createArtefactLabels{false};        // Create labels for artefacts on the fly.

  int nThreads = 1;
  bool printMemory = false; // Reserved alias: true is rejected (no memory report).
  size_t maxMemory = std::numeric_limits<size_t>::max();
  bool dropTFUponFailure = false;
  bool fataliseUponFailure = true; // Reserved alias: false is rejected; use dropTFUponFailure.

  // Selection of tracks sharing clusters.
  bool allowSharingFirstCluster = false;  // Allow sharing the first cluster.
  float sharedClusterMaxDeltaPhi = 0.05f; // Maximum delta phi at the cluster.
  float sharedClusterMaxDeltaEta = 0.03f; // Maximum delta eta at the cluster.
  bool sharedClusterOppositeSign = false; // Require opposite-sign tracklets.

  O2ParamDef(TrackerParamConfig, getParamName().data());

 private:
  static_assert(N == o2::detectors::DetID::MFT, "common ITS settings use ITSCommonCATrackerParam");
};

template <int N>
TrackerParamConfig<N> TrackerParamConfig<N>::sInstance;

} // namespace o2::itsmft

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::itsmft::TrackerParamConfig<o2::detectors::DetID::MFT>> : std::true_type {
};
} // namespace framework

#endif /* ALICEO2_ITSMFT_TRACKING_CONFIG_PARAM_H_ */
