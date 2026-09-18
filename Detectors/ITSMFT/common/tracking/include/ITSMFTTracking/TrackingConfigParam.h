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
#include <cstddef>
#include <cstdint>
#include <limits>

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

/// Shared common-CA controls, with independent ITS and MFT parameter instances.
/// The ITS key remains distinct from the legacy ITSCATrackerParam configuration.
template <int Detector>
struct TrackerParamConfig : public o2::conf::ConfigurableParamHelper<TrackerParamConfig<Detector>> {
  static_assert(Detector == o2::detectors::DetID::ITS || Detector == o2::detectors::DetID::MFT);
  static constexpr int NLayers = Detector == o2::detectors::DetID::ITS ? tracking::ITSNLayers : tracking::MFTNLayers;
  static constexpr int MinTrackLength = tracking::kCAMinTrackLength;
  static constexpr int MaxTrackLength = NLayers;

  int addTimeError[NLayers] = {0};                                                    // Tracking window width in BC.
  int minTrackLgtIter[tracking::MaxIter] = {};                                        // Async minimum track length per iteration; <=0 keeps preset.
  uint32_t startLayerMask[tracking::MaxIter] = {};                                    // Per-pass starts; 0 keeps the preset, bits must name detector layers.
  int maxHolesIter[tracking::MaxIter] = {};                                           // Maximum missing internal layers per iteration.
  uint16_t holeLayerMask = 0;                                                         // Detector layers that may be absent from accepted tracks.
  float minPtIterLgt[tracking::MaxIter * (MaxTrackLength - MinTrackLength + 1)] = {}; // Async minimum pT by track length; <=0 keeps preset.
  float sysErr2Row[NLayers] = {0};                                                    // Additional sensor-row variance for cluster covariance and candidate windows (cm^2).
  float sysErr2Col[NLayers] = {0};                                                    // Additional sensor-column variance for cluster covariance and candidate windows (cm^2).
  float maxChi2ClusterAttachment = -1.f;
  float maxChi2NDF = -1.f;
  float nSigmaCut = -1.f;
  float minPt = -1.f;
  float pvRes = -1.f;
  int LUTbinsU = 64;                                               // Longitudinal ITS bins or radial MFT bins (cm).
  int LUTbinsV = Detector == o2::detectors::DetID::ITS ? 32 : 128; // Phi bins (radians).
  bool useDiamond = Detector == o2::detectors::DetID::MFT;
  float diamondPos[3] = {0.f, 0.f, 0.f}; // Diamond vertex position (cm).
  int trackingMode = -1;                 // -1: use --tracking-mode; 0: sync, 1: async, 2: cosmics, 3: off.
  int nIterations = -1;                  // -1 uses all mode preset passes; otherwise a positive limit no larger than the preset.
  bool shiftRefToCluster{true};          // Shift the linearization reference to the cluster after update.
  bool repeatRefitOut{false};            // Repeat outward refit using the inward refit as a seed.
  bool createArtefactLabels{false};      // Create labels for artefacts on the fly.

  int nThreads = 1;
  size_t maxMemory = std::numeric_limits<size_t>::max();
  bool dropTFUponFailure = false;

  // Selection of tracks sharing clusters.
  bool allowSharingFirstCluster = false;  // Allow sharing the first cluster.
  float sharedClusterMaxDeltaPhi = 0.05f; // Maximum delta phi at the cluster.
  float sharedClusterMaxDeltaEta = 0.03f; // Maximum delta eta at the cluster.
  bool sharedClusterOppositeSign = false; // Require opposite-sign tracklets.

  O2ParamDef(TrackerParamConfig, Detector == o2::detectors::DetID::ITS ? "ITSCommonCATrackerParam" : "MFTCATrackerParam");
};

template <int Detector>
TrackerParamConfig<Detector> TrackerParamConfig<Detector>::sInstance;

using ITSCommonCATrackerParam = TrackerParamConfig<o2::detectors::DetID::ITS>;
using MFTCATrackerParam = TrackerParamConfig<o2::detectors::DetID::MFT>;

} // namespace o2::itsmft

#endif /* ALICEO2_ITSMFT_TRACKING_CONFIG_PARAM_H_ */
