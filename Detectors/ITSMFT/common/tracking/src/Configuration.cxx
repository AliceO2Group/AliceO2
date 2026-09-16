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

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <format>
#include <string_view>
#include <stdexcept>
#include <vector>

#include "DetectorsBase/Propagator.h"
#include "Framework/Logger.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "ITSMFTTracking/Constants.h"
#include "MFTTracking/Constants.h"

namespace
{
constexpr bool iequals(std::string_view a, std::string_view b)
{
  return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                    [](char x, char y) { return std::tolower(x) == std::tolower(y); });
}

template <typename Config>
void resolveSystematicErrors(o2::itsmft::DetectorParameters& parameters, const Config& config)
{
  for (size_t layer = 0; layer < std::size(config.sysErr2Row); ++layer) {
    const auto row = config.sysErr2Row[layer];
    const auto col = config.sysErr2Col[layer];
    if (row < 0.f || col < 0.f) {
      throw std::invalid_argument(std::format("{}.sysErr2Row/Col[{}] must be finite nonnegative variances", config.getName(), layer));
    }
    parameters.SystError2Row[layer] = row;
    parameters.SystError2Col[layer] = col;
  }
}
} // namespace

namespace o2::itsmft
{

void resetDetectorDefaults(TrackingParameters& p, detectors::DetID::ID detId)
{
  if (detId == detectors::DetID::ITS) {
    p = TrackingParameters{};
    p.MinPt.assign(tracking::ITSNLayers - tracking::kCAMinTrackLength + 1, 0.f);
    return;
  }

  if (detId == detectors::DetID::MFT) {
    namespace mft = o2::mft::constants::mft;
    constexpr int nLayers = o2::mft::constants::mft::LayersNumber;

    p = TrackingParameters{};
    p.NLayers = nLayers;
    p.LayerResolution.assign(nLayers, mft::Resolution);
    p.SystError2Row.assign(nLayers, 0.f);
    p.SystError2Col.assign(nLayers, 0.f);
    p.AddTimeError.assign(nLayers, 0u);
    p.ColBins = 64;
    p.RowBins = 128;
    p.UseDiamond = true;
    p.PerPrimaryVertexProcessing = false;
    p.StartLayerMask = (1u << nLayers) - 1u;
    p.MinPt.assign(MFTCATrackerParam::MaxTrackLength - MFTCATrackerParam::MinTrackLength + 1, 0.f);
    return;
  }

  LOGP(fatal, "Unsupported detector id {} in resetDetectorDefaults", static_cast<int>(detId));
}

namespace TrackingMode
{

Type fromString(std::string_view str)
{
  constexpr std::array smodes = {
    std::pair{"sync", Sync},
    std::pair{"async", Async},
    std::pair{"cosmics", Cosmics},
    std::pair{"unset", Unset},
    std::pair{"off", Off}};

  const auto it = std::find_if(smodes.begin(), smodes.end(), [&str](const auto& pair) {
    return iequals(str, pair.first);
  });
  if (it == smodes.end()) {
    LOGP(fatal, "Unrecognized CA tracking mode '{}'", str);
  }
  return it->second;
}

std::string toString(Type mode)
{
  switch (mode) {
    case Sync:
      return "sync";
    case Async:
      return "async";
    case Cosmics:
      return "cosmics";
    case Unset:
      return "unset";
    case Off:
      return "off";
  }
  LOGP(fatal, "Unrecognized CA tracking mode {}", static_cast<int>(mode));
  return "";
}

TrackingPlan getTrackingPlan(detectors::DetID::ID detId, Type mode)
{
  TrackingParameters defaults;
  resetDetectorDefaults(defaults, detId);
  TrackingPlan plan{std::move(static_cast<DetectorParameters&>(defaults)), {}, {}};
  auto& trackParams = plan.iterations;
  if (detId == detectors::DetID::ITS) {
    if (mode == Async) {
      trackParams.assign(3, defaults);
      trackParams[1].TrackletMinPt = 0.2f;
      trackParams[2].TrackletMinPt = 0.1f;
      trackParams[0].MinPt[0] = 1.f / 12.f;
      trackParams[1].MinPt[0] = 1.f / 12.f;
      trackParams[2].MinTrackLength = tracking::kCAMinTrackLength;
      trackParams[2].MinPt[0] = 1.f / 12.f;
      trackParams[2].MinPt[1] = 1.f / 5.f;
      trackParams[2].MinPt[2] = 1.f;
      trackParams[2].MinPt[3] = 1.f / 6.f;
      trackParams[2].StartLayerMask = (1u << 6) | (1u << 3);
    } else if (mode == Sync) {
      trackParams.assign(1, defaults);
      trackParams[0].MinTrackLength = tracking::kCAMinTrackLength;
    } else {
      LOGP(fatal, "ITS common-CA tracking mode '{}' is not supported yet; use 'sync' or 'async'", toString(mode));
    }

    plan.detector.ColBins = 64;
    plan.detector.RowBins = 32;
  } else if (detId == detectors::DetID::MFT) {
    if (mode == Off) {
      return plan;
    }
    if (mode == Unset) {
      LOGP(fatal, "CA tracking mode is unset; set --tracking-mode or MFTCATrackerParam.trackingMode");
    }
    if (mode == Async) {
      trackParams.assign(3, defaults);

      trackParams[1].TrackletMinPt = 0.15f;
      trackParams[2].TrackletMinPt = 0.08f;

      trackParams[0].MinPt[0] = 1.f / 12.f; // 10 clusters
      trackParams[1].MinPt[0] = 1.f / 12.f;

      trackParams[2].MinTrackLength = MFTCATrackerParam::MinTrackLength;
      trackParams[2].MinPt[0] = 1.f / 12.f; // 10 clusters
      trackParams[2].MinPt[1] = 1.f / 8.f;  // 9 clusters
      trackParams[2].MinPt[2] = 1.f / 5.f;  // 8 clusters
      trackParams[2].MinPt[3] = 1.f / 3.f;  // 7 clusters
      trackParams[2].MinPt[4] = 1.f / 2.f;  // 6 clusters
      trackParams[2].MinPt[5] = 1.f / 1.f;  // 5 clusters
    } else if (mode == Sync) {
      trackParams.assign(1, defaults);
      trackParams[0].MinTrackLength = MFTCATrackerParam::MinTrackLength;
    } else if (mode == Cosmics) {
      trackParams.assign(1, defaults);
      trackParams[0].MinTrackLength = MFTCATrackerParam::MinTrackLength;
      plan.detector.ColBins = 32;
      plan.detector.RowBins = 64;
      trackParams[0].PVres = 1.e5f;
      trackParams[0].MaxChi2ClusterAttachment = 60.f;
      trackParams[0].MaxChi2NDF = 40.f;
    } else {
      LOGP(fatal, "Unsupported CA tracking mode {}", toString(mode));
    }
  } else {
    LOGP(fatal, "Unsupported detector id {} in getTrackingPlan", static_cast<int>(detId));
  }

  const auto applyOverrides = [&]<typename Config>(const Config& tc) {
    if (mode != Async) {
      if (std::any_of(std::begin(tc.minTrackLgtIter), std::end(tc.minTrackLgtIter), [](int value) { return value > 0; })) {
        throw std::invalid_argument(tc.getName() + ".minTrackLgtIter overrides are implemented only for async mode");
      }
      if (std::any_of(std::begin(tc.minPtIterLgt), std::end(tc.minPtIterLgt), [](float value) { return value > 0.f; })) {
        throw std::invalid_argument(tc.getName() + ".minPtIterLgt overrides are implemented only for async mode");
      }
    }

    if (mode == Async) {
      for (int ip = 0; ip < static_cast<int>(trackParams.size()); ip++) {
        auto& param = trackParams[ip];
        if (ip < tracking::MaxIter) {
          if (tc.minTrackLgtIter[ip] > 0) {
            param.MinTrackLength = tc.minTrackLgtIter[ip];
          }
          for (int ilg = tc.MaxTrackLength; ilg >= tc.MinTrackLength; ilg--) {
            const int lslot0 = tc.MaxTrackLength - ilg;
            const int lslot = lslot0 + ip * (tc.MaxTrackLength - tc.MinTrackLength + 1);
            if (tc.minPtIterLgt[lslot] > 0.f) {
              param.MinPt[lslot0] = tc.minPtIterLgt[lslot];
            }
          }
        }
      }
    }

    if (tc.nIterations != -1 && (tc.nIterations <= 0 || static_cast<size_t>(tc.nIterations) > trackParams.size())) {
      throw std::invalid_argument(std::format("{}.nIterations={} is invalid for {}: use -1 or 1..{}",
                                              tc.getName(), tc.nIterations, toString(mode), trackParams.size()));
    }
    if (tc.nIterations > 0) {
      trackParams.resize(tc.nIterations);
    }
    constexpr uint32_t allowedStartLayers = (uint32_t{1} << Config::NLayers) - 1;
    for (int iteration = 0; iteration < tracking::MaxIter; ++iteration) {
      if (tc.startLayerMask[iteration] & ~allowedStartLayers) {
        throw std::invalid_argument(std::format("{}.startLayerMask[{}]={} contains bits outside the {} detector layers",
                                                tc.getName(), iteration, tc.startLayerMask[iteration], tc.NLayers));
      }
    }

    plan.execution = {tc.maxMemory, tc.dropTFUponFailure};
    resolveSystematicErrors(plan.detector, tc);
    for (int i{0}; i < tc.NLayers; ++i) {
      plan.detector.AddTimeError[i] = tc.addTimeError[i];
    }
    plan.detector.ColBins = tc.LUTbinsU > 0 ? tc.LUTbinsU : plan.detector.ColBins;
    plan.detector.RowBins = tc.LUTbinsV > 0 ? tc.LUTbinsV : plan.detector.RowBins;

    for (auto& param : trackParams) {
      param.PassFlags.reset();
    }
    if (!trackParams.empty()) {
      trackParams[0].PassFlags.set(IterationStep::FirstPass, IterationStep::RebuildClusterLUT);
    }

    const float bFactor = std::abs(o2::base::Propagator::Instance()->getNominalBz()) / 5.0066791f;
    const float bFactorTracklets = bFactor < 0.01f ? 1.f : bFactor;

    for (auto& p : trackParams) {
      p.TrackletMinPt *= bFactorTracklets;
      for (int ilg = tc.MaxTrackLength; ilg >= tc.MinTrackLength; ilg--) {
        const int lslot = tc.MaxTrackLength - ilg;
        if (lslot < static_cast<int>(p.MinPt.size())) {
          p.MinPt[lslot] *= bFactor;
        }
      }

      p.UseDiamond = tc.useDiamond;
      p.RepeatRefitOut = tc.repeatRefitOut;
      p.ShiftRefToCluster = tc.shiftRefToCluster;
      p.CreateArtefactLabels = tc.createArtefactLabels;
      p.AllowSharingFirstCluster = tc.allowSharingFirstCluster;
      p.SharedClusterMaxDeltaPhi = tc.sharedClusterMaxDeltaPhi;
      p.SharedClusterMaxDeltaEta = tc.sharedClusterMaxDeltaEta;
      p.SharedClusterOppositeSign = tc.sharedClusterOppositeSign;

      const auto iter = &p - trackParams.data();
      if (iter < tracking::MaxIter) {
        p.MaxHoles = tc.maxHolesIter[iter];
      }

      if (tc.startLayerMask[iter] != 0) {
        p.StartLayerMask = tc.startLayerMask[iter];
      }

      p.MaxChi2ClusterAttachment = tc.maxChi2ClusterAttachment > 0 ? tc.maxChi2ClusterAttachment : p.MaxChi2ClusterAttachment;
      p.MaxChi2NDF = tc.maxChi2NDF > 0 ? tc.maxChi2NDF : p.MaxChi2NDF;
      p.PVres = tc.pvRes > 0 ? tc.pvRes : p.PVres;
      p.NSigmaCut *= tc.nSigmaCut > 0 ? tc.nSigmaCut : 1.f;
      p.TrackletMinPt *= tc.minPt > 0 ? tc.minPt : 1.f;
      for (int iD{0}; iD < 3; ++iD) {
        p.Diamond[iD] = tc.diamondPos[iD];
      }
    }
  };
  if (detId == detectors::DetID::ITS) {
    applyOverrides(ITSCommonCATrackerParam::Instance());
  } else {
    applyOverrides(MFTCATrackerParam::Instance());
    LOGP(info, "MFT CA {}: {} passes, material model nominal, index=PhiR phiBins={} radiusBins={} (radians, cm)",
         toString(mode), trackParams.size(), plan.detector.RowBins, plan.detector.ColBins);
    for (size_t iteration = 0; iteration < trackParams.size(); ++iteration) {
      const auto& p = trackParams[iteration];
      LOGP(info, "MFT CA pass {}: minTrackLength={} trackletMinPt={} maxChi2ClusterAttachment={} maxChi2NDF={} startLayerMask={}",
           iteration, p.MinTrackLength, p.TrackletMinPt, p.MaxChi2ClusterAttachment, p.MaxChi2NDF, p.StartLayerMask.value());
    }
  }

  return plan;
}

} // namespace TrackingMode
} // namespace o2::itsmft
