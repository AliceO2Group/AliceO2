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
#include <limits>
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
} // namespace

namespace o2::itsmft
{

std::string TrackingParameters::asString() const
{
  std::string str = std::format("NColB:{} NRowB:{} PerVtx:{} DropFail:{} TtklMinPt:{:.2f} MinCl:{}", ColBins, RowBins, PerPrimaryVertexProcessing, DropTFUponFailure, TrackletMinPt, MinTrackLength);
  auto isSet = [](auto e) { return e >= 0; };
  auto isAnySet = [&isSet](auto v) { return !v.empty() && std::any_of(v.begin(), v.end(), isSet); };
  bool first = true;
  for (int il = NLayers; il >= MinTrackLength; il--) {
    int slot = NLayers - il;
    if (slot < (int)MinPt.size() && MinPt[slot] > 0) {
      if (first) {
        first = false;
        str += " MinPt: ";
      }
      str += std::format("L{}:{:.2f} ", il, MinPt[slot]);
    }
  }
  if (isAnySet(SystError2Row) || isAnySet(SystError2Col)) {
    str += " SystErrRow/Col:";
    for (size_t i = 0; i < SystError2Row.size(); i++) {
      str += std::format("{:.2e}/{:.2e} ", SystError2Row[i], SystError2Col[i]);
    }
  }
  if (isAnySet(AddTimeError)) {
    str += " AddTimeError:";
    for (unsigned int i : AddTimeError) {
      str += std::format("{} ", i);
    }
  }
  if (SharedMaxClusters) {
    str += std::format(" ShaMaxCls:{} ", SharedMaxClusters);
  }
  if (AllowSharingFirstCluster) {
    str += std::format(" ShaClsDPhi:{} ShaClsDEta:{} ShaClsSign:{}", SharedClusterMaxDeltaPhi, SharedClusterMaxDeltaEta, SharedClusterOppositeSign);
  }
  if (MaxHoles) {
    str += std::format(" MaxHoles:{}", MaxHoles);
  }
  if (!InactiveLayerMask.empty()) {
    str += std::format(" InactiveMask:{}", InactiveLayerMask.asString());
  }
  if (!SeedingLayers.empty()) {
    str += std::format(" SeedingLayers:{}", SeedingLayers.asString());
  }
  if (std::numeric_limits<size_t>::max() != MaxMemory) {
    str += std::format(" MemLimit {:.2f} GB", double(MaxMemory) / (1024.f * 1024.f * 1024.f));
  }
  return str;
}

std::string VertexingParameters::asString() const
{
  std::string str = std::format("NColB:{} NRowB:{} MinVtxCont:{} SupLowMultDebris:{} MaxTrkltCls:{} ZCut:{} PhCut:{} PairCut:{} ClCut:{} SeedRad:{}x{}",
                                ColBins, RowBins, clusterContributorsCut, suppressLowMultDebris, maxTrackletsPerCluster, zCut, phiCut, pairCut, clusterCut, seedMemberRadiusTime, seedMemberRadiusZ);
  if (std::numeric_limits<size_t>::max() != MaxMemory) {
    str += std::format(" MemLimit {:.2f} GB", double(MaxMemory) / (1024.f * 1024.f * 1024.f));
  }
  return str;
}

void resetDetectorDefaults(TrackingParameters& p, detectors::DetID::ID detId)
{
  if (detId == detectors::DetID::ITS) {
    p = TrackingParameters{};
    p.MinPt.assign(tracking::ITSNLayers - tracking::kCAMinTrackLength + 1, 0.f);
    return;
  }

  if (detId == detectors::DetID::MFT) {
    namespace mftc = o2::mft::constants;
    namespace mft = mftc::mft;
    constexpr int nLayers = o2::mft::constants::mft::LayersNumber;

    p = TrackingParameters{};
    p.NLayers = nLayers;
    p.LayerZ.clear();
    p.LayerZ.reserve(nLayers);
    for (float z : mft::LayerZCoordinate()) {
      p.LayerZ.push_back(std::abs(z));
    }
    p.LayerColHalfExtent.assign(mftc::index_table::RMax.begin(), mftc::index_table::RMax.end());
    p.IndexRowMin = -20.f;
    p.IndexRowMax = 20.f;
    p.LayerRadii.resize(nLayers);
    for (int i{0}; i < nLayers; ++i) {
      p.LayerRadii[i] = 0.5f * (mftc::index_table::RMin[i] + mftc::index_table::RMax[i]);
    }
    p.LayerResolution.assign(nLayers, mft::Resolution);
    p.SystError2Row.assign(nLayers, 0.f);
    p.SystError2Col.assign(nLayers, 0.f);
    p.AddTimeError.assign(nLayers, 0u);
    p.ColBins = 64;
    p.RowBins = 128;
    p.UseDiamond = true;
    p.PerPrimaryVertexProcessing = false;
    p.StartLayerMask = (1u << nLayers) - 1u;
    p.MinPt.assign(TrackerParamConfig<detectors::DetID::MFT>::MaxTrackLength - TrackerParamConfig<detectors::DetID::MFT>::MinTrackLength + 1, 0.f);
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

void validateCommonCAOptions(detectors::DetID::ID detId)
{
  const auto reject = [](bool unsupported, std::string_view field, std::string_view supported) {
    if (unsupported) {
      throw std::invalid_argument(std::string(field) + " has no implementing common-CA consumer; use " + std::string(supported));
    }
  };
  if (detId == detectors::DetID::ITS) {
    const auto& tc = ITSCommonCATrackerParam::Instance();
    reject(tc.printMemory, "ITSCommonCATrackerParam.printMemory", "false");
    reject(tc.saveTimeBenchmarks, "ITSCommonCATrackerParam.saveTimeBenchmarks", "false");
    return;
  }
  if (detId != detectors::DetID::MFT) {
    throw std::invalid_argument("Unsupported detector in common-CA option validation");
  }
  const auto& tc = TrackerParamConfig<detectors::DetID::MFT>::Instance();
  reject(tc.printMemory, "MFTCATrackerParam.printMemory", "false");
  reject(tc.saveTimeBenchmarks, "MFTCATrackerParam.saveTimeBenchmarks", "false");
  reject(!tc.fataliseUponFailure, "MFTCATrackerParam.fataliseUponFailure", "true; dropTFUponFailure controls recoverable drops");
  reject(tc.deltaTanLres != -1.f, "MFTCATrackerParam.deltaTanLres", "-1");
  reject(tc.doUPCIteration, "MFTCATrackerParam.doUPCIteration", "false");
  reject(tc.overrideBeamEstimation, "MFTCATrackerParam.overrideBeamEstimation", "false");
  if (!tc.useDiamond || tc.perPrimaryVertexProcessing) {
    throw std::invalid_argument("MFT common CA requires MFTCATrackerParam.useDiamond=true and MFTCATrackerParam.perPrimaryVertexProcessing=false");
  }
}

TrackingPlan getTrackingPlan(detectors::DetID::ID detId, Type mode)
{
  validateCommonCAOptions(detId);
  TrackingParameters defaults;
  resetDetectorDefaults(defaults, detId);
  TrackingPlan plan{std::move(static_cast<DetectorParameters&>(defaults)), {}, {}};
  auto& trackParams = plan.iterations;
  if (detId == detectors::DetID::ITS) {
    const auto& tc = ITSCommonCATrackerParam::Instance();
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
    plan.execution = {tc.maxMemory, tc.dropTFUponFailure};
    for (auto& p : trackParams) {
      p.PassFlags.reset();
    }
    trackParams.front().PassFlags.set(IterationStep::FirstPass, IterationStep::RebuildClusterLUT);

    const float bFactor = std::abs(o2::base::Propagator::Instance()->getNominalBz()) / 5.0066791f;
    const float bFactorTracklets = bFactor < 0.01f ? 1.f : bFactor;
    for (auto& p : trackParams) {
      p.TrackletMinPt *= bFactorTracklets;
      for (auto& minPt : p.MinPt) {
        minPt *= bFactor;
      }
      p.UseDiamond = tc.useDiamond;
      for (int iD = 0; iD < 3; ++iD) {
        p.Diamond[iD] = tc.diamondPos[iD];
      }
      p.PVres = tc.pvRes > 0 ? tc.pvRes : p.PVres;
    }
    return plan;
  }
  if (detId != detectors::DetID::MFT) {
    LOGP(fatal, "Unsupported detector id {} in getTrackingPlan", static_cast<int>(detId));
  }

  const auto& tc = TrackerParamConfig<detectors::DetID::MFT>::Instance();

  if (mode == Off) {
    return plan;
  }
  if (mode == Unset) {
    LOGP(fatal, "CA tracking mode is unset; set --tracking-mode or {}.trackingMode", TrackerParamConfig<detectors::DetID::MFT>::getParamName());
  }

  if (mode != Async) {
    if (std::any_of(std::begin(tc.minTrackLgtIter), std::end(tc.minTrackLgtIter), [](int value) { return value > 0; })) {
      throw std::invalid_argument("MFTCATrackerParam.minTrackLgtIter overrides are implemented only for async mode");
    }
    if (std::any_of(std::begin(tc.minPtIterLgt), std::end(tc.minPtIterLgt), [](float value) { return value > 0.f; })) {
      throw std::invalid_argument("MFTCATrackerParam.minPtIterLgt overrides are implemented only for async mode");
    }
  }

  if (mode == Async) {
    trackParams.assign(3, defaults);

    trackParams[1].TrackletMinPt = 0.15f;
    trackParams[2].TrackletMinPt = 0.08f;

    trackParams[0].MinPt[0] = 1.f / 12.f; // 10 clusters
    trackParams[1].MinPt[0] = 1.f / 12.f;

    trackParams[2].MinTrackLength = TrackerParamConfig<detectors::DetID::MFT>::MinTrackLength;
    trackParams[2].MinPt[0] = 1.f / 12.f; // 10 clusters
    trackParams[2].MinPt[1] = 1.f / 8.f;  // 9 clusters
    trackParams[2].MinPt[2] = 1.f / 5.f;  // 8 clusters
    trackParams[2].MinPt[3] = 1.f / 3.f;  // 7 clusters
    trackParams[2].MinPt[4] = 1.f / 2.f;  // 6 clusters
    trackParams[2].MinPt[5] = 1.f / 1.f;  // 5 clusters

    for (int ip = 0; ip < static_cast<int>(trackParams.size()); ip++) {
      auto& param = trackParams[ip];
      if (ip < o2::its::constants::MaxIter) {
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
  } else if (mode == Sync) {
    trackParams.assign(1, defaults);
    trackParams[0].MinTrackLength = TrackerParamConfig<detectors::DetID::MFT>::MinTrackLength;
  } else if (mode == Cosmics) {
    trackParams.assign(1, defaults);
    trackParams[0].MinTrackLength = TrackerParamConfig<detectors::DetID::MFT>::MinTrackLength;
    plan.detector.ColBins = 32;
    plan.detector.RowBins = 64;
    trackParams[0].PVres = 1.e5f;
    trackParams[0].MaxChi2ClusterAttachment = 60.f;
    trackParams[0].MaxChi2NDF = 40.f;
  } else {
    LOGP(fatal, "Unsupported CA tracking mode {}", toString(mode));
  }

  if (tc.nIterations != -1 && (tc.nIterations <= 0 || static_cast<size_t>(tc.nIterations) > trackParams.size())) {
    throw std::invalid_argument(std::format("MFTCATrackerParam.nIterations={} is invalid for {}: use -1 or 1..{}",
                                            tc.nIterations, toString(mode), trackParams.size()));
  }
  if (tc.nIterations > 0) {
    trackParams.resize(tc.nIterations);
  }
  if (tc.materialModel != "nominal") {
    throw std::invalid_argument("MFTCATrackerParam.materialModel='" + tc.materialModel + "' is unsupported; use nominal");
  }
  if (tc.useMatCorrTGeo) {
    throw std::invalid_argument("MFTCATrackerParam.useMatCorrTGeo requests unsupported TGeo material; use materialModel=nominal");
  }
  if (!tc.useFastMaterial) {
    throw std::invalid_argument("MFTCATrackerParam.useFastMaterial=false requests unsupported LUT material; use materialModel=nominal and useFastMaterial=true");
  }
  constexpr uint32_t allowedStartLayers = (uint32_t{1} << tracking::MFTNLayers) - 1;
  for (int iteration = 0; iteration < tracking::MaxIter; ++iteration) {
    if (tc.startLayerMask[iteration] & ~allowedStartLayers) {
      throw std::invalid_argument(std::format("MFTCATrackerParam.startLayerMask[{}]={} contains bits outside the {} MFT layers",
                                              iteration, tc.startLayerMask[iteration], tracking::MFTNLayers));
    }
  }

  plan.execution = {tc.maxMemory, tc.dropTFUponFailure};
  for (int i{0}; i < TrackerParamConfig<detectors::DetID::MFT>::getNLayers(); ++i) {
    plan.detector.SystError2Row[i] = tc.sysErr2Row[i] > 0 ? tc.sysErr2Row[i] : plan.detector.SystError2Row[i];
    plan.detector.SystError2Col[i] = tc.sysErr2Col[i] > 0 ? tc.sysErr2Col[i] : plan.detector.SystError2Col[i];
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

    p.ReseedIfShorter = tc.reseedIfShorter;
    p.RepeatRefitOut = tc.repeatRefitOut;
    p.ShiftRefToCluster = tc.shiftRefToCluster;
    p.CreateArtefactLabels = tc.createArtefactLabels;
    p.AllowSharingFirstCluster = tc.allowSharingFirstCluster;
    p.SharedClusterMaxDeltaPhi = tc.sharedClusterMaxDeltaPhi;
    p.SharedClusterMaxDeltaEta = tc.sharedClusterMaxDeltaEta;
    p.SharedClusterOppositeSign = tc.sharedClusterOppositeSign;
    p.PerPrimaryVertexProcessing = tc.perPrimaryVertexProcessing;

    const auto iter = &p - trackParams.data();
    if (iter < o2::its::constants::MaxIter) {
      p.MaxHoles = tc.maxHolesIter[iter];
    }

    // The legacy NONE tag disables external providers, not nominal material.
    p.CorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE;
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
    p.UseDiamond = tc.useDiamond;
  }

  LOGP(info, "MFT CA {}: {} passes, material model nominal, index=PhiR phiBins={} radiusBins={} (radians, cm)",
       toString(mode), trackParams.size(), plan.detector.RowBins, plan.detector.ColBins);
  if (tc.reseedIfShorter != 0) {
    LOGP(warning, "MFTCATrackerParam.reseedIfShorter={} is reserved and has no effect in the current common refit", tc.reseedIfShorter);
  }
  for (size_t iteration = 0; iteration < trackParams.size(); ++iteration) {
    const auto& p = trackParams[iteration];
    LOGP(info, "MFT CA pass {}: minTrackLength={} trackletMinPt={} maxChi2ClusterAttachment={} maxChi2NDF={} startLayerMask={}",
         iteration, p.MinTrackLength, p.TrackletMinPt, p.MaxChi2ClusterAttachment, p.MaxChi2NDF, p.StartLayerMask.value());
  }

  return plan;
}

} // namespace TrackingMode
} // namespace o2::itsmft
