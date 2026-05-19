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
#include <format>
#include <limits>
#include <string_view>
#include <vector>

#include "Framework/Logger.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/TrackingConfigParam.h"

using namespace o2::its;

std::string TrackingParameters::asString() const
{
  std::string str = std::format("NZb:{} NPhB:{} PerVtx:{} DropFail:{} ClSh:{} TtklMinPt:{:.2f} MinCl:{} MaxHoles:{} HoleMask:{:#x}",
                                ZBins, PhiBins, PerPrimaryVertexProcessing, DropTFUponFailure, ClusterSharing, TrackletMinPt, MinTrackLength, MaxHoles, HoleLayerMask);
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
  if (!SystErrorY2.empty() || !SystErrorZ2.empty()) {
    str += " SystErrY/Z:";
    for (size_t i = 0; i < SystErrorY2.size(); i++) {
      str += std::format("{:.2e}/{:.2e} ", SystErrorY2[i], SystErrorZ2[i]);
    }
  }
  if (!AddTimeError.empty()) {
    str += " AddTimeError:";
    for (unsigned int i : AddTimeError) {
      str += std::format("{} ", i);
    }
  }
  if (std::numeric_limits<size_t>::max() != MaxMemory) {
    str += std::format(" MemLimit {:.2f} GB", double(MaxMemory) / constants::GB);
  }
  return str;
}

std::string VertexingParameters::asString() const
{
  std::string str = std::format("NZb:{} NPhB:{} MinVtxCont:{} SupLowMultDebris:{} MaxTrkltCls:{} ZCut:{} PhCut:{} PairCut:{} ClCut:{} SeedRad:{}x{}",
                                ZBins, PhiBins, clusterContributorsCut, suppressLowMultDebris, maxTrackletsPerCluster, zCut, phiCut, pairCut, clusterCut, seedMemberRadiusTime, seedMemberRadiusZ);
  if (std::numeric_limits<size_t>::max() != MaxMemory) {
    str += std::format(" MemLimit {:.2f} GB", double(MaxMemory) / constants::GB);
  }
  return str;
}

namespace
{
constexpr bool iequals(std::string_view a, std::string_view b)
{
  return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                    [](char x, char y) { return std::tolower(x) == std::tolower(y); });
}
} // namespace

TrackingMode::Type TrackingMode::fromString(std::string_view str)
{
  constexpr std::array smodes = {
    std::pair{"sync", Sync},
    std::pair{"async", Async},
    std::pair{"cosmics", Cosmics},
    std::pair{"unset", Unset},
    std::pair{"off", Off}};

  auto it = std::find_if(smodes.begin(), smodes.end(), [&str](const auto& pair) {
    return iequals(str, pair.first);
  });
  if (it == smodes.end()) {
    LOGP(fatal, "Unrecognized tracking mode '{}'", str);
  }
  return it->second;
}

std::string TrackingMode::toString(TrackingMode::Type mode)
{
  if (mode == TrackingMode::Sync) {
    return "sync";
  } else if (mode == TrackingMode::Async) {
    return "async";
  } else if (mode == TrackingMode::Cosmics) {
    return "cosmics";
  } else if (mode == TrackingMode::Unset) {
    return "unset";
  } else if (mode == TrackingMode::Off) {
    return "off";
  }
  LOGP(fatal, "Unrecognized tracking mode '{}'", (int)mode);
  return ""; // not reachable
}

std::vector<TrackingParameters> TrackingMode::getTrackingParameters(TrackingMode::Type mode)
{
  const auto& tc = o2::its::TrackerParamConfig::Instance();
  std::vector<TrackingParameters> trackParams;

  if (mode == TrackingMode::Async) {
    trackParams.resize(tc.doUPCIteration ? 4 : 3);
    trackParams[1].TrackletMinPt = 0.2f;
    trackParams[1].CellDeltaTanLambdaSigma *= 2.;
    trackParams[2].TrackletMinPt = 0.1f;
    trackParams[2].CellDeltaTanLambdaSigma *= 4.;

    trackParams[0].MinPt[0] = 1.f / 12; // 7cl
    trackParams[1].MinPt[0] = 1.f / 12; // 7cl

    trackParams[2].MinTrackLength = 4;
    trackParams[2].MinPt[0] = 1.f / 12; // 7cl
    trackParams[2].MinPt[1] = 1.f / 5;  // 6cl
    trackParams[2].MinPt[2] = 1.f / 1;  // 5cl
    trackParams[2].MinPt[3] = 1.f / 6;  // 4cl

    trackParams[2].StartLayerMask = (1 << 6) + (1 << 3);
    if (tc.doUPCIteration) {
      trackParams[3].MinTrackLength = 4;
      trackParams[3].TrackletMinPt = 0.1f;
      trackParams[3].CellDeltaTanLambdaSigma *= 4.;
    }
    for (int ip = 0; ip < (int)trackParams.size(); ip++) {
      auto& param = trackParams[ip];
      param.ZBins = 64;
      param.PhiBins = 32;
      // check if something was overridden via configurable params
      if (ip < constants::MaxIter) {
        if (tc.startLayerMask[ip] > 0) {
          param.StartLayerMask = tc.startLayerMask[ip];
        }
        if (tc.minTrackLgtIter[ip] > 0) {
          param.MinTrackLength = tc.minTrackLgtIter[ip];
        }
        for (int ilg = tc.MaxTrackLength; ilg >= tc.MinTrackLength; ilg--) {
          int lslot0 = (tc.MaxTrackLength - ilg), lslot = lslot0 + (ip * (tc.MaxTrackLength - tc.MinTrackLength + 1));
          if (tc.minPtIterLgt[lslot] > 0.) {
            param.MinPt[lslot0] = tc.minPtIterLgt[lslot];
          }
        }
      }
    }
  } else if (mode == TrackingMode::Sync) {
    trackParams.resize(1);
    trackParams[0].ZBins = 64;
    trackParams[0].PhiBins = 32;
    trackParams[0].MinTrackLength = 4;
  } else if (mode == TrackingMode::Cosmics) {
    trackParams.resize(1);
    trackParams[0].MinTrackLength = 4;
    trackParams[0].CellDeltaTanLambdaSigma *= 10;
    trackParams[0].PhiBins = 4;
    trackParams[0].ZBins = 16;
    trackParams[0].PVres = 1.e5f;
    trackParams[0].MaxChi2ClusterAttachment = 60.;
    trackParams[0].MaxChi2NDF = 40.;
  } else {
    LOGP(fatal, "Unsupported ITS tracking mode {} ", toString(mode));
  }

  for (auto& param : trackParams) {
    param.PassFlags.reset();
  }
  trackParams[0].PassFlags.set(IterationStep::FirstPass, IterationStep::RebuildClusterLUT);
  if (trackParams.size() > 3 && tc.doUPCIteration) {
    trackParams[3].PassFlags.set(IterationStep::UseUPCMask, IterationStep::RebuildClusterLUT, IterationStep::SelectUPCVertices);
  }

  float bFactor = std::abs(o2::base::Propagator::Instance()->getNominalBz()) / 5.0066791f;
  float bFactorTracklets = bFactor < 0.01f ? 1.f : bFactor; // for tracklets only

  // global parameters set for every iteration
  for (auto& p : trackParams) {
    // adjust pT settings to actual mag. field
    p.TrackletMinPt *= bFactorTracklets;
    for (int ilg = tc.MaxTrackLength; ilg >= tc.MinTrackLength; ilg--) {
      int lslot = tc.MaxTrackLength - ilg;
      p.MinPt[lslot] *= bFactor;
    }
    p.ReseedIfShorter = tc.reseedIfShorter;
    p.RepeatRefitOut = tc.repeatRefitOut;
    p.ShiftRefToCluster = tc.shiftRefToCluster;
    p.CreateArtefactLabels = tc.createArtefactLabels;

    p.PrintMemory = tc.printMemory;
    p.MaxMemory = tc.maxMemory;
    p.DropTFUponFailure = tc.dropTFUponFailure;
    p.SaveTimeBenchmarks = tc.saveTimeBenchmarks;
    p.FataliseUponFailure = tc.fataliseUponFailure;
    p.AllowSharingFirstCluster = tc.allowSharingFirstCluster;
    const auto iter = &p - trackParams.data();
    if (iter < constants::MaxIter) {
      p.MaxHoles = tc.maxHolesIter[iter];
      p.HoleLayerMask = tc.holeLayerMaskIter[iter];
    }

    if (tc.useMatCorrTGeo) {
      p.CorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrTGeo;
    } else if (tc.useFastMaterial) {
      p.CorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE;
    } else {
      p.CorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT;
    }

    if (p.NLayers == 7) {
      for (int i{0}; i < 7; ++i) {
        p.SystErrorY2[i] = tc.sysErrY2[i] > 0 ? tc.sysErrY2[i] : p.SystErrorY2[i];
        p.SystErrorZ2[i] = tc.sysErrZ2[i] > 0 ? tc.sysErrZ2[i] : p.SystErrorZ2[i];
      }
    }
    for (int i{0}; i < 7; ++i) {
      p.AddTimeError[i] = tc.addTimeError[i];
    }
    p.DoUPCIteration = tc.doUPCIteration;
    p.MaxChi2ClusterAttachment = tc.maxChi2ClusterAttachment > 0 ? tc.maxChi2ClusterAttachment : p.MaxChi2ClusterAttachment;
    p.MaxChi2NDF = tc.maxChi2NDF > 0 ? tc.maxChi2NDF : p.MaxChi2NDF;
    p.PhiBins = tc.LUTbinsPhi > 0 ? tc.LUTbinsPhi : p.PhiBins;
    p.ZBins = tc.LUTbinsZ > 0 ? tc.LUTbinsZ : p.ZBins;
    p.PVres = tc.pvRes > 0 ? tc.pvRes : p.PVres;
    p.NSigmaCut *= tc.nSigmaCut > 0 ? tc.nSigmaCut : 1.f;
    p.CellDeltaTanLambdaSigma *= tc.deltaTanLres > 0 ? tc.deltaTanLres : 1.f;
    p.TrackletMinPt *= tc.minPt > 0 ? tc.minPt : 1.f;
    p.PerPrimaryVertexProcessing = tc.perPrimaryVertexProcessing;
    for (int iD{0}; iD < 3; ++iD) {
      p.Diamond[iD] = tc.diamondPos[iD];
    }
    p.UseDiamond = tc.useDiamond;
  }

  if (trackParams.size() > tc.nIterations) {
    trackParams.resize(tc.nIterations);
  }

  return trackParams;
}

std::vector<VertexingParameters> TrackingMode::getVertexingParameters(TrackingMode::Type mode)
{
  const auto& vc = o2::its::VertexerParamConfig::Instance();
  std::vector<VertexingParameters> vertParams(2); // The number of actual iterations will be set as a configKeyVal to allow for pp/PbPb choice
  for (auto& param : vertParams) {
    param.PassFlags.reset();
  }
  vertParams[0].PassFlags.set(IterationStep::FirstPass, IterationStep::ResetVertices);
  vertParams[1].PassFlags.set(IterationStep::SkipROFsAboveThreshold, IterationStep::MarkVerticesAsUPC);

  // global parameters set for every iteration
  for (auto& p : vertParams) {
    p.vertPerRofThreshold = vc.vertPerRofThreshold;
    p.SaveTimeBenchmarks = vc.saveTimeBenchmarks;
    p.PrintMemory = vc.printMemory;
    p.MaxMemory = vc.maxMemory;
    p.DropTFUponFailure = vc.dropTFUponFailure;
    p.NSigmaCut = vc.nSigmaCut;
    p.maxZPositionAllowed = vc.maxZPositionAllowed;
    p.clusterContributorsCut = vc.clusterContributorsCut;
    p.suppressLowMultDebris = vc.suppressLowMultDebris;
    p.seedMemberRadiusTime = vc.seedMemberRadiusTime;
    p.seedMemberRadiusZ = vc.seedMemberRadiusZ;
    p.phiSpan = vc.phiSpan;
    p.nThreads = vc.nThreads;
    p.ZBins = vc.ZBins;
    p.PhiBins = vc.PhiBins;
    p.useTruthSeeding = vc.useTruthSeeding;
    p.maxTrackletsPerCluster = vc.maxTrackletsPerCluster;
    p.zCut = vc.zCut;
    p.phiCut = vc.phiCut;
    p.pairCut = vc.pairCut;
    p.clusterCut = vc.clusterCut;
    p.coarseZWindow = vc.coarseZWindow;
    p.seedDedupZCut = vc.seedDedupZCut;
    p.refitDedupZCut = vc.refitDedupZCut;
    p.duplicateZCut = vc.duplicateZCut;
    p.finalSelectionZCut = vc.finalSelectionZCut;
    p.duplicateDistance2Cut = vc.duplicateDistance2Cut;
    p.tanLambdaCut = vc.tanLambdaCut;
  }

  if (mode == TrackingMode::Async) {
    // relax for UPC iteration
    vertParams[1].phiCut = 0.015f;
    vertParams[1].tanLambdaCut = 0.015f;
    vertParams[1].maxTrackletsPerCluster = 2000;
  } else if (mode == TrackingMode::Sync || TrackingMode::Cosmics) {
    vertParams.resize(1);
  } else {
    LOGP(fatal, "Unsupported ITS vertexing mode {} ", toString(mode));
  }

  if (vertParams.size() > vc.nIterations) {
    vertParams.resize(vc.nIterations);
  }

  return vertParams;
}
