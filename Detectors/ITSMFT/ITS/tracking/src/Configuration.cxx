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
  std::string str = std::format("NZb:{} NPhB:{} NROFIt:{} DRof:{} PerVtx:{} DropFail:{} ClSh:{} TtklMinPt:{:.2f} MinCl:{}",
                                ZBins, PhiBins, nROFsPerIterations, DeltaROF, PerPrimaryVertexProcessing, DropTFUponFailure, ClusterSharing, TrackletMinPt, MinTrackLength);
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
  str += " SystErrY/Z:";
  for (size_t i = 0; i < SystErrorY2.size(); i++) {
    str += std::format("{:.2e}/{:.2e} ", SystErrorY2[i], SystErrorZ2[i]);
  }
  if (std::numeric_limits<size_t>::max() != MaxMemory) {
    str += std::format(" MemLimit {:.2f} GB", double(MaxMemory) / constants::GB);
  }
  return str;
}

std::string VertexingParameters::asString() const
{
  std::string str = std::format("NZb:{} NPhB:{} DRof:{} ClsCont:{} MaxTrkltCls:{} ZCut:{} PhCut:{}", ZBins, PhiBins, deltaRof, clusterContributorsCut, maxTrackletsPerCluster, zCut, phiCut);
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
      trackParams[3].DeltaROF = 0; // UPC specific setting
    }
    for (size_t ip = 0; ip < trackParams.size(); ip++) {
      auto& param = trackParams[ip];
      param.ZBins = 64;
      param.PhiBins = 32;
      param.CellsPerClusterLimit = 1.e3f;
      param.TrackletsPerClusterLimit = 1.e3f;
      // check if something was overridden via configurable params
      if (ip < tc.MaxIter) {
        if (tc.startLayerMask[ip] > 0) {
          trackParams[2].StartLayerMask = tc.startLayerMask[ip];
        }
        if (tc.minTrackLgtIter[ip] > 0) {
          param.MinTrackLength = tc.minTrackLgtIter[ip];
        }
        for (int ilg = tc.MaxTrackLength; ilg >= tc.MinTrackLength; ilg--) {
          int lslot0 = (tc.MaxTrackLength - ilg), lslot = lslot0 + ip * (tc.MaxTrackLength - tc.MinTrackLength + 1);
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
    trackParams[0].TrackletsPerClusterLimit = 100.;
    trackParams[0].CellsPerClusterLimit = 100.;
  } else {
    LOGP(fatal, "Unsupported ITS tracking mode {} ", toString(mode));
  }

  float bFactor = std::abs(o2::base::Propagator::Instance()->getNominalBz()) / 5.0066791;
  float bFactorTracklets = bFactor < 0.01 ? 1. : bFactor; // for tracklets only
  int nROFsPerIterations = tc.nROFsPerIterations > 0 ? tc.nROFsPerIterations : -1;

  if (tc.nOrbitsPerIterations > 0) {
    /// code to be used when the number of ROFs per orbit is known, this gets priority over the number of ROFs per iteration
  }

  // global parameters set for every iteration
  for (auto& p : trackParams) {
    // adjust pT settings to actual mag. field
    p.TrackletMinPt *= bFactorTracklets;
    for (int ilg = tc.MaxTrackLength; ilg >= tc.MinTrackLength; ilg--) {
      int lslot = tc.MaxTrackLength - ilg;
      p.MinPt[lslot] *= bFactor;
    }
    p.reseedIfShorter = tc.reseedIfShorter;
    p.shiftRefToCluster = tc.shiftRefToCluster;
    p.createArtefactLabels = tc.createArtefactLabels;

    p.PrintMemory = tc.printMemory;
    p.MaxMemory = tc.maxMemory;
    p.DropTFUponFailure = tc.dropTFUponFailure;
    p.SaveTimeBenchmarks = tc.saveTimeBenchmarks;
    p.FataliseUponFailure = tc.fataliseUponFailure;
    p.AllowSharingFirstCluster = tc.allowSharingFirstCluster;

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
    p.DeltaROF = tc.deltaRof;
    p.DoUPCIteration = tc.doUPCIteration;
    p.MaxChi2ClusterAttachment = tc.maxChi2ClusterAttachment > 0 ? tc.maxChi2ClusterAttachment : p.MaxChi2ClusterAttachment;
    p.MaxChi2NDF = tc.maxChi2NDF > 0 ? tc.maxChi2NDF : p.MaxChi2NDF;
    p.PhiBins = tc.LUTbinsPhi > 0 ? tc.LUTbinsPhi : p.PhiBins;
    p.ZBins = tc.LUTbinsZ > 0 ? tc.LUTbinsZ : p.ZBins;
    p.PVres = tc.pvRes > 0 ? tc.pvRes : p.PVres;
    p.NSigmaCut *= tc.nSigmaCut > 0 ? tc.nSigmaCut : 1.f;
    p.CellDeltaTanLambdaSigma *= tc.deltaTanLres > 0 ? tc.deltaTanLres : 1.f;
    p.TrackletMinPt *= tc.minPt > 0 ? tc.minPt : 1.f;
    p.nROFsPerIterations = nROFsPerIterations;
    p.PerPrimaryVertexProcessing = tc.perPrimaryVertexProcessing;
    for (int iD{0}; iD < 3; ++iD) {
      p.Diamond[iD] = tc.diamondPos[iD];
    }
    p.UseDiamond = tc.useDiamond;
    if (tc.useTrackFollower > 0) {
      p.UseTrackFollower = true;
      // Bit 0: Allow for mixing of top&bot extension --> implies Bits 1&2 set
      // Bit 1: Allow for top extension
      // Bit 2: Allow for bot extension
      p.UseTrackFollowerMix = ((tc.useTrackFollower & (1 << 0)) != 0);
      p.UseTrackFollowerTop = ((tc.useTrackFollower & (1 << 1)) != 0);
      p.UseTrackFollowerBot = ((tc.useTrackFollower & (1 << 2)) != 0);
      p.TrackFollowerNSigmaCutZ = tc.trackFollowerNSigmaZ;
      p.TrackFollowerNSigmaCutPhi = tc.trackFollowerNSigmaPhi;
    }
    if (tc.cellsPerClusterLimit >= 0) {
      p.CellsPerClusterLimit = tc.cellsPerClusterLimit;
    }
    if (tc.trackletsPerClusterLimit >= 0) {
      p.TrackletsPerClusterLimit = tc.trackletsPerClusterLimit;
    }
    if (tc.findShortTracks >= 0) {
      p.FindShortTracks = tc.findShortTracks;
    }
  }

  if (trackParams.size() > tc.nIterations) {
    trackParams.resize(tc.nIterations);
  }

  return trackParams;
}

std::vector<VertexingParameters> TrackingMode::getVertexingParameters(TrackingMode::Type mode)
{
  const auto& vc = o2::its::VertexerParamConfig::Instance();
  std::vector<VertexingParameters> vertParams;
  if (mode == TrackingMode::Async) {
    vertParams.resize(2); // The number of actual iterations will be set as a configKeyVal to allow for pp/PbPb choice
    vertParams[1].phiCut = 0.015f;
    vertParams[1].tanLambdaCut = 0.015f;
    vertParams[1].vertPerRofThreshold = 0;
    vertParams[1].deltaRof = 0;
  } else if (mode == TrackingMode::Sync) {
    vertParams.resize(1);
  } else if (mode == TrackingMode::Cosmics) {
    vertParams.resize(1);
  } else {
    LOGP(fatal, "Unsupported ITS vertexing mode {} ", toString(mode));
  }

  // global parameters set for every iteration
  for (auto& p : vertParams) {
    p.SaveTimeBenchmarks = vc.saveTimeBenchmarks;
    p.PrintMemory = vc.printMemory;
    p.MaxMemory = vc.maxMemory;
    p.DropTFUponFailure = vc.dropTFUponFailure;
    p.nIterations = vc.nIterations;
    p.deltaRof = vc.deltaRof;
    p.allowSingleContribClusters = vc.allowSingleContribClusters;
    p.trackletSigma = vc.trackletSigma;
    p.maxZPositionAllowed = vc.maxZPositionAllowed;
    p.clusterContributorsCut = vc.clusterContributorsCut;
    p.phiSpan = vc.phiSpan;
    p.nThreads = vc.nThreads;
    p.ZBins = vc.ZBins;
    p.PhiBins = vc.PhiBins;

    p.useTruthSeeding = vc.useTruthSeeding;
    p.outputContLabels = vc.outputContLabels;
  }
  // set for now outside to not disturb status quo
  vertParams[0].vertNsigmaCut = vc.vertNsigmaCut;
  vertParams[0].vertRadiusSigma = vc.vertRadiusSigma;
  vertParams[0].maxTrackletsPerCluster = vc.maxTrackletsPerCluster;
  vertParams[0].lowMultBeamDistCut = vc.lowMultBeamDistCut;
  vertParams[0].zCut = vc.zCut;
  vertParams[0].phiCut = vc.phiCut;
  vertParams[0].pairCut = vc.pairCut;
  vertParams[0].clusterCut = vc.clusterCut;
  vertParams[0].histPairCut = vc.histPairCut;
  vertParams[0].tanLambdaCut = vc.tanLambdaCut;

  return vertParams;
}
