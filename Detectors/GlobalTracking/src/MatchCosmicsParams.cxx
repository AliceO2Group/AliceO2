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

/// \file MatchCosmicsParams.h
/// \brief Configurable params for cosmics matching
/// \author ruben.shahoyan@cern.ch

#include "GlobalTracking/MatchCosmicsParams.h"
#include "Framework/Logger.h"
#include <map>

O2ParamImpl(o2::globaltracking::MatchCosmicsParams);

std::string o2::globaltracking::getMatchCosmicsPreset(const std::string& name)
{
  // physics-v1, tuned on PbPb 2025 (LHC25an 567939, 19 kHz) for fakes, on the cosmics run 562658 and on a cosmic MC with PbPb-like
  // distortions for the efficiency (TPC-only legs):
  // - seed cuts against collision tracks: pT, transverse DCA to the beam line in cm and in sigma, TPC clusters
  // - realistic systematic errors for the leg comparison (true pairs had y / snp / q/pt pulls 2-4x too wide), so that the crude pair
  //   chi2 cut is meaningful; per-parameter windows open except tgl (3 sigma: the main discriminant against random pairs)
  // - loose cut on the chi2 of the refitted legs; z test of TPC-only legs at a common time and same-half veto switched on
  // - TPC-only legs on opposite sides (time from z continuity, ~94 % of the PbPb candidates were random pairs): both legs and the
  //   refitted cosmic above 2 GeV (offline: PbPb candidates / 10.5, cosmic MC efficiency 81.8 -> 80.6 %)
  // - TOF flight pair as selection criterion (needs TOF clusters, added to the inputs): pairs confirmed by a top / bottom TOF hit pair
  //   with the muon's flight time win the selection and are refitted at the TOF time (568041: fewer cosmics, more of them TOF-tagged);
  //   the common-time refit of same-side legs (refitSameSideAtCommonTime) is not used: in PbPb it added mostly collision-track pairs
  static const std::map<std::string, std::string> presets{
    {"physics-v1",
     "cosmicsMatch.minSeedPt=1;cosmicsMatch.minSeedDCAxy=3;cosmicsMatch.minSeedDCAxyNSigma=10;cosmicsMatch.minSeedNClTPC=30;"
     "cosmicsMatch.crudeChi2Cut=50;cosmicsMatch.systSigma2[0]=0.25;cosmicsMatch.systSigma2[2]=4e-4;cosmicsMatch.systSigma2[4]=2.5e-3;"
     "cosmicsMatch.crudeNSigma2Cut[0]=144;cosmicsMatch.crudeNSigma2Cut[2]=144;cosmicsMatch.crudeNSigma2Cut[3]=9;cosmicsMatch.crudeNSigma2Cut[4]=144;"
     "cosmicsMatch.maxChi2Match=1000;cosmicsMatch.constrainTPCOnlyZ=true;cosmicsMatch.vetoSameHalf=true;cosmicsMatch.minPtOppositeSides=2;"
     "cosmicsMatch.tofFlightSelection=true"}};
  auto it = presets.find(name);
  if (it == presets.end()) {
    std::string known;
    for (const auto& [k, v] : presets) {
      known += " " + k;
    }
    LOGP(fatal, "Unknown cosmics matching preset {}, known:{}", name, known);
    return {};
  }
  return it->second;
}
