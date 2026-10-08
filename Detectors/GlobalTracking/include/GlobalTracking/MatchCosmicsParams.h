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

/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_MATCHCOSMICS_PARAMS_H
#define ALICEO2_MATCHCOSMICS_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include <string>

namespace o2
{
namespace globaltracking
{

struct MatchCosmicsParams : public o2::conf::ConfigurableParamHelper<MatchCosmicsParams> {
  float dcaCutChi2[o2::dataformats::GlobalTrackID::NSources] = {};           // optional (>0) chi2 cut on track DCA to any of its compatible vertices
  float systSigma2[o2::track::kNParams] = {0.01f, 0.01f, 1e-4f, 1e-4f, 0.f}; // extra error to be added at legs comparison
  float crudeNSigma2Cut[o2::track::kNParams] = {49.f, 49.f, 49.f, 49.f, 49.f};
  float crudeChi2Cut = 999.f;
  float maxChi2Match = -1.f;      // reject cosmics whose top/bottom refitted legs disagree by more than this chi2 (< 0: no cut)
  float minPtOppositeSides = 0.f; // TPC-only legs on opposite TPC sides: reject cosmics with the pT of either leg or of the refitted cosmic below this (scaled with field; 0: no cut)
  float timeToleranceMUS = 0.f;
  float maxStep = 10.f;
  float maxSnp = 0.99f;
  float minSeedPt = 0.10f;     // use only tracks above this pT (scaled with field)
  int minSeedNClTPC = 0;       // use only TPC-only seeds with at least this number of clusters (0: no cut)
  float minSeedDCAxy = 0.f;    // use only tracks with |DCA_xy| to the beam line >= this [cm] (0: no cut; rejects collision tracks in physics data)
  float minSeedDCAxyNSigma = 0.f;         // use only tracks with |DCA_xy| >= this * sigma(DCA_xy) (0: no cut; poorly measured collision tracks)
  bool constrainTPCOnlyZ = false;         // TPC-only legs: test z at a common time (same side, or a leg with known time), else require the time implied by z continuity in both brackets
  bool vetoSameHalf = false;              // reject pairs whose two legs lie on the same side of the closest approach (two pieces of one leg)
  bool refitSameSideAtCommonTime = false; // TPC-only legs on the same side: compare them refitted at the centre of their brackets' overlap instead of at their own time0s
  bool tofFlightSelection = false;        // needs TOF clusters: accepted pairs of TPC-only legs pointing to a top / bottom TOF hit pair with the muon's flight time win the selection, refit at that time (if that fails, at their time without TOF)
  float tofRoad = 5.f;                    // half-width [cm] in y and z of the road at the TOF around the outward continuation of a TPC-only leg
  float tofFlightTolerance = 2.f;         // max. deviation [ns] of the top / bottom TOF time difference from the flight time along the helix
  float tofTimeError = 0.1f;              // error [mus] of the TOF time of a confirmed cosmic, for its refit and time window (covers TPC vs TOF offsets)
  float nSigmaTError = 4.f;    // number of sigmas on track time error for matching (except for TPC which provides an interval)
  float tpcExtraZError2 = 1.f; // extra error^2 on the TPC-only track Z coordinate
  float fiducialRIP = 1.0f;    // consider track having |Y@x=0|< this as passing DCA cut (if requested)
  float fiducialZIP = 20.f;    // consider track having |Z@x=0|< this as passing DCA cut (if requested)
  bool allowTPCOnly = true;
  bool discardPVContributors = true; // used only if the global option --use-pv-info is requested
  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT;

  O2ParamDef(MatchCosmicsParams, "cosmicsMatch");
};

/// key=value string (configKeyValues syntax) of a named set of MatchCosmicsParams settings; the cosmics-match workflow applies it before
/// --configKeyValues, so single keys can still be overridden. Unknown names are fatal.
/// "physics-v1": cosmics in collision data (seed cuts against collision tracks, realistic systematic errors for the pair chi2, tgl window,
/// pT cut on pairs of TPC-only legs on opposite TPC sides)
std::string getMatchCosmicsPreset(const std::string& name);

} // namespace globaltracking
} // end namespace o2

#endif
