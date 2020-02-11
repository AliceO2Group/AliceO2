// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_MATCHITSTPC_PARAMS_H
#define ALICEO2_MATCHITSTPC_PARAMS_H

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace globaltracking
{

// There are configurable params for TPC-ITS matching
struct MatchITSTPCParams : public o2::conf::ConfigurableParamHelper<MatchITSTPCParams> {
  bool runAfterBurner = false;
  float crudeAbsDiffCut[o2::track::kNParams] = {2.f, 2.f, 0.2f, 0.2f, 4.f};
  float crudeNSigma2Cut[o2::track::kNParams] = {49.f, 49.f, 49.f, 49.f, 49.f};

  float minTPCPt = 0.04;   ///< cut on minimal pT of TPC tracks to consider for matching
  int minTPCClusters = 25; ///< minimum number of clusters to consider
  int askMinTPCRow = 15;   ///< disregard tracks starting above this row

  float cutMatchingChi2 = 30.f; ///< cut on matching chi2

  float cutABTrack2ClChi2 = 30.f; ///< cut on AfterBurner track-cluster chi2

  int maxMatchCandidates = 5; ///< max allowed matching candidates per TPC track

  int ABRequireToReachLayer = 5; ///< AB tracks should reach at least this layer from above

  float TPCITSTimeBinSafeMargin = 1.f; ///< safety margin (in TPC time bins) for ITS-TPC tracks time (in TPC time bins!) comparison

  float TPCTimeEdgeZSafeMargin = 20.f; ///< safety margin in cm when estimating TPC track tMin and tMax from assigned time0 and its track Z position

  float TimeBinTolerance = 10.f; ///<tolerance in time-bin for ITS-TPC time bracket matching (not used ? TODO)

  O2ParamDef(MatchITSTPCParams, "tpcitsMatch");
};

} // namespace globaltracking
} // end namespace o2

#endif
