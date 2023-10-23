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

#ifndef ALICEO2_MATCHITSTPC_PARAMS_H
#define ALICEO2_MATCHITSTPC_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "ReconstructionDataFormats/Track.h"
#include "DetectorsBase/Propagator.h"

namespace o2
{
namespace globaltracking
{

// There are configurable params for TPC-ITS matching
struct MatchTPCITSParams : public o2::conf::ConfigurableParamHelper<MatchTPCITSParams> {
  enum ValidateMatchByFIT { Disable,
                            Prefer,
                            Require }; // flags for usage of FT0 in match validation
  enum TimeOutliersPolicy {            // policy for matching timestamps outside of respective ITS ROF bracket
    Tolerate,                          // accept as is
    Adjust,                            // adjust to closest ITS bracket boundary
    Reject                             // reject match
  };
  bool runAfterBurner = true;                     ///< run afterburner for TPCtrack-ITScluster matching
  ValidateMatchByFIT validateMatchByFIT = Prefer; ///< when comparing ITS-TPC matches, prefer those which have time of Interaction Candidate
  TimeOutliersPolicy ITSTimeOutliersPolicy = Adjust;
  float crudeAbsDiffCut[o2::track::kNParams] = {2.f, 2.f, 0.2f, 0.2f, 4.f};
  float crudeNSigma2Cut[o2::track::kNParams] = {49.f, 49.f, 49.f, 49.f, 49.f};

  float XMatchingRef = 70.f; ///< reference radius to propagate tracks for matching

  float minTPCTrackR = 50.; ///< cut on minimal TPC tracks radius to consider for matching, 666*pt_gev*B_kgaus/5
  float minITSTrackR = 50.; ///< cut on minimal ITS tracks radius to consider for matching, 666*pt_gev*B_kgaus/5
  int minTPCClusters = 25; ///< minimum number of clusters to consider
  int askMinTPCRow = 15;   ///< disregard tracks starting above this row

  float cutMatchingChi2 = 30.f; ///< cut on matching chi2

  int maxMatchCandidates = 5; ///< max allowed matching candidates per TPC track

  float safeMarginTimeCorrErr = 0; ///< safety marging (in \mus) for TPC track time corrected by ITS constraint

  float safeMarginTPCITSTimeBin = 1.f; ///< safety margin (in TPC time bins) for ITS-TPC tracks time (in TPC time bins!) comparison

  float safeMarginTPCTimeEdge = 20.f; ///< safety margin in cm when estimating TPC track tMin and tMax from assigned time0 and its track Z position

  float tpcTimeICMatchingNSigma = 4.; ///< nsigma for matching TPC corrected time and InteractionCandidate from FT0

  float tpcExtConstrainedNSigma = 4.; ///< nsigma to apply to externally (TRD,TOF) time-constrained TPC tracks time error

  float tfEdgeTimeToleranceMUS = 1.; ///< corrected TPC time allowed to go out from the TF time edges by this amount

  float maxVDriftUncertainty = 0.02; ///< max assumed VDrift relative uncertainty, used only in VDrift calibration mode
  float maxVDriftTrackQ2Pt = 1.0;    ///< use only tracks below this q/pt (with field only)
  float maxVDritTimeOffset = 5.;     ///< max possible TDrift offset to calibrate

  float globalTimeBiasMUS = 0.; ///< global time shift to apply to assigned time, brute force way to eliminate bias wrt FIT
  float globalTimeExtraErrorMUS = 0.; ///< extra error to add to global time estimate

  //___________________ AfterBurner params
  int requireToReachLayerAB = 5;   ///< AB tracks should reach at least this layer from above
  int lowestLayerAB = 3;           ///< lowest layer to reach in AfterBurner
  int minContributingLayersAB = 2; ///< AB tracks must have at least this amount on contributing layers
  int maxABLinksOnLayer = 10;      ///< max prolongations for single seed from one to next layer
  int maxABFinalHyp = 20;          ///< max final hypotheses per TPC seed
  float cutABTrack2ClChi2 = 30.f;  ///< cut on AfterBurner track-cluster chi2
  float nABSigmaY = 4.;            ///< nSigma cut on afterburner track-cluster Y distance
  float nABSigmaZ = 4.;            ///< nSigma cut on afterburner track-cluster Z distance
  float err2ABExtraY = 0.1 * 0.1;  ///< extra "systematic" error on Y
  float err2ABExtraZ = 0.1 * 0.1;  ///< extra "systematic" error on Z

  int verbosity = 0; ///< verbosit level

  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT; /// Material correction type

  O2ParamDef(MatchTPCITSParams, "tpcitsMatch");
};

} // namespace globaltracking
} // end namespace o2

#endif
