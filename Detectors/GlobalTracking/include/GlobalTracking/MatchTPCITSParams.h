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

  bool runAfterBurner = false;                    ///< run afterburner for TPCtrack-ITScluster matching
  ValidateMatchByFIT validateMatchByFIT = Prefer; ///< when comparing ITS-TPC matches, prefer those which have time of Interaction Candidate
  float crudeAbsDiffCut[o2::track::kNParams] = {2.f, 2.f, 0.2f, 0.2f, 4.f};
  float crudeNSigma2Cut[o2::track::kNParams] = {49.f, 49.f, 49.f, 49.f, 49.f};

  float minTPCTrackR = 26.6; ///< cut on minimal TPC tracks radius to consider for matching, 666*pt_gev*B_kgaus/5
  float minITSTrackR = 26.6; ///< cut on minimal ITS tracks radius to consider for matching, 666*pt_gev*B_kgaus/5
  int minTPCClusters = 25; ///< minimum number of clusters to consider
  int askMinTPCRow = 15;   ///< disregard tracks starting above this row

  float cutMatchingChi2 = 30.f; ///< cut on matching chi2

  float cutABTrack2ClChi2 = 30.f; ///< cut on AfterBurner track-cluster chi2

  int maxMatchCandidates = 5; ///< max allowed matching candidates per TPC track

  int requireToReachLayerAB = 5; ///< AB tracks should reach at least this layer from above

  float safeMarginTPCITSTimeBin = 1.f; ///< safety margin (in TPC time bins) for ITS-TPC tracks time (in TPC time bins!) comparison

  float safeMarginTPCTimeEdge = 20.f; ///< safety margin in cm when estimating TPC track tMin and tMax from assigned time0 and its track Z position

  float timeBinTolerance = 10.f; ///<tolerance in time-bin for ITS-TPC time bracket matching (not used ? TODO)

  float tpcTimeICMatchingNSigma = 4.; ///< nsigma for matching TPC corrected time and InteractionCandidate from FT0

  float tpcExtConstrainedNSigma = 4.; ///< nsigma to apply to externally (TRD,TOF) time-constrained TPC tracks time error

  float maxVDriftUncertainty = 0.; ///< max assumed VDrift uncertainty, used only in VDrift calibration mode

  float maxTglForVDriftCalib = 1.; ///< maximum ITS tgl to collect data for VDrift calibration
  int nBinsTglVDriftCalib = 50;    ///< number of bins in reference ITS tgl for VDrift calibration
  int nBinsDTglVDriftCalib = 100;  ///< number of bins in delta tgl for VDrift calibration

  o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT; /// Material correction type

  O2ParamDef(MatchTPCITSParams, "tpcitsMatch");
};

} // namespace globaltracking
} // end namespace o2

#endif
