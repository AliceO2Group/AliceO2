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

/// @file   AlignConfig.h
/// @brief  Configuration file for global alignment

#ifndef ALICEO2_ALIGN_CONFIG_H
#define ALICEO2_ALIGN_CONFIG_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "DetectorsBase/Propagator.h"

namespace o2
{
namespace align
{
using PropatatorD = o2::base::PropagatorImpl<double>;
struct AlignConfig : public o2::conf::ConfigurableParamHelper<AlignConfig> {
  enum TrackType { Collision,
                   Cosmic,
                   NTrackTypes };

  float maxStep = 3.;  // max step for propagation
  float maxSnp = 0.95; // max snp for propagation
  int matCorType = (int)o2::base::PropagatorD::MatCorrType::USEMatCorrLUT;
  float q2PtMin[NTrackTypes] = {0.01, 0.01};
  float q2PtMax[NTrackTypes] = {10., 10.};
  float tglMax[NTrackTypes] = {3., 10.};
  float defPTB0Coll = 0.6;
  float defPTB0Cosm = 3.0;
  int minPoints[NTrackTypes] = {4, 10};
  int minDetAcc[NTrackTypes] = {1, 1};

  float minScatteringAngleToAccount = 0.0003;

  int verbose = 0;

  int vtxMinCont = 2;     // require min number of contributors in Vtx
  int vtxMaxCont = 99999; // require max number of contributors in Vtx
  int vtxMinContVC = 20;  // min number of contributors to use as constraint

  int minPointTotal = 4; // total min number of alignment point to account track
  int minDetectors = 1;  // min number of detectors per track
  int minITSClusters = 4;  // min ITS clusters to accept the track
  int minTRDTracklets = 3; // min TRD tracklets to accept the track
  int minTPCClusters = 10; // discard tracks with less clusters
  int minTOFClusters = 1;  // min TOF clusters to accept track
  int maxTPCRowsCombined = 1;        // allow combining clusters on so many rows to a single cluster
  int discardEdgePadrows = 3;        // discard padrow if its distance to stack edge padrow < this
  float discardSectorEdgeDepth = 2.5; // discard clusters too close to the sector edge
  float ITSOverlapMargin = 0.15;     // consider for overlaps only clusters within this marging from the chip edge (in cm)
  float ITSOverlapMaxChi2 = 16;      // max chi2 between track and overlapping cluster
  int ITSOverlapEdgeRows = 1;        // require clusters to not have pixels closer than this distance from the edge
  float ITSOverlapMaxDZ = 0.3;       // max difference in Z for clusters on overlapping ITS chips to consider as candidate for a double hit

  int minPointTotalCosm = 4;      // total min number of alignment point to account cosmic track
  int minDetectorsCosm = 1;       // min number of detectors per cosmic track
  int minITSClustersCosm = 0;     // min ITS clusters to accept the cosmic track
  int minITSClustersCosmLeg = 2;  // min ITS clusters per leg to accept the cosmic track
  int minTRDTrackletsCosm = 0;    // min TRD tracklets to accept the cosmic track
  int minTRDTrackletsCosmLeg = 2; // min TRD tracklets per leg to accept the cosmic track
  int minTPCClustersCosm = 0;     // discard cosmic tracks with less clusters
  int minTPCClustersCosmLeg = 10; // discard cosmic tracks with less clusters per leg
  int minTOFClustersCosm = 0;     // min TOF clusters to accept track
  int minTOFClustersCosmLeg = 1;  // min TOF clusters per leg to accept track

  int minTPCPadRow = 6;   // min TPC pad-row to account
  int maxTPCPadRow = 146; // max TPC pad-row to account

  float cosmMaxDSnp = 0.025; // reject cosmic tracks with larger than this snp difference
  float cosmMaxDTgl = 0.1;   // reject cosmic tracks with larger than this tgl difference

  float maxDCAforVC[2] = {-1, -1}; // DCA cut in R,Z to allow track be subjected to vertex constraint
  float maxChi2forVC = -1;         // track-vertex chi2 cut to allow the track be subjected to vertex constraint
  float alignParamZero = 1e-13;    // assign 0 to final alignment parameter if its abs val is below this threshold
  float controlFraction = -1.;     // fraction for which control output is requested, if negative - only 1st instance of device will write them
  float MPRecOutFraction = -1.;    // compact Millepede2Record fraction, if negative - only 1st instance of device will write them

  bool MilleOut = true;       // Mille output
  bool KalmanResid = true;    // Kalman residuals
  bool MilleOutBin = true;    // text vs binary output for mille data
  bool GZipMilleOut = false;  // compress binary records

  std::string mpDatFileName{"mpData"};            //  file name for records mille data output
  std::string mpParFileName{"mpParams.txt"};      //  file name for MP params
  std::string mpConFileName{"mpConstraints.txt"}; //  file name for MP constraints
  std::string mpSteerFileName{"mpSteer.txt"};     //  file name for MP steering
  std::string residFileName{"mpContolRes"};       //  file name for optional control residuals
  std::string mpLabFileName{"mpResultsLabeled.txt"}; //  file name for relabeled MP params
  //
  std::string outCDBPath{};        // output OCDB path
  std::string outCDBComment{};     // optional comment to add to output cdb objects
  std::string outCDBResponsible{}; // optional responsible for output metadata

  O2ParamDef(AlignConfig, "alignConf");
};

} // namespace align
} // namespace o2

#endif
