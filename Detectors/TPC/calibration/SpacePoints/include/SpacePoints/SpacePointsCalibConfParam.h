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

/// \author ole.schmidt@cern.ch

#ifndef ALICEO2_SCDCALIB_PARAMS_H
#define ALICEO2_SCDCALIB_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace tpc
{

// These are configurable params for the TPC space point calibration
struct SpacePointsCalibConfParam : public o2::conf::ConfigurableParamHelper<SpacePointsCalibConfParam> {

  int maxTracksPerCalibSlot = 3'500'000; ///< the number of tracks which is required to obtain an average correction map

  // define track cuts for track interpolation
  int minTPCNCls = 70;             ///< min number of TPC clusters
  int minTPCNClsNoOuterPoint = 50; ///< min number of TPC clusters if no hit in TRD or TOF exists
  float maxTPCChi2 = 4.f;          ///< cut on TPC reduced chi2
  int minITSNCls = 4;              ///< min number of ITS clusters
  int minITSNClsNoOuterPoint = 6;  ///< min number of ITS clusters if no hit in TRD or TOF exists
  int minTRDNTrklts = 3;           ///< min number of TRD space points
  float maxITSChi2 = 20.f;         ///< cut on ITS reduced chi2
  float maxTRDChi2 = 10.f;         ///< cut on TRD reduced chi2
  float minPtNoOuterPoint = 0.8f;  ///< minimum pt for ITS-TPC tracks to be considered for extrapolation

  // other settings for track interpolation
  float sigYZ2TOF{.75f}; ///< for now assume cluster error for TOF equal for all clusters in both Y and Z
  float maxSnp{.85f};    ///< max snp when propagating tracks
  float maxStep{2.f};    ///< maximum step for propagation
  bool debugTRDTOF{false}; ///< if true, ITS-TPC-TRD-TOF tracks and their seeding ITS-TPC-TRD track will both be interpolated and their residuals stored

  // steering of map creation after the residuals have already been written to file
  bool writeBinnedResiduals{false}; ///< when creating the map from unbinned residuals store the binned residuals together with the voxel results
  bool useTrackData{false}; ///< if we have the track data available, we can redefine the above cuts for the map creation, e.g. minTPCNCls etc
  bool timeFilter{false};   ///< consider only residuals as input from TFs with a specific time range specified via startTimeMS and endTimeMS
  long startTimeMS{0L};     ///< the start of the time range in MS
  long endTimeMS{1999999999999L}; ///< the end of the time range in MS
  bool cutOnDCA{true};            ///< when creating the map from unbinned residuals cut on DCA estimated from ITS outer parameter
  float maxDCA = 10.f;            ///< DCA cut value in cm

  // parameters for outlier rejection
  bool skipOutlierFiltering{false};      ///< if set, the outlier filtering will not be applied at all
  bool writeUnfiltered{false};           ///< if set, all residuals and track parameters will be aggregated and dumped additionally without outlier rejection
  int nMALong{15};                       ///< number of points to be used for moving average (long range)
  int nMAShort{3};                       ///< number of points to be used for estimation of distance from local line (short range)
  float maxRejFrac{.15f};                ///< if the fraction of rejected clusters of a track is higher, the full track is invalidated
  float maxRMSLong{.8f};                 ///< maximum variance of the cluster residuals wrt moving avarage for a track to be considered
  int minNCl = 30;                       ///< min number of clusters in a track to be used for calibration
  float maxQ2Pt = 3.f;                   ///< max fitted q/pt for a track to be used for calibration
  float maxDevHelixY = .3f;              ///< max deviation in Y for clusters wrt helix fit
  float maxDevHelixZ = .3f;              ///< max deviation in Z for clusters wrt helix fit
  int minNumberOfAcceptedResiduals = 30; ///< min number of accepted residuals for
  float maxStdDevMA = 25.f;              ///< max cluster std. deviation (Y^2 + Z^2) wrt moving average to accept

  // settings for voxel residuals extraction
  int minEntriesPerVoxel = 15;         ///< minimum number of points in voxel for processing
  float LTMCut = .75f;                 ///< fraction op points to keep when trimming input data
  float minFracLTM = .5f;              ///< minimum fraction of points to keep when trimming data to fit expected sigma
  float minValidVoxFracDrift = .5f;    ///< if more than this fraction of bins are bad for one pad row the whole pad row is declared bad
  int minGoodXBinsToCover = 3;         ///< minimum number of consecutive good bins, otherwise bins are declared bad
  int maxBadXBinsToCover = 4;          ///< a lower number of consecutive bad X bins will not be declared bad
  float maxFracBadRowsPerSector = .4f; ///< maximum fraction of bad rows before whole sector is masked
  float maxFitErrY2 = 1.f;             ///< maximum fit error for Y2
  float maxFitErrX2 = 9.f;             ///< maximum fit error for X2
  float maxFitCorrXY = .95f;           ///< maximum fit correlation for x and y
  float maxSigY = 1.1f;                ///< maximum sigma for y of the voxel
  float maxSigZ = .7f;                 ///< maximum sigma for z of the voxel
  float maxGaussStdDev = 5.f;          ///< maximum number of sigmas to be considered for gaussian kernel smoothing

  O2ParamDef(SpacePointsCalibConfParam, "scdcalib");
};

} // namespace tpc
} // end namespace o2

#endif // ALICEO2_SCDCALIB_PARAMS_H
