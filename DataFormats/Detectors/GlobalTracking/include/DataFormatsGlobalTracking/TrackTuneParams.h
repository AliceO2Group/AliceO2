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

#ifndef ALICEO2_TRACK_TUNE_PARAMS_H
#define ALICEO2_TRACK_TUNE_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <array>

namespace o2
{
namespace globaltracking
{

// There are configurable params for tracks ad hoc tuning

struct TrackTuneParams : public o2::conf::ConfigurableParamHelper<TrackTuneParams> {
  enum AddCovType {  // how to add covariances to tracks
    Disable,         // do not add
    NoCorrelations,  // add ignoring correlations (i.e. to diagonal elements only)
    WithCorrelations // add adjusting non-diagonal elements to preserve correlation coefficients
  };
  AddCovType tpcCovInnerType = AddCovType::Disable;
  AddCovType tpcCovOuterType = AddCovType::Disable;
  bool sourceLevelTPC = true;    // if TPC corrections are allowed, apply them TPC source output level (tracking), otherwise in the global tracking consumers BEFORE update by external detector
  bool applyWhenReading = false; // if true, then apply at reading tracks from the file. This better NOT used as there is no way to apply lumi-dependent scaling in the reader
  bool useTPCInnerCorr = false; // request to correct TPC inner param
  bool useTPCOuterCorr = false; // request to correct TPC outer param
  float tpcParInner[5] = {};    // ad hoc correction to be added to TPC param at the inner XRef
  float tpcParOuter[5] = {};    // ad hoc correction to be added to TPC param at the outer XRef
  float tpcCovInner[5] = {};    // ad hoc errors to be added to TPC cov.matrix at the inner XRef (not squared!)
  float tpcCovOuter[5] = {};    // ad hoc errors to be added to TPC outer param cov.matrix at the outer XRef (not squared!)
  float tpcCovInnerSlope[5] = {}; // slope of the error with respect to lumi, total correction = [tpcCovInner + lumi*tpcCovInnerSlope]^2
  float tpcCovOuterSlope[5] = {}; // slope of the error with respect to lumi, total correction = [tpcCovOuter + lumi*tpcCovOuterSlope]^2

  std::array<float, 5> getCovInnerTotal(float scale) const;
  std::array<float, 5> getCovOuterTotal(float scale) const;

  O2ParamDef(TrackTuneParams, "trackTuneParams");
};

} // namespace globaltracking
} // end namespace o2

#endif
