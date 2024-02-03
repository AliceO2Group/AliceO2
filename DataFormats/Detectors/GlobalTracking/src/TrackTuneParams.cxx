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

/// \file TrackTuneParams.h
/// \brief Configurable params for tracks ad hoc tuning
/// \author ruben.shahoyan@cern.ch

#include "DataFormatsGlobalTracking/TrackTuneParams.h"
O2ParamImpl(o2::globaltracking::TrackTuneParams);

using namespace o2::globaltracking;

std::array<float, 5> TrackTuneParams::getCovInnerTotal(float scale) const
{
  std::array<float, 5> cov{};
  for (int i = 0; i < 5; i++) {
    cov[i] = tpcCovInner[i] + scale * tpcCovInnerSlope[i];
    cov[i] *= cov[i];
  }
  return cov;
}

std::array<float, 5> TrackTuneParams::getCovOuterTotal(float scale) const
{
  std::array<float, 5> cov{};
  for (int i = 0; i < 5; i++) {
    cov[i] = tpcCovOuter[i] + scale * tpcCovOuterSlope[i];
    cov[i] *= cov[i];
  }
  return cov;
}
