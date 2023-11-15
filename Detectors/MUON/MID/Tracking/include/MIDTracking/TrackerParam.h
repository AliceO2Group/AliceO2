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

/// \file TrackerParam.h
/// \brief Configurable parameters for MID tracking
/// \author Philippe Pillot, Subatech

#ifndef O2_MID_TRACKERPARAM_H_
#define O2_MID_TRACKERPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace mid
{

/// Configurable parameters for MID tracking
struct TrackerParam : public o2::conf::ConfigurableParamHelper<TrackerParam> {

  float impactParamCut = 210.; ///< impact parameter cut to select track seeds
  float sigmaCut = 3.;         ///< sigma cut to select clusters and tracks during tracking

  std::size_t maxCandidates = 1000000; ///< maximum number of track candidates above which the tracking abort

  O2ParamDef(TrackerParam, "MIDTracking");
};

} // namespace mid
} // end namespace o2

#endif // O2_MID_TRACKERPARAM_H_
