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

/// \file TrackMatcherParam.h
/// \brief Configurable parameters for MCH-MID track matching
/// \author Philippe Pillot, Subatech

#ifndef O2_MUON_TRACKMATCHERPARAM_H_
#define O2_MUON_TRACKMATCHERPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace muon
{

/// Configurable parameters for MCH-MID track matching
struct TrackMatcherParam : public o2::conf::ConfigurableParamHelper<TrackMatcherParam> {

  double sigmaCut = 4.; ///< to select compatible MCH and MID tracks according to their matching chi2

  O2ParamDef(TrackMatcherParam, "MUONMatching");
};

} // namespace muon
} // namespace o2

#endif // O2_MUON_TRACKMATCHERPARAM_H_
