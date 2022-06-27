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

/// \file PreClusterFinderParam.h
/// \brief Configurable parameters for MCH preclustering
/// \author Philippe Pillot, Subatech

#ifndef O2_MCH_PRECLUSTERFINDERPARAM_H_
#define O2_MCH_PRECLUSTERFINDERPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace mch
{

/// Configurable parameters for MCH preclustering
struct PreClusterFinderParam : public o2::conf::ConfigurableParamHelper<PreClusterFinderParam> {

  bool excludeCorners = false; ///< exclude corners when looking for fired neighbouring pads

  O2ParamDef(PreClusterFinderParam, "MCHPreClustering");
};

} // namespace mch
} // end namespace o2

#endif // O2_MCH_PRECLUSTERFINDERPARAM_H_
