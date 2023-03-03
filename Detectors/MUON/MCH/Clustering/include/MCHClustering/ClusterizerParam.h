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

/// \file ClusterizerParam.h
/// \brief Configurable parameters for MCH clustering
/// \author Philippe Pillot, Subatech

#ifndef O2_MCH_CLUSTERIZERPARAM_H_
#define O2_MCH_CLUSTERIZERPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace mch
{

/// Configurable parameters for MCH clustering
struct ClusterizerParam : public o2::conf::ConfigurableParamHelper<ClusterizerParam> {

  double lowestPadCharge = 4.f * 0.22875f; ///< minimum charge of a pad

  double defaultClusterResolutionX = 0.2; ///< default cluster resolution in x direction (cm)
  double defaultClusterResolutionY = 0.2; ///< default cluster resolution in y direction (cm)

  double badClusterResolutionX = 10.; ///< bad (e.g. mono-cathode) cluster resolution in x direction (cm)
  double badClusterResolutionY = 10.; ///< bad (e.g. mono-cathode) cluster resolution in y direction (cm)

  bool legacy = true; ///< use original (run2) clustering

  O2ParamDef(ClusterizerParam, "MCHClustering");
};

} // namespace mch
} // end namespace o2

#endif // O2_MCH_CLUSTERIZERPARAM_H_
