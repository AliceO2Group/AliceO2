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

#ifndef O2_ECAL_BASEPARAM_H
#define O2_ECAL_BASEPARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace ecal
{
struct ECalBaseParam : public o2::conf::ConfigurableParamHelper<ECalBaseParam> {
  float rMin = 125.0;    // cm
  float rMax = 155.0;    // cm
  float zLength = 350.0; // cm

  bool enableFwdEndcap = true;

  O2ParamDef(ECalBaseParam, "ECalBase");
};

} // namespace ecal
} // end namespace o2

#endif