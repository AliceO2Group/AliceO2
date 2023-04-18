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

#ifndef O2_MID_FILTERERBCPARAM_H
#define O2_MID_FILTERERBCPARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace mid
{

/**
 * @class FiltererBCParam
 * @brief Configurable parameters for the BC filterer
 */
struct FiltererBCParam : public o2::conf::ConfigurableParamHelper<FiltererBCParam> {

  int maxBCDiffLow = -1;   ///< Maximum BC diff in the lower side
  int maxBCDiffHigh = 1;   ///< Maximum BC diff in the upper side
  bool selectOnly = false; ///< Selects BCs but does not merge them

  O2ParamDef(FiltererBCParam, "MIDFiltererBC");
};
} // namespace mid
} // namespace o2

#endif
