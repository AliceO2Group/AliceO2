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

#ifndef O2_MCH_CONDITIONS_STATUSMAP_CREATOR_PARAM_H_
#define O2_MCH_CONDITIONS_STATUSMAP_CREATOR_PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::mch
{

/**
 * @class StatusMapCreatorParam
 * @brief Configurable parameters for the statusmap creator
 */
struct StatusMapCreatorParam : public o2::conf::ConfigurableParamHelper<StatusMapCreatorParam> {

  bool useBadChannels = true; ///< reject bad channels (obtained during pedestal calibration runs)
  bool useRejectList = true;  ///< use extra (relative to bad channels above) rejection list

  bool isActive() const { return useBadChannels || useRejectList; }

  O2ParamDef(StatusMapCreatorParam, "MCHStatusMap");
};

} // namespace o2::mch

#endif
