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

#ifndef O2_MCH_CALIBRATION_BADCHANNEL_CALIBRATOR_PARAM_H_
#define O2_MCH_CALIBRATION_BADCHANNEL_CALIBRATOR_PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::mch::calibration
{

/**
 * @class BadChannelCalibratorParam
 * @brief Configurable parameters for the Bad Channel Calibrator
 */
struct BadChannelCalibratorParam : public o2::conf::ConfigurableParamHelper<BadChannelCalibratorParam> {

  float maxPed = 200.f; ///< maximum allowed pedestal value
  float maxNoise = 2.f; ///< maximum allowed noise value

  int minRequiredNofEntriesPerChannel = 10000; ///< mininum pedestal digits per channel needed to assess a channel quality
  float minRequiredCalibratedFraction = 0.9f;  ///< minimum fraction of channels for which we need a quality value to produce a bad channel map.

  bool onlyAtEndOfStream = {true}; ///< only produce bad channel map at end of stream (EoS). In that case the minRequiredCalibratedFraction and minRequiredNofEntriesPerChannel are irrelevant.

  O2ParamDef(BadChannelCalibratorParam, "MCHBadChannelCalibratorParam");
};
} // namespace o2::mch::calibration

#endif
