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

#ifndef O2_FT0CHANNELTIMECALIBRATIONSPEC_H
#define O2_FT0CHANNELTIMECALIBRATIONSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "FITCalibration/FITCalibrationDevice.h"
#include "FT0Calibration/FT0CalibrationInfoObject.h"
#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"
#include "FT0Calibration/FT0ChannelTimeTimeSlotContainer.h"

namespace o2::ft0
{
o2::framework::DataProcessorSpec getFT0ChannelTimeCalibrationSpec()
{
  using CalibrationDeviceType = o2::fit::FITCalibrationDevice<o2::ft0::FT0CalibrationInfoObject,
                                                              o2::ft0::FT0ChannelTimeTimeSlotContainer, o2::ft0::FT0ChannelTimeCalibrationObject>;

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "FT0_ChannelTime"});
  outputs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "FT0_ChannelTime"});

  constexpr const char* DEFAULT_INPUT_LABEL = "calib";

  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back(DEFAULT_INPUT_LABEL, "FT0", "CALIB_INFO");

  return o2::framework::DataProcessorSpec{
    "calib-ft0-channel-time",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<CalibrationDeviceType>(DEFAULT_INPUT_LABEL)},
    o2::framework::Options{}};
}
} // namespace o2::ft0

#endif //O2_FT0CHANNELTIMECALIBRATIONSPEC_H
