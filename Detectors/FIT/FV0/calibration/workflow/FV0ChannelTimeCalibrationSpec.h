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

#ifndef O2_FV0CHANNELTIMECALIBRATIONSPEC_H
#define O2_FV0CHANNELTIMECALIBRATIONSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "FITCalibration/FITCalibrationDevice.h"
#include "FV0Calibration/FV0ChannelTimeCalibrationObject.h"
#include "FV0Calibration/FV0ChannelTimeTimeSlotContainer.h"
#include "FV0Calibration/FV0CalibrationInfoObject.h"

namespace o2::fv0
{

o2::framework::DataProcessorSpec getFV0ChannelTimeCalibrationSpec()
{
  using CalibrationDeviceType = o2::fit::FITCalibrationDevice<o2::fv0::FV0CalibrationInfoObject,
                                                              o2::fv0::FV0ChannelTimeTimeSlotContainer, o2::fv0::FV0ChannelTimeCalibrationObject>;

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "FIT_CALIB"}, o2::framework::Lifetime::Sporadic);
  outputs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "FIT_CALIB"}, o2::framework::Lifetime::Sporadic);

  constexpr const char* DEFAULT_INPUT_LABEL = "calib";

  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back(DEFAULT_INPUT_LABEL, "FV0", "CALIB_INFO");
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return o2::framework::DataProcessorSpec{
    "calib-fv0-channel-time",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<CalibrationDeviceType>(DEFAULT_INPUT_LABEL, ccdbRequest)},
    o2::framework::Options{
      {"tf-per-slot", o2::framework::VariantType::UInt32, 5u, {""}},
      {"max-delay", o2::framework::VariantType::UInt32, 3u, {""}},
      {"updateInterval", o2::framework::VariantType::UInt32, 10u, {""}}}};
}
} // namespace o2::fv0

#endif //O2_FV0CHANNELTIMECALIBRATIONSPEC_H
