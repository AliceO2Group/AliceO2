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

#ifndef O2_CALIBRATION_LHCCLOCK_CALIBRATOR_H
#define O2_CALIBRATION_LHCCLOCK_CALIBRATOR_H

/// @file   LHCClockCalibratorSpec.h
/// @brief  Device to calibrate LHC clock phase using FT0 data

#include "Framework/DataProcessorSpec.h"
#include "FT0Calibration/LHCClockDataHisto.h"
#include "FT0Calibration/LHCphaseCalibrationObject.h"
#include "FT0Calibration/FT0CalibrationInfoObject.h"
#include "FITCalibration/FITCalibrationDevice.h"

using namespace o2::framework;
namespace o2::ft0
{
o2::framework::DataProcessorSpec getLHCClockCalibDeviceSpec()
{
  using CalibrationDeviceType = o2::fit::FITCalibrationDevice<o2::ft0::FT0CalibrationInfoObject, o2::ft0::LHCClockDataHisto, o2::ft0::LHCphaseCalibrationObject>;

  constexpr const char* DEFAULT_INPUT_LABEL = "calib";

  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back(DEFAULT_INPUT_LABEL, "FT0", "CALIB_INFO");
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "FIT_CALIB"}, o2::framework::Lifetime::Sporadic);
  outputs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "FIT_CALIB"}, o2::framework::Lifetime::Sporadic);
  return o2::framework::DataProcessorSpec{
    "ft0-lhcphase-calibration",
    inputs, // o2::framework::Inputs{{"input", "FT0", "CALIBDATA"}},
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<CalibrationDeviceType>(ccdbRequest, DEFAULT_INPUT_LABEL)},
    Options{
      {"tf-per-slot", VariantType::UInt32, 26000u, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::UInt32, 3u, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 500, {"minimum number of entries to fit single time slot"}}}};
}

} // namespace o2::ft0

#endif
