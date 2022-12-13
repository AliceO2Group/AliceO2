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
#include "DataFormatsFT0/GlobalOffsetsCalibrationObject.h"
#include "DataFormatsFT0/GlobalOffsetsInfoObject.h"
#include "FT0Calibration/GlobalOffsetsContainer.h"
#include "FITCalibration/FITCalibrationDevice.h"

using namespace o2::framework;
namespace o2::ft0
{
o2::framework::DataProcessorSpec getGlobalOffsetsCalibrationSpec()
{
  using CalibrationDeviceType = o2::fit::FITCalibrationDevice<o2::ft0::GlobalOffsetsInfoObject, o2::ft0::GlobalOffsetsContainer, o2::ft0::GlobalOffsetsCalibrationObject>;
  LOG(info) << " getGlobalOffsetsCalibrationSpec()";

  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  const o2::header::DataDescription inputDataDescriptor{"CALIB_INFO"};
  const o2::header::DataDescription outputDataDescriptor{"FT0_GLTIME_CALIB"};
  CalibrationDeviceType::prepareVecInputSpec(inputs, o2::header::gDataOriginFT0, inputDataDescriptor);
  CalibrationDeviceType::prepareVecOutputSpec(outputs, outputDataDescriptor);
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return o2::framework::DataProcessorSpec{
    "ft0-global-offsets",
    inputs, // o2::framework::Inputs{{"input", "FT0", "CALIBDATA"}},
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<CalibrationDeviceType>(ccdbRequest, outputDataDescriptor)},
    Options{
      {"tf-per-slot", VariantType::UInt32, 55000u, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::UInt32, 3u, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 500, {"minimum number of entries to fit single time slot"}},
      {"extra-info-per-slot", o2::framework::VariantType::String, "", {"Extra info for time slot(usually for debugging)"}}}};
}

} // namespace o2::ft0

#endif
