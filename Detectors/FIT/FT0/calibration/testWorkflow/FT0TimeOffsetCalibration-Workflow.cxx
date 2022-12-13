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

#include "Framework/DataProcessorSpec.h"

#include "CommonUtils/ConfigurableParam.h"
#include "DataFormatsFT0/SpectraInfoObject.h"
#include "FITCalibration/FITCalibrationDevice.h"
#include "FT0Calibration/FT0TimeOffsetSlotContainer.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // probably some option will be added
  std::vector<o2::framework::ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using CalibrationDeviceType = o2::fit::FITCalibrationDevice<float,
                                                              o2::ft0::FT0TimeOffsetSlotContainer, o2::ft0::TimeSpectraInfoObject>;
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  const o2::header::DataDescription inputDataDescriptor{"TIME_SPECTRA"};
  const o2::header::DataDescription outputDataDescriptor{"FT0_TIME_CALIB"};
  CalibrationDeviceType::prepareVecInputSpec(inputs, o2::header::gDataOriginFT0, inputDataDescriptor);
  CalibrationDeviceType::prepareVecOutputSpec(outputs, outputDataDescriptor);

  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  o2::framework::DataProcessorSpec dataProcessorSpec{
    "ft0-time-offset-calib",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<CalibrationDeviceType>(ccdbRequest, outputDataDescriptor)},
    o2::framework::Options{
      {"tf-per-slot", o2::framework::VariantType::UInt32, 56000u, {""}},
      {"max-delay", o2::framework::VariantType::UInt32, 3u, {""}},
      {"extra-info-per-slot", o2::framework::VariantType::String, "", {"Extra info for time slot(usually for debugging)"}}}};

  WorkflowSpec workflow;
  workflow.emplace_back(dataProcessorSpec);
  return workflow;
}
