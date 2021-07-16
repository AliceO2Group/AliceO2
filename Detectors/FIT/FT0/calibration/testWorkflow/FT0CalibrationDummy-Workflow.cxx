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

//Dummy, delete this file if example not needed anymore
#include "Framework/DataProcessorSpec.h"
#include "FITCalibration/FITCalibrationDevice.h"
#include "FT0Calibration/FT0CalibrationInfoObject.h"
#include "FT0Calibration/FT0ChannelTimeTimeSlotContainer.h"
#include "FT0Calibration/FT0DummyCalibrationObject.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::fit;
using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  WorkflowSpec workflow;
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));

  using CalibrationDeviceType = o2::fit::FITCalibrationDevice<o2::ft0::FT0CalibrationInfoObject,
                                                              o2::ft0::FT0ChannelTimeTimeSlotContainer, o2::ft0::FT0DummyCalibrationObject>;

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "FIT_CALIB"});
  outputs.emplace_back(o2::framework::ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "FIT_CALIB"});

  constexpr const char* DEFAULT_INPUT_LABEL = "calib";

  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back(DEFAULT_INPUT_LABEL, "FT0", "CALIB_INFO");

  o2::framework::DataProcessorSpec dataProcessorSpec{
    "calib-ft0-channel-time",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<CalibrationDeviceType>(DEFAULT_INPUT_LABEL)},
    o2::framework::Options{}};

  workflow.emplace_back(dataProcessorSpec);
  return workflow;
}
