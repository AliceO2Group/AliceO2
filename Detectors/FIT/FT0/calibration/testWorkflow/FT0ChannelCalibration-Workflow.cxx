// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.


#include "FITCalibration/FITCalibrationDevice.h"


using namespace o2::calibration::fit;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  workflowOptions.emplace_back(o2::framework::ConfigParamSpec{"initialTimestamp",
                                                           o2::framework::VariantType::Int, 0,
                                                           {"Timestamp of initial calibration object that will be read from CCDB"}});
}

#include "Framework/runDataProcessing.h"

using namespace o2::framework;
WorkflowSpec defineDataProcessing(ConfigContext const& config)
{

  using CalibrationDeviceType = o2::calibration::fit::FITCalibrationDevice<o2::calibration::fit::FT0CalibrationInfoObject,
    o2::calibration::fit::FT0ChannelDataTimeSlotContainer, o2::calibration::fit::FT0CalibrationObject>;


  auto initialTimestamp = config.options().get<int>("initialTimestamp");
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCLB, o2::calibration::Utils::gDataDescriptionCLBPayload});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCLB, o2::calibration::Utils::gDataDescriptionCLBInfo});

  constexpr const char* inputDataLabel = "calib";
  constexpr const char* calibrationObjectPath = "FT0/Calibration/CalibrationObject";

  std::vector<InputSpec> inputs;
  inputs.emplace_back(inputDataLabel, "FT0", "CALIB_INFO");

  DataProcessorSpec dataProcessorSpec{
    "calib-ft0channel-calibration",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CalibrationDeviceType>(inputDataLabel, calibrationObjectPath, initialTimestamp)},
    Options {}
  };


  WorkflowSpec workflow;
  workflow.emplace_back(dataProcessorSpec);
  return workflow;
}
