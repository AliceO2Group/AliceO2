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

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/WorkflowSpec.h"
#include "BadChannelCalibrationDevice.h"

using namespace o2::framework;

const char* specName = "mch-badchannel-calibrator";

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"input-pdigits-data-description", VariantType::String, "PDIGITS", {"input pedestal digits data description"}});
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
}

// ------------------------------------------------------------------

DataProcessorSpec getBadChannelCalibratorSpec(const char* specName, const std::string inputSpec)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "MCH_BADCHAN"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "MCH_BADCHAN"}, Lifetime::Sporadic);
  outputs.emplace_back(OutputSpec{"MCH", "PEDESTALS", 0, Lifetime::Sporadic});
  std::vector<InputSpec> inputs = o2::framework::select(fmt::format("digits:MCH/{}", inputSpec.data()).c_str());
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return DataProcessorSpec{
    specName,
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::mch::calibration::BadChannelCalibrationDevice>(ccdbRequest)},
    Options{
      {"logging-interval", VariantType::Int, 0, {"time interval in seconds between logging messages (set to zero to disable)"}},
    }};
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  const std::string inputSpec = configcontext.options().get<std::string>("input-pdigits-data-description");
  WorkflowSpec specs;
  specs.emplace_back(getBadChannelCalibratorSpec(specName, inputSpec));
  return specs;
}
