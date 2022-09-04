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

#include "Framework/ConfigParamSpec.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"use-clusters", VariantType::Bool, false, {"Use clusters instead of digits"}});
  workflowOptions.push_back(ConfigParamSpec{"processing-mode", VariantType::Int, 0, {"processing mode: 0 - accumulate & normalize, 1 - accumulated and send w/o normalization, 2 - receive maps from mode 1 accumulators and normalized"}});
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"
#include "ITSCalibration/NoiseCalibratorSpec.h"
#include "ITSCalibration/NoiseCalibrator.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  specs.emplace_back(o2::its::getNoiseCalibratorSpec(configcontext.options().get<bool>("use-clusters"), configcontext.options().get<int>("processing-mode")));
  return specs;
}
