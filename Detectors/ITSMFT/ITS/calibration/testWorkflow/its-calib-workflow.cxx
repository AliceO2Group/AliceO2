// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  workflowOptions.push_back(
    ConfigParamSpec{
      "doNoise",
      VariantType::Bool,
      true,
      {"Generate noisy-pixel maps"}});
  workflowOptions.push_back(
    ConfigParamSpec{
      "configKeyValues",
      VariantType::String,
      "",
      {"Semicolon separated key=value strings"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"
#include "ITSCalibration/NoiseCalibratorSpec.h"
#include "ITSCalibration/NoiseCalibrator.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  auto doNoise = configcontext.options().get<bool>("doNoise");

  LOG(INFO) << "ITS calibration workflow options";
  LOG(INFO) << "Generate noisy-pixel maps: " << doNoise;

  if (doNoise) {
    specs.emplace_back(o2::its::getNoiseCalibratorSpec());
  }

  return specs;
}
