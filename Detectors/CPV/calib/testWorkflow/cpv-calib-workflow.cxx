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
#include "PedestalCalibratorSpec.h"
#include "NoiseCalibratorSpec.h"
#include "GainCalibratorSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"pedestals", o2::framework::VariantType::Bool, false, {"do pedestal calibration"}});
  workflowOptions.push_back(ConfigParamSpec{"noise", o2::framework::VariantType::Bool, false, {"do noise scan calibration"}});
  workflowOptions.push_back(ConfigParamSpec{"gains", o2::framework::VariantType::Bool, false, {"do gain calibration"}});
  // workflowOptions.push_back(ConfigParamSpec{"badmap", o2::framework::VariantType::Bool, false, {"do bad map calibration"}});
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  auto doPedestals = configcontext.options().get<bool>("pedestals");
  auto doNoise = configcontext.options().get<bool>("noise");
  auto doGain = configcontext.options().get<bool>("gains");
  // auto doBadMap = configcontext.options().get<bool>("badmap");
  int nProcessors = (int)doPedestals + (int)doNoise + (int)doGain;
  if (nProcessors > 1) {
    LOG(error) << "Can not run several calibrations simulteneously in one executable";
    return specs;
  }
  if (doGain) {
    specs.emplace_back(getCPVGainCalibratorSpec());
  }
  //  if (doBadMap) {
  //    specs.emplace_back(getCPVBadMapCalibratorSpec());
  //  }
  if (doPedestals) {
    specs.emplace_back(getCPVPedestalCalibratorSpec());
  }
  if (doNoise) {
    specs.emplace_back(getCPVNoiseCalibratorSpec());
  }
  return specs;
}
