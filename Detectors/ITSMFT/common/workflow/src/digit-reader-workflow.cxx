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

#include "ITSMFTWorkflow/DigitReaderSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    ConfigParamSpec{"disable-mc", VariantType::Bool, false, {"disable mc truth"}},
    ConfigParamSpec{"enable-calib-data", VariantType::Bool, false, {"enable writing GBT calibration data"}},
    ConfigParamSpec{"runmft", VariantType::Bool, false, {"expect MFT data"}},
    ConfigParamSpec{"suppress-triggers-output", VariantType::Bool, false, {"suppress dummy triggers output"}},
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec wf;
  bool useMC = !cfgc.options().get<bool>("disable-mc");
  bool calib = cfgc.options().get<bool>("enable-calib-data");
  bool withTriggers = !cfgc.options().get<bool>("suppress-triggers-output");
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  if (cfgc.options().get<bool>("runmft")) {
    wf.emplace_back(o2::itsmft::getMFTDigitReaderSpec(useMC, calib, withTriggers));
  } else {
    wf.emplace_back(o2::itsmft::getITSDigitReaderSpec(useMC, calib, withTriggers));
  }
  return wf;
}
