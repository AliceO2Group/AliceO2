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

#include "StrangenessTrackingWorkflow/StrangenessTrackingSpec.h"
#include "StrangenessTrackingWorkflow/StrangenessTrackingWriterSpec.h"

#include "CommonUtils/ConfigurableParam.h"
#include "StrangenessTracking/StrangenessTrackingConfigParam.h"

#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/ConfigContext.h"

#include <vector>

using namespace o2::framework;
using namespace o2::strangeness_tracking;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto useRootInput = !configcontext.options().get<bool>("disable-root-input");

  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  auto wf = o2::strangeness_tracking::getWorkflow(useMC, useRootInput);
  wf.emplace_back(getStrangenessTrackingWriterSpec());

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, wf);

  // write the configuration used for the reco workflow
  o2::conf::ConfigurableParam::writeINI("o2strangeness_tracking_workflow_configuration.ini");

  return std::move(wf);
}
