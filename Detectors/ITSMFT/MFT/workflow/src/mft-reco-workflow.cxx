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

#include "MFTWorkflow/RecoWorkflow.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:MFT|mft).*[W,w]riter.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"digits-from-upstream", o2::framework::VariantType::Bool, false, {"digits will be provided from upstream, skip digits reader"}},
    {"clusters-from-upstream", o2::framework::VariantType::Bool, false, {"clusters will be provided from upstream, skip clusterizer"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"do not write output root files"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"disable-tracking", o2::framework::VariantType::Bool, false, {"disable tracking step"}},
    {"run-assessment", o2::framework::VariantType::Bool, false, {"run MFT assessment workflow"}},
    {"disable-process-gen", o2::framework::VariantType::Bool, false, {"disable processing of all generated tracks (depends on --run-assessment)"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration
  o2::conf::ConfigurableParam::writeINI("o2mftrecoflow_configuration.ini");

  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto extDigits = configcontext.options().get<bool>("digits-from-upstream");
  auto extClusters = configcontext.options().get<bool>("clusters-from-upstream");
  auto disableRootOutput = configcontext.options().get<bool>("disable-root-output");
  auto runAssessment = configcontext.options().get<bool>("run-assessment");
  auto processGen = !configcontext.options().get<bool>("disable-process-gen");
  auto runTracking = !configcontext.options().get<bool>("disable-tracking");

  auto wf = o2::mft::reco_workflow::getWorkflow(useMC, extDigits, extClusters, disableRootOutput, runAssessment, processGen, runTracking);

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, wf);

  return std::move(wf);
}
