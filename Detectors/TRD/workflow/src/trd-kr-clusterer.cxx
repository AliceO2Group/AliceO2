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
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "TRDWorkflow/KrClustererSpec.h"
#include "TRDWorkflowIO/TRDDigitReaderSpec.h"
#include "TRDWorkflowIO/KrClusterWriterSpec.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input readers"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writers"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:TRD|trd).*[W,w]riter.*"));
}
// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2trdkrcluster_workflow_configuration.ini");

  o2::framework::WorkflowSpec specs;

  // input
  if (!configcontext.options().get<bool>("disable-root-input")) {
    specs.emplace_back(o2::trd::getTRDDigitReaderSpec(false));
  }

  // processing devices
  specs.emplace_back(o2::trd::getKrClustererSpec());

  // output devices
  if (!configcontext.options().get<bool>("disable-root-output")) {
    specs.emplace_back(o2::trd::getKrClusterWriterSpec());
  }

  return specs;
}
