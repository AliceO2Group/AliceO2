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

/// \file tracks-matcher-workflow.cxx
/// \brief Implementation of a DPL device to run the MCH-MID track matcher
///
/// \author Philippe Pillot, Subatech

#include <string>
#include <vector>
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigContext.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CallbacksPolicy.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "MCHIO/TrackReaderSpec.h"
#include "MIDWorkflow/TrackReaderSpec.h"
#include "TrackMatcherSpec.h"
#include "TrackWriterSpec.h"

using namespace o2::framework;

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:MUON|muon).*[W,w]riter.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.emplace_back("disable-root-input", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"disable root-files input reader"});
  workflowOptions.emplace_back("disable-root-output", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"do not write output root file"});
  workflowOptions.emplace_back("disable-mc", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"disable MC propagation even if available"});
  workflowOptions.emplace_back("configKeyValues", VariantType::String, "",
                               ConfigParamSpec::HelpString{"Semicolon separated key=value strings"});
  o2::raw::HBFUtilsInitializer::addConfigOption(workflowOptions);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& configcontext)
{
  WorkflowSpec specs{};

  auto disableRootOutput = configcontext.options().get<bool>("disable-root-output");
  auto disableRootInput = configcontext.options().get<bool>("disable-root-input");
  auto useMC = !configcontext.options().get<bool>("disable-mc");

  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  if (!disableRootInput) {
    specs.emplace_back(o2::mch::getTrackReaderSpec(useMC, "mch-track-reader"));
    specs.emplace_back(o2::mid::getTrackReaderSpec(useMC, "mid-track-reader"));
  }

  specs.emplace_back(o2::muon::getTrackMatcherSpec(useMC, "muon-track-matcher"));

  if (!disableRootOutput) {
    specs.emplace_back(o2::muon::getTrackWriterSpec(useMC, "muon-track-writer", "muontracks.root"));
  }

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2matchmchmid-workflow_configuration.ini");

  return specs;
}
