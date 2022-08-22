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

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "GlobalTrackingWorkflow/GlobalFwdMatchingSpec.h"
#include "GlobalTrackingWorkflow/GlobalFwdTrackWriterSpec.h"
#include "GlobalTrackingWorkflow/MatchedMFTMCHWriterSpec.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "GlobalTracking/MatchGlobalFwdParam.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:FWD|fwd).*[W,w]riter.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"do not write output root files"}},
    {"enable-match-output", o2::framework::VariantType::Bool, false, {"stores mftmch matching info on mftmchmatches.root"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2matchmftmch-workflow_configuration.ini");

  auto useMC = !configcontext.options().get<bool>("disable-mc");

  bool disableRootOutput = configcontext.options().get<bool>("disable-root-output");
  bool matchRootOutput = configcontext.options().get<bool>("enable-match-output");

  const auto& matchingParam = o2::globaltracking::GlobalFwdMatchingParam::Instance();

  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::globaltracking::getGlobalFwdMatchingSpec(useMC, matchRootOutput));
  auto srcTracks = GID::getSourcesMask("MFT,MCH");
  auto srcClusters = GID::getSourcesMask("MFT");
  auto matchMask = GID::MASK_NONE;

  if (!disableRootOutput) {
    specs.emplace_back(o2::globaltracking::getGlobalFwdTrackWriterSpec(useMC));
  }

  if (matchRootOutput && (!disableRootOutput)) {
    specs.emplace_back(o2::globaltracking::getMFTMCHMatchesWriterSpec(useMC));
  }

  if (matchingParam.isMatchUpstream()) {
    matchMask = matchMask | GID::getSourcesMask("MFT-MCH");
  }

  if (matchingParam.useMIDMatch) {
    matchMask = matchMask | GID::getSourcesMask("MCH-MID");
  }

  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, srcClusters, matchMask, srcTracks, useMC, srcClusters, srcTracks);

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}
