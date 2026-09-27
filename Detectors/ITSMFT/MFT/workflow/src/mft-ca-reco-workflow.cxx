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

/// @file mft-ca-reco-workflow.cxx

#include "MFTWorkflow/CARecoWorkflow.h"

#include <string>

#include "CommonUtils/ConfigurableParam.h"
#include "DataFormatsITSMFT/DPLAlpideParamInitializer.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "ITSMFTTracking/TrackingConfigParam.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:MFT|mft).*[W,w]riter.*"));
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"digits-from-upstream", o2::framework::VariantType::Bool, false, {"digits will be provided from upstream, skip digits reader"}},
    {"clusters-from-upstream", o2::framework::VariantType::Bool, false, {"clusters will be provided from upstream, skip clusterizer"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"do not write output root files"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"disable-tracking", o2::framework::VariantType::Bool, false, {"disable tracking step"}},
    {"run-assessment", o2::framework::VariantType::Bool, false, {"run MFT assessment workflow"}},
    {"disable-process-gen", o2::framework::VariantType::Bool, false, {"disable processing of all generated tracks (depends on --run-assessment)"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"nThreads", VariantType::Int, 1, {"Number of CA tracker threads"}},
    {"use-geom", VariantType::Bool, false, {"alias for --use-full-geometry"}},
    {"use-full-geometry", o2::framework::VariantType::Bool, false, {"use full geometry instead of the light-weight MFT part"}},
    {"use-irframes", o2::framework::VariantType::Bool, false, {"consume ITS IR frames"}},
    {"tracking-mode", VariantType::String, "sync", {"sync,async,cosmics,unset,off; async uses 3 passes by default (MFTCATrackerParam.nIterations=-1); set nIterations=1 to retain one pass"}},
    {"run-tracks2records", o2::framework::VariantType::Bool, false, {"run MFT alignment tracks to records workflow"}},
    {"cluster-rof-branch-only", o2::framework::VariantType::Bool, false, {"writer will store only ClustersROF branch"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  o2::itsmft::DPLAlpideParamInitializer::addMFTConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configContext)
{
  const auto options = o2::mft::ca::readWorkflowOptions(configContext, o2::mft::ca::WorkflowKind::Reconstruction);
  auto workflow = o2::mft::ca_reco_workflow::getWorkflow(options);
  o2::conf::ConfigurableParam::writeINI("o2mftcarecoflow_configuration.ini");

  o2::raw::HBFUtilsInitializer hbfInitializer(configContext, workflow);
  return workflow;
}
