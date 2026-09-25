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

/// @file   its-ca-tracker-workflow.cxx
/// \brief  ITS common-CA tracker workflow: tracking on ITS cluster
///         inputs with tracker-only outputs.

#include <tuple>
#include <vector>

#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/ConfigParamSpec.h"
#include "ITSCAWorkflow/CATrackerSpec.h"
#include "ITSCAWorkflow/ConfigPreflight.h"
#include "ITSMFTCAWriter/ITSCATrackWriterSpec.h"
#include "ITSMFTTracking/Configuration.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:ITS|its).*[W,w]riter.*"));
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"disable-mc", VariantType::Bool, false, {"disable MC labels"}});
  workflowOptions.push_back(ConfigParamSpec{"disable-root-output", VariantType::Bool, false, {"do not write output root files"}});
  workflowOptions.push_back(ConfigParamSpec{"vertex-source", VariantType::String, "", {"diamond or truth; alternatively select exactly one legacy vertex alias"}});
  workflowOptions.push_back(ConfigParamSpec{"truth-context", VariantType::String, "collisioncontext.root", {"digitization context for truth vertices, independent of MC output labels"}});
  workflowOptions.push_back(ConfigParamSpec{"use-full-geometry", VariantType::Bool, false, {"alias for --use-geom"}});
  workflowOptions.push_back(ConfigParamSpec{"use-geom", VariantType::Bool, false, {"use geometry from the global geometry manager"}});
  workflowOptions.push_back(ConfigParamSpec{"tracking-mode", VariantType::String, "sync", {"ITS tracking mode: 'sync' or 'async'"}});
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings (e.g. ITSCommonCATrackerParam.useDiamond=true)"}});
  o2::raw::HBFUtilsInitializer::addConfigOption(workflowOptions);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  // Help constructs device descriptions without starting tracking.
  const auto options = config.helpOnCommandLine() ? o2::its::ca::WorkflowOptions{} : o2::its::ca::readWorkflowOptions(config);
  WorkflowSpec specs;
  specs.emplace_back(o2::its::ca::getCATrackerSpec(options));
  if (options.writeRootOutput) {
    specs.emplace_back(o2::its::ca::getTrackWriterSpec(options.useMC));
  }

  o2::raw::HBFUtilsInitializer hbfIni(config, specs);

  return specs;
}
