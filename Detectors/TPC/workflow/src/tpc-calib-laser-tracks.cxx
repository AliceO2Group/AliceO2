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
#include "TPCWorkflow/CalibLaserTracksSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"input-spec", VariantType::String, "input:TPC/TRACKS", {"selection string input specs"}},
    {"use-filtered-tracks", VariantType::Bool, false, {"use prefiltered laser tracks as input"}},
  };

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  WorkflowSpec specs;
  std::string inputSpec = config.options().get<std::string>("input-spec");
  if (config.options().get<bool>("use-filtered-tracks")) {
    inputSpec = "input:TPC/LASERTRACKS";
  }
  specs.emplace_back(o2::tpc::getCalibLaserTracks(inputSpec));
  return specs;
}
