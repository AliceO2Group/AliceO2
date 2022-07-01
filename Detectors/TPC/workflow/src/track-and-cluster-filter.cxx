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

#include <vector>

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "TPCCalibration/TrackDump.h"
#include "TPCReaderWorkflow/TPCSectorCompletionPolicy.h"
#include "TPCWorkflow/TrackAndClusterFilterSpec.h"

using namespace o2::framework;

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  // the TPC sector completion policy checks when the set of TPC/CLUSTERNATIVE data is complete
  // in addition we require to have input from all other routes
  policies.push_back(o2::tpc::TPCSectorCompletionPolicy("tpc-track-and-cluster-filter",
                                                        o2::tpc::TPCSectorCompletionPolicy::Config::RequireAll,
                                                        InputSpec{"cluster", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}})());
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{

  std::vector<ConfigParamSpec> options{
    // {"use-digit-input", VariantType::Bool, false, {"use TPC digits as input, instead of raw data"}},
    {"data-description", VariantType::String, "TRACKS", {"Can be used to select filterd tracks, e.g. 'LASERTRACKS', 'MIPS'"}},
  };

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfg)
{
  // const bool useDigitsAsInput = cfg.options().get<bool>("use-digit-input");
  const auto dataDescription = cfg.options().get<std::string>("data-description");

  WorkflowSpec specs;

  specs.emplace_back(o2::tpc::getTrackAndClusterFilterSpec(dataDescription));

  return specs;
}
