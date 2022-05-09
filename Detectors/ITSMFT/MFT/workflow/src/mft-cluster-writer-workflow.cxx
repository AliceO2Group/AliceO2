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

#include "MFTWorkflow/ClusterWriterSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:MFT|mft).*[W,w]riter.*"));
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(
    ConfigParamSpec{"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}});
}

#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  WorkflowSpec specs;
  specs.emplace_back(o2::mft::getClusterWriterSpec(useMC));
  return specs;
}
