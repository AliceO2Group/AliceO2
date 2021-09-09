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

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"

#include "TPCWorkflow/ApplyCCDBCalibSpec.h"

using namespace o2::framework;

//// customize the completion policy
//void customize(std::vector<o2::framework::CompletionPolicy>& policies)
//{
//using o2::framework::CompletionPolicy;
//policies.push_back(CompletionPolicyHelpers::defineByName("tpc-apply-ccdb-calib", CompletionPolicy::CompletionOp::Consume));
//}

//// we need to add workflow options before including Framework/runDataProcessing
//void customize(std::vector<ConfigParamSpec>& workflowOptions)
//{
//std::vector<ConfigParamSpec> options{
////{"enable-writer", VariantType::Bool, false, {"selection string input specs"}},
//};

//std::swap(workflowOptions, options);
//}
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{

  using namespace o2::tpc;

  WorkflowSpec workflow;
  workflow.emplace_back(getApplyCCDBCalibSpec());

  return workflow;
}
