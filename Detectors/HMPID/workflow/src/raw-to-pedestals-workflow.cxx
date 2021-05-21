// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   raw-to-pedestals-workflow.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 01 feb 2021
///

#include "Framework/Variant.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamSpec.h"
#include "CommonUtils/ConfigurableParam.h"

// customize dispatch policy, dispatch immediately what is ready
void customize(std::vector<o2::framework::DispatchPolicy>& policies)
{
  using DispatchOp = o2::framework::DispatchPolicy::DispatchOp;
  auto readerMatcher = [](auto const& spec) {
    return true;
  };
  auto triggerMatcher = [](auto const& query) {
    return true;
  };
  policies.push_back({"raw-hmpid-pedestals", readerMatcher, DispatchOp::WhenReady, triggerMatcher});
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::string keyvaluehelp("Semicolon separated key=value strings ...");
  workflowOptions.push_back(o2::framework::ConfigParamSpec{"configKeyValues", o2::framework::VariantType::String, "", {keyvaluehelp}});
}

#include "Framework/runDataProcessing.h"
#include "HMPIDWorkflow/PedestalsCalculationSpec.h"

using namespace o2;
using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext& cx)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(cx.options().get<std::string>("configKeyValues"));
  DataProcessorSpec producer = o2::hmpid::getPedestalsCalculationSpec();
  specs.push_back(producer);
  return specs;
}
