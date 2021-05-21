// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
#include "Framework/Variant.h"

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
  policies.push_back({"decoded-hmpid-digits", readerMatcher, DispatchOp::WhenReady, triggerMatcher});
}

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  using o2::framework::CompletionPolicyHelpers;
  policies.push_back(CompletionPolicyHelpers::defineByName("raw-hmpid-decode", CompletionPolicy::CompletionOp::Consume));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::string keyvaluehelp("Semicolon separated key=value strings ...");
  workflowOptions.push_back(o2::framework::ConfigParamSpec{"configKeyValues", o2::framework::VariantType::String, "", {keyvaluehelp}});
  workflowOptions.push_back(o2::framework::ConfigParamSpec{"ignore-dist-stf", o2::framework::VariantType::Bool, false, {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}});
}

#include "Framework/runDataProcessing.h"
#include "HMPIDWorkflow/DataDecoderSpec2.h"

using namespace o2;
using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext& cx)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(cx.options().get<std::string>("configKeyValues"));
  auto askSTFDist = !cx.options().get<bool>("ignore-dist-stf");
  DataProcessorSpec producer = o2::hmpid::getDecodingSpec2(askSTFDist);
  specs.push_back(producer);
  return specs;
}
