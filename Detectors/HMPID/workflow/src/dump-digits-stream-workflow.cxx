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

///
/// \file    dump-digits-workflow.cxx
/// \author  Antonio Franco
///
///

#include "Framework/WorkflowSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/Variant.h"
#include "CommonUtils/ConfigurableParam.h"

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(o2::framework::CompletionPolicyHelpers::consumeWhenAllOrdered("HMP-DigitsDump.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::string keyvaluehelp("Semicolon separated key=value strings ...");
  workflowOptions.push_back(o2::framework::ConfigParamSpec{"configKeyValues", o2::framework::VariantType::String, "", {keyvaluehelp}});
}

#include "Framework/runDataProcessing.h"
#include "HMPIDWorkflow/DumpDigitsSpec.h"

using namespace o2;
using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext& cx)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(cx.options().get<std::string>("configKeyValues"));
  DataProcessorSpec consumer = o2::hmpid::getDumpDigitsSpec();
  specs.push_back(consumer);
  return specs;
}
