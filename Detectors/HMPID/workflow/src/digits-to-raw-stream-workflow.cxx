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

/// \file   write-raw-from-digits-workflow.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 01 feb 2021
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

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  using o2::framework::CompletionPolicyHelpers;
  policies.push_back(CompletionPolicyHelpers::defineByName("digit-hmpid-write", CompletionPolicy::CompletionOp::Consume));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::string keyvaluehelp("Semicolon separated key=value strings ...");
  workflowOptions.push_back(o2::framework::ConfigParamSpec{"configKeyValues", o2::framework::VariantType::String, "", {keyvaluehelp}});
}

#include "Framework/runDataProcessing.h"
#include "../include/HMPIDWorkflow/WriteRawFileSpec.h"

using namespace o2;
using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext& cx)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(cx.options().get<std::string>("configKeyValues"));
  DataProcessorSpec consumer = o2::hmpid::getWriteRawFileSpec();
  specs.push_back(consumer);
  return specs;
}
