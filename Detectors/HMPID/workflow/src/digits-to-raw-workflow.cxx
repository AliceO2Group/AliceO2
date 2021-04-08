// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   digits-to-raw-workflow.cxx
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
#include "DetectorsCommonDataFormats/NameConf.h"

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  using o2::framework::CompletionPolicyHelpers;
  policies.push_back(o2::framework::CompletionPolicyHelpers::defineByName("digit-hmpid-write", CompletionPolicy::CompletionOp::Consume));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::string keyvaluehelp("Semicolon separated key=value strings ...");
  //workflowOptions.push_back(ConfigParamSpec{"digitization-config", o2::framework::VariantType::String, std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE), "configKeyValues file from digitization, used for raw output only!!!");
  workflowOptions.push_back(o2::framework::ConfigParamSpec{"digitization-config", o2::framework::VariantType::String, "none", {"configKeyValues file from digitization, used for raw output only!!!"}});
  workflowOptions.push_back(o2::framework::ConfigParamSpec{"configKeyValues", o2::framework::VariantType::String, "", {keyvaluehelp}});
}

#include "Framework/runDataProcessing.h"
#include "HMPIDWorkflow/DigitsToRawSpec.h"

using namespace o2;
using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext& configcontext)
{
  WorkflowSpec specs;
  std::string confDig = configcontext.options().get<std::string>("digitization-config");
  if (!confDig.empty() && confDig != "none") {
    o2::conf::ConfigurableParam::updateFromFile(confDig);
  }
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  DataProcessorSpec consumer = o2::hmpid::getDigitsToRawSpec();
  specs.push_back(consumer);
  return specs;
}
