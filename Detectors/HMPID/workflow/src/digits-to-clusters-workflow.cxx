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

/// \file digits-to-clusters-workflow.h
/// \brief Workflow for clusterization for HMPID; read upstream/from file write upstream/to file

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbackService.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/Task.h"
#include "Framework/Variant.h"
#include "Framework/WorkflowSpec.h"

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(o2::framework::CompletionPolicyHelpers::consumeWhenAllOrdered(".*HMPClustersWriter.*"));
}

using o2::framework::ConfigParamSpec;
using o2::framework::VariantType;
// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::string keyvaluehelp("Semicolon separated key=value strings ...");
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {keyvaluehelp}});

  workflowOptions.push_back(ConfigParamSpec{"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input readers"}});

  workflowOptions.push_back(ConfigParamSpec{"disable-root-output", VariantType::Bool, false, {"disable root-files output writers"}});

  o2::raw::HBFUtilsInitializer::addConfigOption(workflowOptions);
}

#include "Framework/runDataProcessing.h"
#include "HMPIDWorkflow/DigitsToClustersSpec.h"
#include "HMPIDWorkflow/ClustersWriterSpec.h"
#include "HMPIDWorkflow/DigitsReaderSpec.h"
// #include "HMPIDWorkflow/HMPIDDigitizerSpec.h"ss

using namespace o2;
using namespace o2::framework;

WorkflowSpec defineDataProcessing(const ConfigContext& configcontext)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(
    configcontext.options().get<std::string>("configKeyValues"));

  auto disableRootInp = configcontext.options().get<bool>("disable-root-input");  // read upstream by default
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output"); // write upstream by default

  DataProcessorSpec consumer = hmpid::getDigitsToClustersSpec();

  specs.push_back(consumer);

  // Read to File; input file and dir can be specified using
  // --hmpid-digit-infile and --input-dir (from DigitsReaderSpec Class)
  if (!disableRootInp) {
    specs.emplace_back(hmpid::getDigitsReaderSpec());
  }

  // Write to Cluster-File; output file and dir can be specified using
  // --outfile and --output-dir (from MakeTreeRootWriter Class)
  if (!disableRootOut) {
    specs.push_back(hmpid::getClusterWriterSpec());
  }

  return specs;
}
