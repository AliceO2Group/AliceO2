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
#include <string>
#include <fmt/format.h>

#include "Algorithm/RangeTokenizer.h"
#include "Framework/WorkflowSpec.h"
//#include "Framework/DataProcessorSpec.h"
//#include "Framework/DataSpecUtils.h"
#include "Framework/ControlService.h"
//#include "Framework/InputRecordWalker.h"
//#include "Framework/Logger.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCBase/CRU.h"
#include "TPCWorkflow/IDCToVectorSpec.h"

using namespace o2::framework;
using namespace o2::tpc;

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  policies.push_back(CompletionPolicyHelpers::defineByName("tpc-idc-to-vector", CompletionPolicy::CompletionOp::Consume));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::string crusDefault = "0-" + std::to_string(CRU::MaxCRU - 1);

  std::vector<ConfigParamSpec> options{
    {"input-spec", VariantType::String, "A:TPC/RAWDATA", {"selection string input specs"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings (e.g.: 'TPCCalibPedestal.FirstTimeBin=10;...')"}},
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}},
    {"crus", VariantType::String, crusDefault.c_str(), {"List of TPC crus, comma separated ranges, e.g. 0-3,7,9-15"}},
  };

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{

  using namespace o2::tpc;

  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(config.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpcidc_configuration.ini");

  const std::string inputSpec = config.options().get<std::string>("input-spec");

  const auto crus = o2::RangeTokenizer::tokenize<uint32_t>(config.options().get<std::string>("crus"));

  WorkflowSpec workflow;

  workflow.emplace_back(getIDCToVectorSpec(inputSpec, crus));

  return workflow;
}
