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
#include "Algorithm/RangeTokenizer.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCWorkflow/TPCAggregateCMVSpec.h"
#include "Framework/CompletionPolicyHelpers.h"

using namespace o2::framework;

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  policies.push_back(CompletionPolicyHelpers::defineByName("tpc-aggregate-*.*", CompletionPolicy::CompletionOp::Consume));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  const std::string cruDefault = "0-" + std::to_string(o2::tpc::CRU::MaxCRU - 1);

  std::vector<ConfigParamSpec> options{
    {"configFile", VariantType::String, "", {"Configuration file for configurable parameters"}},
    {"timeframes", VariantType::Int, 2000, {"Number of TFs aggregated per calibration interval"}},
    {"crus", VariantType::String, cruDefault.c_str(), {"List of CRUs, comma-separated ranges, e.g. 0-3,7,9-15"}},
    {"input-lanes", VariantType::Int, 1, {"Number of aggregate pipelines set by --output-lanes in TPCDistributeCMVSpec"}},
    {"use-precise-timestamp", VariantType::Bool, false, {"Use precise timestamp metadata from distribute when writing to CCDB"}},
    {"enable-CCDB-output", VariantType::Bool, false, {"Send output to the CCDB populator"}},
    {"n-TFs-buffer", VariantType::Int, 1, {"Buffer size that was set in TPCFLPCMVSpec"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon-separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using namespace o2::tpc;

  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(config.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpcaggregatecmv_configuration.ini");

  const auto tpcCRUs = o2::RangeTokenizer::tokenize<int>(config.options().get<std::string>("crus"));
  auto timeframes = static_cast<unsigned int>(config.options().get<int>("timeframes"));
  int aggregateLanes = config.options().get<int>("input-lanes");
  if (aggregateLanes <= 0) {
    aggregateLanes = 1;
  }
  const bool usePreciseTimestamp = config.options().get<bool>("use-precise-timestamp");
  const bool sendCCDB = config.options().get<bool>("enable-CCDB-output");

  int nTFsBuffer = config.options().get<int>("n-TFs-buffer");
  if (nTFsBuffer <= 0) {
    nTFsBuffer = 1;
  }

  // convert total TFs per interval to number of buffered TFs
  assert(timeframes >= static_cast<unsigned int>(nTFsBuffer));
  timeframes /= static_cast<unsigned int>(nTFsBuffer);

  const std::vector<uint32_t> rangeCRUs(tpcCRUs.begin(), tpcCRUs.end());

  WorkflowSpec workflow;
  workflow.reserve(static_cast<size_t>(aggregateLanes));
  LOGP(info, "Starting CMV aggregate with {} lanes, {} timeframes, {} n-TFs-buffer", aggregateLanes, timeframes, nTFsBuffer);
  for (int ilane = 0; ilane < aggregateLanes; ++ilane) {
    workflow.emplace_back(getTPCAggregateCMVSpec(ilane, rangeCRUs, timeframes, sendCCDB, usePreciseTimestamp, nTFsBuffer));
  }
  return workflow;
}
