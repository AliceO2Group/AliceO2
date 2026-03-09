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
#include "TPCWorkflow/TPCFactorizeCMVSpec.h"
#include "Framework/CompletionPolicyHelpers.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  policies.push_back(CompletionPolicyHelpers::defineByName(
    "tpc-factorize-cmv-*.*", CompletionPolicy::CompletionOp::Consume));
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  const std::string cruDefault = "0-" + std::to_string(o2::tpc::CRU::MaxCRU - 1);

  std::vector<ConfigParamSpec> options{
    {"configFile", VariantType::String, "", {"Configuration file for configurable parameters"}},
    {"timeframes", VariantType::Int, 2000, {"Number of TFs aggregated per calibration interval"}},
    {"crus", VariantType::String, cruDefault.c_str(), {"List of CRUs, comma-separated ranges, e.g. 0-3,7,9-15"}},
    {"input-lanes", VariantType::Int, 2, {"Number of parallel pipelines set in the TPCDistributeCMVSpec device"}},
    {"use-precise-timestamp", VariantType::Bool, false, {"Use precise timestamp from distributor when writing to CCDB"}},
    {"enable-CCDB-output", VariantType::Bool, false, {"Send output to the CCDB populator"}},
    {"n-TFs-buffer", VariantType::Int, 1, {"Buffer size that was set in TPCFLPCMVSpec"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon-separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using namespace o2::tpc;

  o2::conf::ConfigurableParam::updateFromFile(config.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpcfactorizecmv_configuration.ini");

  const auto tpcCRUs = o2::RangeTokenizer::tokenize<int>(config.options().get<std::string>("crus"));

  auto timeframes = static_cast<unsigned int>(config.options().get<int>("timeframes"));
  const auto nLanes = static_cast<unsigned int>(config.options().get<int>("input-lanes"));
  const bool usePreciseTimestamp = config.options().get<bool>("use-precise-timestamp");
  const bool sendCCDB = config.options().get<bool>("enable-CCDB-output");

  int nTFsBuffer = config.options().get<int>("n-TFs-buffer");
  if (nTFsBuffer <= 0) {
    nTFsBuffer = 1;
  }

  assert(timeframes >= static_cast<unsigned int>(nTFsBuffer));
  timeframes /= static_cast<unsigned int>(nTFsBuffer);

  const std::vector<uint32_t> rangeCRUs(tpcCRUs.begin(), tpcCRUs.end());

  WorkflowSpec workflow;
  workflow.reserve(nLanes);
  for (int ilane = 0; ilane < static_cast<int>(nLanes); ++ilane) {
    workflow.emplace_back(getTPCFactorizeCMVSpec(
      ilane,
      rangeCRUs,
      timeframes,
      sendCCDB,
      usePreciseTimestamp,
      nTFsBuffer));
  }
  return workflow;
}