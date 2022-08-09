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
#include "TPCWorkflow/TPCDistributeIDCSpec.h"
#include "Framework/CompletionPolicyHelpers.h"

using namespace o2::framework;

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  policies.push_back(CompletionPolicyHelpers::defineByName("tpc-distribute-idc.*", CompletionPolicy::CompletionOp::Consume));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  const std::string cruDefault = "0-" + std::to_string(o2::tpc::CRU::MaxCRU - 1);

  std::vector<ConfigParamSpec> options{
    {"crus", VariantType::String, cruDefault.c_str(), {"List of CRUs, comma separated ranges, e.g. 0-3,7,9-15"}},
    {"timeframes", VariantType::Int, 2000, {"Number of TFs which will be aggregated per aggregation interval."}},
    {"firstTF", VariantType::Int, -1, {"First time frame index. (if set to -1 the first TF will be automatically detected. Values < -1 are setting an offset for skipping the first TFs)"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"lanes", VariantType::Int, 1, {"Number of lanes of this device (CRUs are split per lane)"}},
    {"send-precise-timestamp", VariantType::Bool, false, {"Send precise timestamp which can be used for writing to CCDB"}},
    {"output-lanes", VariantType::Int, 2, {"Number of parallel pipelines which will be used in the factorization device."}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using namespace o2::tpc;

  // set up configuration
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpcdistributeidc_configuration.ini");

  const auto tpcCRUs = o2::RangeTokenizer::tokenize<int>(config.options().get<std::string>("crus"));
  const auto nCRUs = tpcCRUs.size();
  const auto timeframes = static_cast<unsigned int>(config.options().get<int>("timeframes"));
  const auto outlanes = static_cast<unsigned int>(config.options().get<int>("output-lanes"));
  const auto nLanes = static_cast<unsigned int>(config.options().get<int>("lanes"));
  const auto firstTF = static_cast<unsigned int>(config.options().get<int>("firstTF"));
  const bool sendPrecisetimeStamp = config.options().get<bool>("send-precise-timestamp");

  const auto crusPerLane = nCRUs / nLanes + ((nCRUs % nLanes) != 0);
  WorkflowSpec workflow;
  for (int ilane = 0; ilane < nLanes; ++ilane) {
    const auto first = tpcCRUs.begin() + ilane * crusPerLane;
    if (first >= tpcCRUs.end()) {
      break;
    }
    const auto last = std::min(tpcCRUs.end(), first + crusPerLane);
    const std::vector<uint32_t> rangeCRUs(first, last);
    workflow.emplace_back(getTPCDistributeIDCSpec(ilane, rangeCRUs, timeframes, outlanes, firstTF, sendPrecisetimeStamp));
  }

  return workflow;
}
