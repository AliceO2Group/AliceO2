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

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  const std::string cruDefault = "0-" + std::to_string(o2::tpc::CRU::MaxCRU - 1);

  std::vector<ConfigParamSpec> options{
    {"crus", VariantType::String, cruDefault.c_str(), {"List of CRUs, comma separated ranges, e.g. 0-3,7,9-15"}},
    {"timeframes", VariantType::Int, 2000, {"Number of TFs which will be aggregated per aggregation interval."}},
    {"firstTF", VariantType::Int, -1, {"First time frame index. (if set to -1 the first TF will be automatically detected. Values < -1 are setting an offset for skipping the first TFs)"}},
    {"load-from-file", VariantType::Bool, false, {"load average and grouped IDCs from IDCGroup.root file."}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
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
  const auto firstTF = static_cast<unsigned int>(config.options().get<int>("firstTF"));
  const auto loadFromFile = config.options().get<bool>("load-from-file");

  const auto first = tpcCRUs.begin();
  const auto last = std::min(tpcCRUs.end(), first + nCRUs);
  const std::vector<uint32_t> rangeCRUs(first, last);
  WorkflowSpec workflow{getTPCDistributeIDCSpec(rangeCRUs, timeframes, outlanes, firstTF, loadFromFile)};
  return workflow;
}
