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
#include "TPCWorkflow/TPCFLPIDCSpec.h"
#include "TPCBase/CRU.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  const std::string cruDefault = "0-" + std::to_string(o2::tpc::CRU::MaxCRU - 1);
  const int defaultlanes = std::max(1u, std::thread::hardware_concurrency() / 2);

  std::vector<ConfigParamSpec> options{
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}},
    {"loadStatusMap", VariantType::Bool, false, {"Loading pad status map from the CCDB."}},
    {"lanes", VariantType::Int, defaultlanes, {"Number of parallel processing lanes (crus are split per device)."}},
    {"time-lanes", VariantType::Int, 1, {"Number of parallel processing lanes (timeframes are split per device)."}},
    {"crus", VariantType::String, cruDefault.c_str(), {"List of CRUs, comma separated ranges, e.g. 0-3,7,9-15"}},
    {"rangeIDC", VariantType::Int, 200, {"Number of 1D-IDCs which will be used for the calculation of the fourier coefficients."}},
    {"minIDCsPerTF", VariantType::Int, 10, {"minimum number of IDCs per TF (needed for sending to 1D-IDCs to EPNs. Depends on number of orbits per TF. 10 for 128 orbits per TF)."}},
    {"idc0File", VariantType::String, "", {"file to reference IDC0 object"}},
    {"disableIDC0CCDB", VariantType::Bool, false, {"Disabling loading the IDC0 object from the CCDB (no normalization is applied for IDC1 calculation)"}},
    {"enable-synchronous-processing", VariantType::Bool, false, {"Enable calculation and sending of 1D-IDCs for synchronous processing"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings (e.g. 'TPCIDCGroupParam.Method=0;')"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using namespace o2::tpc;
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  const auto tpcCRUs = o2::RangeTokenizer::tokenize<int>(config.options().get<std::string>("crus"));
  const auto nCRUs = tpcCRUs.size();
  const auto nLanes = std::min(static_cast<unsigned long>(config.options().get<int>("lanes")), nCRUs);
  const auto time_lanes = static_cast<unsigned int>(config.options().get<int>("time-lanes"));
  const auto crusPerLane = nCRUs / nLanes + ((nCRUs % nLanes) != 0);
  const auto loadStatusMap = config.options().get<bool>("loadStatusMap");
  const auto rangeIDC = static_cast<unsigned int>(config.options().get<int>("rangeIDC"));
  const auto minIDCsPerTF = static_cast<unsigned int>(config.options().get<int>("minIDCsPerTF"));
  TPCFLPIDCDevice::setMinIDCsPerTF(minIDCsPerTF);

  const std::string idc0File = config.options().get<std::string>("idc0File");
  const auto disableIDC0CCDB = config.options().get<bool>("disableIDC0CCDB");
  const auto enableSynchProc = config.options().get<bool>("enable-synchronous-processing");

  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(config.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::writeINI("o2tpcflp_configuration.ini");

  WorkflowSpec workflow;
  if (nLanes <= 0) {
    return workflow;
  }

  for (int ilane = 0; ilane < nLanes; ++ilane) {
    const auto first = tpcCRUs.begin() + ilane * crusPerLane;
    if (first >= tpcCRUs.end()) {
      break;
    }
    const auto last = std::min(tpcCRUs.end(), first + crusPerLane);
    const std::vector<uint32_t> rangeCRUs(first, last);
    workflow.emplace_back(timePipeline(getTPCFLPIDCSpec(ilane, rangeCRUs, rangeIDC, loadStatusMap, idc0File, disableIDC0CCDB, enableSynchProc), time_lanes));
  }

  return workflow;
}
