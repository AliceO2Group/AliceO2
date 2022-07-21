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
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCWorkflow/TPCFactorizeSACSpec.h"
#include "TPCCalibration/SACFactorization.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  const std::string cruDefault = "0-" + std::to_string(o2::tpc::CRU::MaxCRU - 1);

  std::vector<ConfigParamSpec> options{
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}},
    {"timeframes", VariantType::Int, 2000, {"Number of TFs which will be aggregated per aggregation interval."}},
    {"nthreads-SAC-factorization", VariantType::Int, 1, {"Number of threads which will be used during the factorization of the SACs."}},
    {"debug", VariantType::Bool, false, {"create debug files"}},
    {"compression", VariantType::Int, 1, {"compression of DeltaSAC: 0 -> No, 1 -> Medium (data compression ratio 2), 2 -> High (data compression ratio ~6)"}},
    {"input-lanes", VariantType::Int, 2, {"Number of parallel pipelines which were set in the TPCDistributeSACSpec device."}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using namespace o2::tpc;

  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(config.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpcfactorizesac_configuration.ini");

  const auto timeframes = static_cast<unsigned int>(config.options().get<int>("timeframes"));
  const auto debug = config.options().get<bool>("debug");
  const auto nthreadsFactorization = static_cast<unsigned long>(config.options().get<int>("nthreads-SAC-factorization"));
  SACFactorization::setNThreads(nthreadsFactorization);
  const auto nLanes = static_cast<unsigned int>(config.options().get<int>("input-lanes"));

  const int compressionTmp = config.options().get<int>("compression");
  SACFactorization::SACDeltaCompression compression;
  switch (compressionTmp) {
    case static_cast<int>(SACFactorization::SACDeltaCompression::NO):
    case static_cast<int>(SACFactorization::SACDeltaCompression::MEDIUM):
    case static_cast<int>(SACFactorization::SACDeltaCompression::HIGH):
      compression = static_cast<SACFactorization::SACDeltaCompression>(compressionTmp);
      break;
    default:
      LOGP(error, "wrong compression type set. Setting compression to medium compression");
      compression = static_cast<SACFactorization::SACDeltaCompression>(SACFactorization::SACDeltaCompression::MEDIUM);
      break;
  }

  WorkflowSpec workflow;
  workflow.reserve(nLanes);
  for (int ilane = 0; ilane < nLanes; ++ilane) {
    workflow.emplace_back(getTPCFactorizeSACSpec(ilane, timeframes, compression, debug));
  }
  return workflow;
}
