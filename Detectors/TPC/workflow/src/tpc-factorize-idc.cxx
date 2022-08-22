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
#include "TPCWorkflow/TPCFactorizeIDCSpec.h"
#include "TPCCalibration/IDCFactorization.h"
#include "TPCCalibration/IDCAverageGroup.h"
#include "Framework/CompletionPolicyHelpers.h"

using namespace o2::framework;

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  policies.push_back(CompletionPolicyHelpers::defineByName("tpc-factorize-idc.*", CompletionPolicy::CompletionOp::Consume));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  const std::string cruDefault = "0-" + std::to_string(o2::tpc::CRU::MaxCRU - 1);

  std::vector<ConfigParamSpec> options{
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}},
    {"timeframes", VariantType::Int, 2000, {"Number of TFs which will be aggregated per aggregation interval."}},
    {"timeframesDeltaIDC", VariantType::Int, 100, {"Number of TFs used for storing the IDCDelta struct in the CCDB."}},
    {"nthreads-IDC-factorization", VariantType::Int, 1, {"Number of threads which will be used during the factorization of the IDCs."}},
    {"nthreads-grouping", VariantType::Int, 1, {"Number of threads which will be used during the grouping of IDCDelta."}},
    {"sendOutputFFT", VariantType::Bool, false, {"sending the output for fourier transform device"}},
    {"crus", VariantType::String, cruDefault.c_str(), {"List of CRUs, comma separated ranges, e.g. 0-3,7,9-15"}},
    {"compression", VariantType::Int, 2, {"compression of DeltaIDC: 0 -> No, 1 -> Medium (data compression ratio 2), 2 -> High (data compression ratio ~6)"}},
    {"input-lanes", VariantType::Int, 2, {"Number of parallel pipelines which were set in the TPCDistributeIDCSpec device."}},
    {"groupPads", VariantType::String, "5,6,7,8,4,5,6,8,10,13", {"number of pads in a row which will be grouped per region"}},
    {"groupRows", VariantType::String, "2,2,2,3,3,3,2,2,2,2", {"number of pads in row direction which will be grouped per region"}},
    {"groupLastRowsThreshold", VariantType::String, "1", {"set threshold in row direction for merging the last group to the previous group per region"}},
    {"groupLastPadsThreshold", VariantType::String, "1", {"set threshold in pad direction for merging the last group to the previous group per region"}},
    {"use-precise-timestamp", VariantType::Bool, false, {"Use precise timestamp from distribute when writing to CCDB"}},
    {"enable-CCDB-output", VariantType::Bool, false, {"send output for ccdb populator"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings (e.g. for pp 50kHz: 'TPCIDCCompressionParam.maxIDCDeltaValue=15;')"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using namespace o2::tpc;

  const std::string sgroupPads = config.options().get<std::string>("groupPads");
  const std::string sgroupRows = config.options().get<std::string>("groupRows");
  const std::string sgroupLastRowsThreshold = config.options().get<std::string>("groupLastRowsThreshold");
  const std::string sgroupLastPadsThreshold = config.options().get<std::string>("groupLastPadsThreshold");
  ParameterIDCGroup::setGroupingParameterFromString(sgroupPads, sgroupRows, sgroupLastRowsThreshold, sgroupLastPadsThreshold);

  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(config.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpcfactorizeidc_configuration.ini");

  const auto tpcCRUs = o2::RangeTokenizer::tokenize<int>(config.options().get<std::string>("crus"));
  const auto nCRUs = tpcCRUs.size();
  const auto timeframes = static_cast<unsigned int>(config.options().get<int>("timeframes"));
  const auto timeframesDeltaIDC = static_cast<unsigned int>(config.options().get<int>("timeframesDeltaIDC"));
  const auto sendOutputFFT = config.options().get<bool>("sendOutputFFT");
  const auto nthreadsFactorization = static_cast<unsigned long>(config.options().get<int>("nthreads-IDC-factorization"));
  IDCFactorization::setNThreads(nthreadsFactorization);
  const auto nthreadsGrouping = static_cast<unsigned long>(config.options().get<int>("nthreads-grouping"));
  IDCAverageGroup<IDCAverageGroupTPC>::setNThreads(nthreadsGrouping);
  const auto nLanes = static_cast<unsigned int>(config.options().get<int>("input-lanes"));
  const bool usePrecisetimeStamp = config.options().get<bool>("use-precise-timestamp");
  const bool sendCCDB = config.options().get<bool>("enable-CCDB-output");

  const int compressionTmp = config.options().get<int>("compression");
  IDCDeltaCompression compression;
  switch (compressionTmp) {
    case static_cast<int>(IDCDeltaCompression::NO):
    case static_cast<int>(IDCDeltaCompression::MEDIUM):
    case static_cast<int>(IDCDeltaCompression::HIGH):
      compression = static_cast<IDCDeltaCompression>(compressionTmp);
      break;
    default:
      LOGP(error, "wrong compression type set. Setting compression to medium compression");
      compression = static_cast<IDCDeltaCompression>(IDCDeltaCompression::MEDIUM);
      break;
  }

  const auto first = tpcCRUs.begin();
  const auto last = std::min(tpcCRUs.end(), first + nCRUs);
  const std::vector<uint32_t> rangeCRUs(first, last);

  WorkflowSpec workflow;
  workflow.reserve(nLanes);
  for (int ilane = 0; ilane < nLanes; ++ilane) {
    workflow.emplace_back(getTPCFactorizeIDCSpec(ilane, rangeCRUs, timeframes, timeframesDeltaIDC, compression, usePrecisetimeStamp, sendOutputFFT, sendCCDB));
  }
  return workflow;
}
