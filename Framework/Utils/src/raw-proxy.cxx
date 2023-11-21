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

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include <vector>

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(
    ConfigParamSpec{
      "proxy-name", VariantType::String, "readout-proxy", {"name of the proxy processor"}});

  workflowOptions.push_back(
    ConfigParamSpec{
      "dataspec", VariantType::String, "A:FLP/RAWDATA;B:FLP/DISTSUBTIMEFRAME/0", {"selection string for the data to be proxied"}});

  workflowOptions.push_back(
    ConfigParamSpec{
      "inject-missing-data", VariantType::Bool, false, {"inject missing data according to dataspec if not found in the input"}});

  workflowOptions.push_back(
    ConfigParamSpec{
      "sporadic-outputs", VariantType::Bool, false, {"consider all the outputs as sporadic"}});

  workflowOptions.push_back(
    ConfigParamSpec{
      "print-input-sizes", VariantType::Int, 0, {"print statistics about sizes per input spec every n TFs"}});

  workflowOptions.push_back(
    ConfigParamSpec{
      "throwOnUnmatched", VariantType::Bool, false, {"throw if unmatched input data is found"}});

  workflowOptions.push_back(
    ConfigParamSpec{
      "timeframes-shm-limit", VariantType::String, "0", {"Minimum amount of SHM required in order to publish data"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  auto processorName = config.options().get<std::string>("proxy-name");
  auto outputconfig = config.options().get<std::string>("dataspec");
  bool injectMissingData = config.options().get<bool>("inject-missing-data");
  bool sporadicOutputs = config.options().get<bool>("sporadic-outputs");
  auto printSizes = config.options().get<unsigned int>("print-input-sizes");
  bool throwOnUnmatched = config.options().get<bool>("throwOnUnmatched");
  uint64_t minSHM = std::stoul(config.options().get<std::string>("timeframes-shm-limit"));
  std::vector<InputSpec> matchers = select(outputconfig.c_str());
  Outputs readoutProxyOutput;
  for (auto const& matcher : matchers) {
    readoutProxyOutput.emplace_back(DataSpecUtils::asOutputSpec(matcher));
    readoutProxyOutput.back().lifetime = sporadicOutputs ? Lifetime::Sporadic : Lifetime::Timeframe;
  }

  // we use the same specs as filters in the dpl adaptor
  auto filterSpecs = readoutProxyOutput;
  DataProcessorSpec readoutProxy = specifyExternalFairMQDeviceProxy(
    processorName.c_str(),
    std::move(readoutProxyOutput),
    "type=pair,method=connect,address=ipc:///tmp/readout-pipe-0,rateLogging=1,transport=shmem",
    dplModelAdaptor(filterSpecs, throwOnUnmatched), minSHM, false, injectMissingData, printSizes);

  WorkflowSpec workflow;
  workflow.emplace_back(readoutProxy);
  return workflow;
}
