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

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ChannelSpecHelpers.h"
#include "GlobalTrackingWorkflow/ReaderDriverSpec.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"

using namespace o2::framework;

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"timeframes-shm-limit", VariantType::String, "0", {"Minimum amount of SHM required in order to publish data"}},
    {"metric-feedback-channel-format", VariantType::String, "name=metric-feedback,type=pull,method=connect,address=ipc://{}metric-feedback-{},transport=shmem,rateLogging=0", {"format for the metric-feedback channel for TF rate limiting"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options, "o2_tfidinfo.root,upstream");
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  int rateLimitingIPCID = std::stoi(configcontext.options().get<std::string>("timeframes-rate-limit-ipcid"));
  std::string chanFmt = configcontext.options().get<std::string>("metric-feedback-channel-format");
  std::string metricChannel{};
  if (rateLimitingIPCID > -1 && !chanFmt.empty()) {
    metricChannel = fmt::format(fmt::runtime(chanFmt), o2::framework::ChannelSpecHelpers::defaultIPCFolder(), rateLimitingIPCID);
  }
  size_t minSHM = std::stoul(configcontext.options().get<std::string>("timeframes-shm-limit"));

  WorkflowSpec specs;
  specs.emplace_back(o2::globaltracking::getReaderDriverSpec(metricChannel, minSHM));

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}
