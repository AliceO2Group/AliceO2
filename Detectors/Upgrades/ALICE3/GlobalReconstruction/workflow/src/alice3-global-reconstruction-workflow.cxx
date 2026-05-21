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

#include "ALICE3GlobalReconstructionWorkflow/RecoWorkflow.h"
#include "CommonUtils/ConfigurableParam.h"

#include "Framework/CallbacksPolicy.h"
#include "Framework/ConfigContext.h"
#include "Framework/CompletionPolicyHelpers.h"

#include <stdexcept>
#include <vector>

using namespace o2::framework;

void customize(std::vector<CallbacksPolicy>& policies)
{
  // o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<CompletionPolicy>& policies)
{
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:TRK|trk).*[W,w]riter.*"));
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"disable-root-output", VariantType::Bool, false, {"do not write output root files"}},
    {"disable-mc", VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"tracking-from-hits-config", VariantType::String, "", {"JSON file with tracking from hits configuration"}},
    {"tracking-from-clusters-config", VariantType::String, "", {"JSON file with tracking from clusters configuration"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"gpu-device", VariantType::Int, 1, {"use gpu device: CPU=1,CUDA=2,HIP=3 (default: CPU)"}},
    {"tracking-threads", VariantType::Int, 1, {"number of CPU threads used by TRK tracking"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto hitRecoConfig = configcontext.options().get<std::string>("tracking-from-hits-config");
  auto clusterRecoConfig = configcontext.options().get<std::string>("tracking-from-clusters-config");
  auto gpuDevice = static_cast<o2::gpu::gpudatatypes::DeviceType>(configcontext.options().get<int>("gpu-device"));
  auto trackingThreads = configcontext.options().get<int>("tracking-threads");
  auto disableRootOutput = configcontext.options().get<bool>("disable-root-output");
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  if (hitRecoConfig.empty() && clusterRecoConfig.empty()) {
    throw std::invalid_argument("no reconstruction input configured: provide either --tracking-from-hits-config <file> or --tracking-from-clusters-config <file>");
  }

  o2::conf::ConfigurableParam::writeINI("o2alice3globalrecoflow_configuration.ini");

  return o2::trk::global_reco_workflow::getWorkflow(useMC, hitRecoConfig, clusterRecoConfig, disableRootOutput, gpuDevice, trackingThreads);
}
