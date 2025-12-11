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

#include "TRKWorkflow/RecoWorkflow.h"
#include "CommonUtils/ConfigurableParam.h"
#include "ITStracking/TrackingConfigParam.h"
#include "ITStracking/Configuration.h"

#include "Framework/CallbacksPolicy.h"
#include "Framework/ConfigContext.h"
#include "Framework/CompletionPolicyHelpers.h"

#include <vector>

using namespace o2::framework;

void customize(std::vector<CallbacksPolicy>& policies)
{
  // o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:TRK|trk).*[W,w]riter.*"));
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    {"digits-from-upstream", VariantType::Bool, false, {"digits will be provided from upstream, skip digits reader"}},
    {"clusters-from-upstream", VariantType::Bool, false, {"clusters will be provided from upstream, skip clusterizer"}},
    {"disable-root-output", VariantType::Bool, false, {"do not write output root files"}},
    {"disable-mc", VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"disable-tracking", VariantType::Bool, false, {"disable tracking step"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"use-gpu-workflow", VariantType::Bool, false, {"use GPU workflow (default: false)"}},
    {"gpu-device", VariantType::Int, 1, {"use gpu device: CPU=1,CUDA=2,HIP=3 (default: CPU)"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto useGpuWF = configcontext.options().get<bool>("use-gpu-workflow");
  auto gpuDevice = static_cast<o2::gpu::gpudatatypes::DeviceType>(configcontext.options().get<int>("gpu-device"));
  auto extDigits = configcontext.options().get<bool>("digits-from-upstream");
  auto extClusters = configcontext.options().get<bool>("clusters-from-upstream");
  auto disableRootOutput = configcontext.options().get<bool>("disable-root-output");
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  // write the configuration used for the reco workflow
  o2::conf::ConfigurableParam::writeINI("o2itsrecoflow_configuration.ini");

  return o2::trk::reco_workflow::getWorkflow(useMC, extDigits, extClusters, disableRootOutput, useGpuWF, gpuDevice);
}
