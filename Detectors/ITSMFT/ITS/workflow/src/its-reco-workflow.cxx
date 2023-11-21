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

#include "ITSWorkflow/RecoWorkflow.h"
#include "CommonUtils/ConfigurableParam.h"
#include "ITStracking/TrackingConfigParam.h"
#include "ITStracking/Configuration.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/ConfigContext.h"
#include "Framework/CompletionPolicyHelpers.h"

#include <vector>

using namespace o2::framework;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:ITS|its).*[W,w]riter.*"));
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*irframe-writer.*"));
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"digits-from-upstream", o2::framework::VariantType::Bool, false, {"digits will be provided from upstream, skip digits reader"}},
    {"clusters-from-upstream", o2::framework::VariantType::Bool, false, {"clusters will be provided from upstream, skip clusterizer"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"do not write output root files"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"trackerCA", o2::framework::VariantType::Bool, false, {"use trackerCA (default: trackerCM)"}},
    {"ccdb-meanvertex-seed", o2::framework::VariantType::Bool, false, {"use MeanVertex from CCDB if available to provide beam position seed (default: false)"}},
    {"select-with-triggers", o2::framework::VariantType::String, "none", {"use triggers to prescale processed ROFs: phys, trd, none"}},
    {"tracking-mode", o2::framework::VariantType::String, "sync", {"sync,async,cosmics"}},
    {"disable-tracking", o2::framework::VariantType::Bool, false, {"disable tracking step"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"use-gpu-workflow", o2::framework::VariantType::Bool, false, {"use GPU workflow (default: false)"}},
    {"gpu-device", o2::framework::VariantType::Int, 1, {"use gpu device: CPU=1,CUDA=2,HIP=3 (default: CPU)"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto beamPosOVerride = configcontext.options().get<bool>("ccdb-meanvertex-seed");
  auto useCAtracker = configcontext.options().get<bool>("trackerCA");
  auto trmode = configcontext.options().get<std::string>("tracking-mode");
  auto selTrig = configcontext.options().get<std::string>("select-with-triggers");
  auto useGpuWF = configcontext.options().get<bool>("use-gpu-workflow");
  auto gpuDevice = static_cast<o2::gpu::GPUDataTypes::DeviceType>(configcontext.options().get<int>("gpu-device"));
  auto extDigits = configcontext.options().get<bool>("digits-from-upstream");
  auto extClusters = configcontext.options().get<bool>("clusters-from-upstream");
  auto disableRootOutput = configcontext.options().get<bool>("disable-root-output");
  if (configcontext.options().get<bool>("disable-tracking")) {
    trmode = "";
  }
  std::transform(trmode.begin(), trmode.end(), trmode.begin(), [](unsigned char c) { return std::tolower(c); });

  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  int trType = 0;
  if (!selTrig.empty() && selTrig != "none") {
    if (selTrig == "phys") {
      trType = 1;
    } else if (selTrig == "trd") {
      trType = 2;
    } else {
      LOG(fatal) << "Unknown trigger type requested for events prescaling: " << selTrig;
    }
  }
  auto wf = o2::its::reco_workflow::getWorkflow(useMC,
                                                useCAtracker,
                                                trmode,
                                                beamPosOVerride,
                                                extDigits,
                                                extClusters,
                                                disableRootOutput,
                                                trType,
                                                useGpuWF,
                                                gpuDevice);

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, wf);

  // write the configuration used for the reco workflow
  o2::conf::ConfigurableParam::writeINI("o2itsrecoflow_configuration.ini");

  return std::move(wf);
}
