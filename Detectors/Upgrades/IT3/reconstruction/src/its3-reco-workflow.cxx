// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITS3Reconstruction/RecoWorkflow.h"
#include "CommonUtils/ConfigurableParam.h"
#include "ITStracking/TrackingConfigParam.h"
#include "ITStracking/Configuration.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"

#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainITS.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"digits-from-upstream", o2::framework::VariantType::Bool, false, {"digits will be provided from upstream, skip digits reader"}},
    {"clusters-from-upstream", o2::framework::VariantType::Bool, false, {"clusters will be provided from upstream, skip clusterizer"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"do not write output root files"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"tracking-mode", o2::framework::VariantType::String, "sync", {"sync,async,cosmics"}},
    {"entropy-encoding", o2::framework::VariantType::Bool, false, {"produce entropy encoded data"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"gpuDevice", o2::framework::VariantType::Int, 1, {"use gpu device: CPU=1,CUDA=2,HIP=3 (default: CPU)"}}};

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
  auto trmode = configcontext.options().get<std::string>("tracking-mode");
  auto gpuDevice = static_cast<o2::gpu::GPUDataTypes::DeviceType>(configcontext.options().get<int>("gpuDevice"));
  auto extDigits = configcontext.options().get<bool>("digits-from-upstream");
  auto extClusters = configcontext.options().get<bool>("clusters-from-upstream");
  auto disableRootOutput = configcontext.options().get<bool>("disable-root-output");
  auto eencode = configcontext.options().get<bool>("entropy-encoding");

  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  auto wf = o2::its3::reco_workflow::getWorkflow(useMC, trmode, gpuDevice, extDigits, extClusters, disableRootOutput, eencode);

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, wf);

  // write the configuration used for the reco workflow
  o2::conf::ConfigurableParam::writeINI("o2its3recoflow_configuration.ini");

  return std::move(wf);
}
