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

/// @file   gpu-reco-workflow.cxx
/// @author David Rohr

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/ConcreteDataMatcher.h"
#include "TPCReaderWorkflow/TPCSectorCompletionPolicy.h"
#include "Framework/CustomWorkflowTerminationHook.h"
#include "GPUWorkflow/GPUWorkflowSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/CallbacksPolicy.h"
#include "TPCBase/Sector.h"
#include "Algorithm/RangeTokenizer.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

#include <unordered_map>
#include <numeric>

using namespace o2::framework;
using namespace o2::dataformats;
using namespace o2::gpu;
using CompletionPolicyData = std::vector<InputSpec>;
CompletionPolicyData gPolicyData;
static constexpr unsigned long gTpcSectorMask = 0xFFFFFFFFF;
static std::shared_ptr<GPURecoWorkflowSpec> gTask;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{

  std::vector<ConfigParamSpec> options{
    {"input-type", VariantType::String, "digits", {"digitizer, digits, zsraw, zsonthefly, clustersnative, compressed-clusters-root, compressed-clusters-ctf, trd-tracklets"}},
    {"output-type", VariantType::String, "tracks", {"clustersnative, tracks, compressed-clusters-ctf, qa, no-shared-cluster-map, send-clusters-per-sector, trd-tracks"}},
    {"disable-root-input", VariantType::Bool, true, {"disable root-files input reader"}},
    {"disable-mc", VariantType::Bool, false, {"disable sending of MC information"}},
    {"ignore-dist-stf", VariantType::Bool, false, {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings (e.g.: 'TPCHwClusterer.peakChargeThreshold=4;...')"}},
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// customize dispatch policy, dispatch immediately what is ready
void customize(std::vector<DispatchPolicy>& policies)
{
  using DispatchOp = DispatchPolicy::DispatchOp;
  policies.push_back({"prompt-for-gpu-reco", [](auto const& spec) { return true; }, DispatchOp::WhenReady});
}

void customize(std::vector<CompletionPolicy>& policies)
{
  policies.push_back(o2::tpc::TPCSectorCompletionPolicy("gpu-reconstruction.*", o2::tpc::TPCSectorCompletionPolicy::Config::RequireAll, &gPolicyData, &gTpcSectorMask)());
}

void customize(o2::framework::OnWorkflowTerminationHook& hook)
{
  hook = [](const char* idstring) {
    if (gTask) {
      gTask->deinitialize();
    }
  };
}

#include "Framework/runDataProcessing.h" // the main driver

using namespace o2::framework;

enum struct ioType { Digits,
                     Clusters,
                     ZSRaw,
                     ZSRawOTF,
                     CompClustROOT,
                     CompClustCTF,
                     Tracks,
                     QA,
                     TRDTracklets,
                     TRDTracks,
                     NoSharedMap,
                     SendClustersPerSector };

static const std::unordered_map<std::string, ioType> InputMap{
  {"digits", ioType::Digits},
  {"clusters", ioType::Clusters},
  {"zsraw", ioType::ZSRaw},
  {"zsonthefly", ioType::ZSRawOTF},
  {"compressed-clusters-root", ioType::CompClustROOT},
  {"compressed-clusters-ctf", ioType::CompClustCTF},
  {"trd-tracklets", ioType::TRDTracklets}};

static const std::unordered_map<std::string, ioType> OutputMap{
  {"clusters", ioType::Clusters},
  {"tracks", ioType::Tracks},
  {"compressed-clusters-ctf", ioType::CompClustCTF},
  {"qa", ioType::QA},
  {"no-shared-cluster-map", ioType::NoSharedMap},
  {"send-clusters-per-sector", ioType::SendClustersPerSector},
  {"trd-tracks", ioType::TRDTracks}};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;
  std::vector<int> tpcSectors(o2::tpc::Sector::MAXSECTOR);
  std::iota(tpcSectors.begin(), tpcSectors.end(), 0);

  auto inputType = cfgc.options().get<std::string>("input-type");
  bool doMC = !cfgc.options().get<bool>("disable-mc");

  o2::conf::ConfigurableParam::updateFromFile(cfgc.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2gpurecoworkflow_configuration.ini");

  std::vector<ioType> outputTypes, inputTypes;
  try {
    outputTypes = o2::RangeTokenizer::tokenize<ioType>(cfgc.options().get<std::string>("output-type"), [](std::string const& token) { return OutputMap.at(token); });
    inputTypes = o2::RangeTokenizer::tokenize<ioType>(cfgc.options().get<std::string>("input-type"), [](std::string const& token) { return InputMap.at(token); });
  } catch (std::out_of_range&) {
    throw std::invalid_argument("invalid input / output type");
  }

  auto isEnabled = [](auto& list, ioType type) {
    return std::find(list.begin(), list.end(), type) != list.end();
  };

  GPURecoWorkflowSpec::Config cfg;
  cfg.decompressTPC = isEnabled(inputTypes, ioType::CompClustCTF);
  cfg.decompressTPCFromROOT = isEnabled(inputTypes, ioType::CompClustROOT);
  cfg.zsDecoder = isEnabled(inputTypes, ioType::ZSRaw);
  cfg.zsOnTheFly = isEnabled(inputTypes, ioType::ZSRawOTF);
  cfg.caClusterer = cfg.zsDecoder || cfg.zsOnTheFly || isEnabled(inputTypes, ioType::Digits);
  cfg.outputTracks = isEnabled(outputTypes, ioType::Tracks);
  cfg.outputCompClusters = isEnabled(outputTypes, ioType::CompClustROOT);
  cfg.outputCompClustersFlat = isEnabled(outputTypes, ioType::CompClustCTF);
  cfg.outputCAClusters = isEnabled(outputTypes, ioType::Clusters);
  cfg.outputQA = isEnabled(outputTypes, ioType::QA);
  cfg.outputSharedClusterMap = (cfg.outputCAClusters || cfg.caClusterer || isEnabled(inputTypes, ioType::Clusters)) && cfg.outputTracks && !isEnabled(outputTypes, ioType::NoSharedMap);
  cfg.processMC = doMC;
  cfg.sendClustersPerSector = isEnabled(outputTypes, ioType::SendClustersPerSector);
  cfg.askDISTSTF = !cfgc.options().get<bool>("ignore-dist-stf");
  cfg.readTRDtracklets = isEnabled(inputTypes, ioType::TRDTracklets);
  cfg.runTRDTracking = isEnabled(outputTypes, ioType::TRDTracks);

  Inputs ggInputs;
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false, true, false, true, true, o2::base::GRPGeomRequest::Aligned, ggInputs, true);

  auto task = std::make_shared<GPURecoWorkflowSpec>(&gPolicyData, cfg, tpcSectors, gTpcSectorMask, ggRequest);
  Inputs taskInputs = task->inputs();
  std::move(ggInputs.begin(), ggInputs.end(), std::back_inserter(taskInputs));
  gTask = task;

  specs.emplace_back(DataProcessorSpec{
    "gpu-reconstruction",
    taskInputs,
    task->outputs(),
    AlgorithmSpec{adoptTask<GPURecoWorkflowSpec>(task)}});

  if (!cfgc.options().get<bool>("ignore-dist-stf")) {
    GlobalTrackID::mask_t srcTrk = GlobalTrackID::getSourcesMask("none");
    GlobalTrackID::mask_t srcCl = GlobalTrackID::getSourcesMask("TPC");
    o2::globaltracking::InputHelper::addInputSpecs(cfgc, specs, srcCl, srcTrk, srcTrk, doMC);
  }

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);

  return specs;
}
