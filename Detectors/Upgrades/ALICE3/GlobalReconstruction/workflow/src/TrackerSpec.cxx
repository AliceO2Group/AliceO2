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
#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <fstream>
#include <numeric>

#include "CommonUtils/DLLoaderBase.h"
#include "CommonDataFormat/IRFrame.h"
#include "DataFormatsTRK/Cluster.h"
#include "DataFormatsTRK/ROFRecord.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/Configuration.h"
#include "Field/MagneticField.h"
#include "Field/MagFieldParam.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "ITStracking/TrackingConfigParam.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TRKBase/GeometryTGeo.h"
#include "TRKBase/SegmentationChip.h"
#include "TRKSimulation/Hit.h"
#include "ALICE3GlobalReconstruction/TimeFrame.h"
#include "ALICE3GlobalReconstructionWorkflow/TrackerSpec.h"
#include "ALICE3GlobalReconstructionWorkflow/TrackerSpecImpl.h"
#include <TGeoGlobalMagField.h>

#ifdef O2_WITH_ACTS
#include "ALICE3GlobalReconstruction/TrackerACTS.h"
#endif

#include <TFile.h>
#include <TTree.h>

namespace o2
{
using namespace framework;
namespace trk
{
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

namespace
{
class ALICE3TrackingBackendLoader : public o2::utils::DLLoaderBase<ALICE3TrackingBackendLoader>
{
  O2DLLoaderDef(ALICE3TrackingBackendLoader)
};

O2DLLoaderImpl(ALICE3TrackingBackendLoader)

  constexpr const char* kGPUBackendFunction = "runALICE3GPUTracking";
} // namespace

TrackerDPL::TrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr,
                       bool isMC,
                       const std::string& hitRecoConfigFileName,
                       const std::string& clusterRecoConfigFileName,
                       o2::gpu::gpudatatypes::DeviceType dType,
                       int trackingThreads)
{
  if (!hitRecoConfigFileName.empty()) {
    std::ifstream configFile(hitRecoConfigFileName);
    mHitRecoConfig = nlohmann::json::parse(configFile);
  }
  if (!clusterRecoConfigFileName.empty()) {
    std::ifstream configFile(clusterRecoConfigFileName);
    mClusterRecoConfig = nlohmann::json::parse(configFile);
  }
  mIsMC = isMC;
  mDeviceType = dType;
  mTrackingThreads = std::max(1, trackingThreads);
}

void TrackerDPL::init(InitContext& ic)
{
#ifdef O2_WITH_ACTS
  mUseACTS = ic.options().get<bool>("useACTS");
#endif
}

void TrackerDPL::stop()
{
  LOGF(info, "CPU Reconstruction total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

std::vector<o2::its::TrackingParameters> TrackerDPL::createTrackingParamsFromConfig()
{
  std::vector<o2::its::TrackingParameters> trackingParams;
  auto loadTrackingParamsFromJson = [](std::vector<o2::its::TrackingParameters>& trackingParams, const nlohmann::json& paramConfigJson) {
    for (const auto& paramConfig : paramConfigJson) {
      o2::its::TrackingParameters params;
      auto applyPassFlag = [&](const char* name, o2::its::IterationStep step) {
        if (paramConfig.contains(name)) {
          if (paramConfig[name].get<bool>()) {
            params.PassFlags.set(step);
          } else {
            params.PassFlags.reset(step);
          }
        }
      };

      if (paramConfig.contains("NLayers")) {
        params.NLayers = paramConfig["NLayers"].get<int>();
      }
      if (paramConfig.contains("ZBins")) {
        params.ZBins = paramConfig["ZBins"].get<int>();
      }
      if (paramConfig.contains("PhiBins")) {
        params.PhiBins = paramConfig["PhiBins"].get<int>();
      }
      if (paramConfig.contains("SharedMaxClusters")) {
        params.SharedMaxClusters = paramConfig["SharedMaxClusters"].get<int>();
      }
      if (paramConfig.contains("MinTrackLength")) {
        params.MinTrackLength = paramConfig["MinTrackLength"].get<int>();
      }
      if (paramConfig.contains("ReseedIfShorter")) {
        params.ReseedIfShorter = paramConfig["ReseedIfShorter"].get<int>();
      }
      if (paramConfig.contains("StartLayerMask")) {
        params.StartLayerMask = paramConfig["StartLayerMask"].get<uint16_t>();
      }

      if (paramConfig.contains("NSigmaCut")) {
        params.NSigmaCut = paramConfig["NSigmaCut"].get<float>();
      }
      if (paramConfig.contains("PVres")) {
        params.PVres = paramConfig["PVres"].get<float>();
      }
      if (paramConfig.contains("TrackletMinPt")) {
        params.TrackletMinPt = paramConfig["TrackletMinPt"].get<float>();
      }
      if (paramConfig.contains("CellDeltaTanLambdaSigma")) {
        params.CellDeltaTanLambdaSigma = paramConfig["CellDeltaTanLambdaSigma"].get<float>();
      }
      if (paramConfig.contains("MaxChi2ClusterAttachment")) {
        params.MaxChi2ClusterAttachment = paramConfig["MaxChi2ClusterAttachment"].get<float>();
      }
      if (paramConfig.contains("MaxChi2NDF")) {
        params.MaxChi2NDF = paramConfig["MaxChi2NDF"].get<float>();
      }

      if (paramConfig.contains("UseDiamond")) {
        params.UseDiamond = paramConfig["UseDiamond"].get<bool>();
      }
      if (paramConfig.contains("AllowSharingFirstCluster")) {
        params.AllowSharingFirstCluster = paramConfig["AllowSharingFirstCluster"].get<bool>();
      }
      if (paramConfig.contains("RepeatRefitOut")) {
        params.RepeatRefitOut = paramConfig["RepeatRefitOut"].get<bool>();
      }
      if (paramConfig.contains("ShiftRefToCluster")) {
        params.ShiftRefToCluster = paramConfig["ShiftRefToCluster"].get<bool>();
      }
      if (paramConfig.contains("PerPrimaryVertexProcessing")) {
        params.PerPrimaryVertexProcessing = paramConfig["PerPrimaryVertexProcessing"].get<bool>();
      }
      if (paramConfig.contains("SaveTimeBenchmarks")) {
        params.SaveTimeBenchmarks = paramConfig["SaveTimeBenchmarks"].get<bool>();
      }
      if (paramConfig.contains("DoUPCIteration")) {
        params.DoUPCIteration = paramConfig["DoUPCIteration"].get<bool>();
      }
      if (paramConfig.contains("FataliseUponFailure")) {
        params.FataliseUponFailure = paramConfig["FataliseUponFailure"].get<bool>();
      }
      if (paramConfig.contains("CreateArtefactLabels")) {
        params.CreateArtefactLabels = paramConfig["CreateArtefactLabels"].get<bool>();
      }
      if (paramConfig.contains("TrackFollower")) {
        const auto mode = paramConfig["TrackFollower"].get<std::string>();
        if (mode == "top" || mode == "outward") {
          params.PassFlags.set(o2::its::IterationStep::TrackFollowerTop);
        } else if (mode == "bot" || mode == "bottom" || mode == "inward") {
          params.PassFlags.set(o2::its::IterationStep::TrackFollowerBot);
        } else if (mode == "mix" || mode == "both") {
          params.PassFlags.set(o2::its::IterationStep::TrackFollowerTop);
          params.PassFlags.set(o2::its::IterationStep::TrackFollowerBot);
        } else if (mode != "off") {
          LOGP(fatal, "Invalid ALICE3 TRK tracking parameter TrackFollower: {}", mode);
        }
      }
      if (paramConfig.contains("TrackFollowerNSigmaCutZ")) {
        params.TrackFollowerNSigmaCutZ = paramConfig["TrackFollowerNSigmaCutZ"].get<float>();
      }
      if (paramConfig.contains("TrackFollowerNSigmaCutPhi")) {
        params.TrackFollowerNSigmaCutPhi = paramConfig["TrackFollowerNSigmaCutPhi"].get<float>();
      }
      if (paramConfig.contains("TrackFollowerMaxHypotheses")) {
        params.TrackFollowerMaxHypotheses = std::max(1, paramConfig["TrackFollowerMaxHypotheses"].get<int>());
      }
      applyPassFlag("FirstPass", o2::its::IterationStep::FirstPass);
      applyPassFlag("RebuildClusterLUT", o2::its::IterationStep::RebuildClusterLUT);
      applyPassFlag("UseUPCMask", o2::its::IterationStep::UseUPCMask);
      applyPassFlag("SelectUPCVertices", o2::its::IterationStep::SelectUPCVertices);
      applyPassFlag("ResetVertices", o2::its::IterationStep::ResetVertices);
      applyPassFlag("SkipROFsAboveThreshold", o2::its::IterationStep::SkipROFsAboveThreshold);
      applyPassFlag("MarkVerticesAsUPC", o2::its::IterationStep::MarkVerticesAsUPC);
      applyPassFlag("TrackFollowerTop", o2::its::IterationStep::TrackFollowerTop);
      applyPassFlag("TrackFollowerBot", o2::its::IterationStep::TrackFollowerBot);
      if (paramConfig.contains("PrintMemory")) {
        params.PrintMemory = paramConfig["PrintMemory"].get<bool>();
      }
      if (paramConfig.contains("DropTFUponFailure")) {
        params.DropTFUponFailure = paramConfig["DropTFUponFailure"].get<bool>();
      }

      if (paramConfig.contains("LayerZ")) {
        params.LayerZ = paramConfig["LayerZ"].get<std::vector<float>>();
      }
      if (paramConfig.contains("LayerRadii")) {
        params.LayerRadii = paramConfig["LayerRadii"].get<std::vector<float>>();
      }
      if (paramConfig.contains("LayerxX0")) {
        params.LayerxX0 = paramConfig["LayerxX0"].get<std::vector<float>>();
      }
      if (paramConfig.contains("LayerResolution")) {
        params.LayerResolution = paramConfig["LayerResolution"].get<std::vector<float>>();
      }
      if (paramConfig.contains("SystErrorY2")) {
        params.SystErrorY2 = paramConfig["SystErrorY2"].get<std::vector<float>>();
      }
      if (paramConfig.contains("SystErrorZ2")) {
        params.SystErrorZ2 = paramConfig["SystErrorZ2"].get<std::vector<float>>();
      }
      if (paramConfig.contains("MinPt")) {
        params.MinPt = paramConfig["MinPt"].get<std::vector<float>>();
      }
      if (paramConfig.contains("AddTimeError")) {
        params.AddTimeError = paramConfig["AddTimeError"].get<std::vector<UInt_t>>();
      }

      if (paramConfig.contains("Diamond") && paramConfig["Diamond"].is_array() && paramConfig["Diamond"].size() == 3) {
        params.Diamond[0] = paramConfig["Diamond"][0].get<float>();
        params.Diamond[1] = paramConfig["Diamond"][1].get<float>();
        params.Diamond[2] = paramConfig["Diamond"][2].get<float>();
      }

      if (paramConfig.contains("MaxMemory")) {
        params.MaxMemory = paramConfig["MaxMemory"].get<size_t>();
      }

      if (paramConfig.contains("CorrType")) {
        int corrTypeInt = paramConfig["CorrType"].get<int>();
        params.CorrType = static_cast<o2::base::PropagatorImpl<float>::MatCorrType>(corrTypeInt);
      }

      const auto nLayers = static_cast<size_t>(params.NLayers);
      LOG_IF(fatal, params.LayerZ.size() != nLayers) << "Invalid ALICE3 TRK tracking parameter LayerZ: expected " << nLayers << " entries, got " << params.LayerZ.size();
      LOG_IF(fatal, params.LayerRadii.size() != nLayers) << "Invalid ALICE3 TRK tracking parameter LayerRadii: expected " << nLayers << " entries, got " << params.LayerRadii.size();
      LOG_IF(fatal, params.LayerxX0.size() != nLayers) << "Invalid ALICE3 TRK tracking parameter LayerxX0: expected " << nLayers << " entries, got " << params.LayerxX0.size();
      LOG_IF(fatal, params.LayerResolution.size() != nLayers) << "Invalid ALICE3 TRK tracking parameter LayerResolution: expected " << nLayers << " entries, got " << params.LayerResolution.size();
      LOG_IF(fatal, params.SystErrorY2.size() != nLayers) << "Invalid ALICE3 TRK tracking parameter SystErrorY2: expected " << nLayers << " entries, got " << params.SystErrorY2.size();
      LOG_IF(fatal, params.SystErrorZ2.size() != nLayers) << "Invalid ALICE3 TRK tracking parameter SystErrorZ2: expected " << nLayers << " entries, got " << params.SystErrorZ2.size();
      LOG_IF(fatal, params.AddTimeError.size() != nLayers) << "Invalid ALICE3 TRK tracking parameter AddTimeError: expected " << nLayers << " entries, got " << params.AddTimeError.size();

      LOG_IF(fatal, params.MinTrackLength > params.NLayers) << "Invalid ALICE3 TRK tracking parameter MinTrackLength: expected <= NLayers (" << params.NLayers << "), got " << params.MinTrackLength;
      const auto minPtSize = static_cast<size_t>(params.NLayers - params.MinTrackLength + 1);
      LOG_IF(fatal, params.MinPt.size() != minPtSize) << "Invalid ALICE3 TRK tracking parameter MinPt: expected " << minPtSize << " entries, got " << params.MinPt.size();

      trackingParams.push_back(params);
    }
  };

  if (mHitRecoConfig.contains("trackingparams") && mHitRecoConfig["trackingparams"].is_array()) {
    loadTrackingParamsFromJson(trackingParams, mHitRecoConfig["trackingparams"]);
  } else if (mClusterRecoConfig.contains("trackingparams") && mClusterRecoConfig["trackingparams"].is_array()) {
    loadTrackingParamsFromJson(trackingParams, mClusterRecoConfig["trackingparams"]);
  } else {
    LOGP(fatal, "No trackingparams field found in configuration or it is not an array. Returning empty vector.");
    return trackingParams;
  }

  LOGP(info, "Loaded {} tracking parameter sets from configuration", trackingParams.size());
  return trackingParams;
}

void TrackerDPL::run(ProcessingContext& pc)
{
  if (mMemoryPool.get() == nullptr) {
    mMemoryPool = std::make_shared<its::BoundedMemoryResource>();
  }
  if (mTaskArena.get() == nullptr) {
    mTaskArena = std::make_shared<tbb::task_arena>(mTrackingThreads);
  }

  mTrackingParams = createTrackingParamsFromConfig();

  auto cput = mTimer.CpuTime();
  auto realt = mTimer.RealTime();
  mTimer.Start(false);

  const bool useGPU = mDeviceType != o2::gpu::gpudatatypes::DeviceType::CPU;

  if (useGPU) {
    runGPUTracking(pc);
  } else {
    o2::trk::TimeFrame<11> timeFrame;
    o2::its::TrackerTraits<11> itsTrackerTraits;
    runTracking(pc, timeFrame, itsTrackerTraits);
  }

  pc.services().get<o2::framework::ControlService>().endOfStream();
  pc.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);

  mTimer.Stop();
  LOGP(info, "CPU Reconstruction time for this TF {} s (cpu), {} s (wall)", mTimer.CpuTime() - cput, mTimer.RealTime() - realt);
}

void TrackerDPL::runGPUTracking(ProcessingContext& pc)
{
  auto& loader = ALICE3TrackingBackendLoader::Instance();
  switch (mDeviceType) {
    case o2::gpu::gpudatatypes::DeviceType::CUDA:
#ifdef TRK_HAS_CUDA_TRACKING
      loader.executeFunctionAlias<int, TrackerDPL*, ProcessingContext*>("O2ALICE3GlobalReconstructionWorkflowCUDA", kGPUBackendFunction, this, &pc);
      return;
#else
      LOGP(fatal, "CUDA TRK GPU tracking was requested but this build has no CUDA TRK GPU tracking backend");
#endif
    case o2::gpu::gpudatatypes::DeviceType::HIP:
#ifdef TRK_HAS_HIP_TRACKING
      loader.executeFunctionAlias<int, TrackerDPL*, ProcessingContext*>("O2ALICE3GlobalReconstructionWorkflowHIP", kGPUBackendFunction, this, &pc);
      return;
#else
      LOGP(fatal, "HIP TRK GPU tracking was requested but this build has no HIP TRK GPU tracking backend");
#endif
    default:
      LOGP(fatal, "Unsupported TRK GPU device type {}", static_cast<int>(mDeviceType));
  }
}

void TrackerDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TRK CA-Tracker total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTrackerSpec(bool useMC, const std::string& hitRecoConfig, const std::string& clusterRecoConfig, o2::gpu::gpudatatypes::DeviceType dType, int trackingThreads)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TRK", "TRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRK", "TRACKSROF", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRK", "IRFRAMES", 0, Lifetime::Timeframe);
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                              false,                          // GRPECS=true
                                                              false,                          // GRPLHCIF
                                                              false,                          // GRPMagField
                                                              false,                          // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry, but ignored until it will be put in the CCDB
                                                              inputs,
                                                              true);

  if (!hitRecoConfig.empty()) {
    if (useMC) {
      outputs.emplace_back("TRK", "TRACKSMCTR", 0, Lifetime::Timeframe);
    }
    return DataProcessorSpec{
      "trk-hits-tracker",
      {},
      outputs,
      AlgorithmSpec{adaptFromTask<TrackerDPL>(ggRequest,
                                              useMC,
                                              hitRecoConfig,
                                              clusterRecoConfig,
                                              dType,
                                              trackingThreads)},
      Options{ConfigParamSpec{"max-loops", VariantType::Int, 1, {"max number of loops"}}
#ifdef O2_WITH_ACTS
              ,
              {"useACTS", o2::framework::VariantType::Bool, false, {"Use ACTS for tracking"}}
#endif
      }};
  }

  inputs.emplace_back("dummy", "TRK", "DUMMY", 0, Lifetime::Timeframe);

  if (!clusterRecoConfig.empty()) {
    inputs.pop_back();
    constexpr int nLayers{11};
    for (int iLayer = 0; iLayer < nLayers; ++iLayer) {
      inputs.emplace_back(std::format("compClusters_{}", iLayer), "TRK", "COMPCLUSTERS", iLayer, Lifetime::Timeframe);
      inputs.emplace_back(std::format("patterns_{}", iLayer), "TRK", "PATTERNS", iLayer, Lifetime::Timeframe);
      inputs.emplace_back(std::format("ROframes_{}", iLayer), "TRK", "CLUSTERSROF", iLayer, Lifetime::Timeframe);
      if (useMC) {
        inputs.emplace_back(std::format("trkmclabels_{}", iLayer), "TRK", "CLUSTERSMCTR", iLayer, Lifetime::Timeframe);
      }
    }
  }

  if (useMC) {
    outputs.emplace_back("TRK", "TRACKSMCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "trk-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackerDPL>(ggRequest,
                                            useMC,
                                            hitRecoConfig,
                                            clusterRecoConfig,
                                            dType,
                                            trackingThreads)},
    Options{}};
}

} // namespace trk
} // namespace o2
