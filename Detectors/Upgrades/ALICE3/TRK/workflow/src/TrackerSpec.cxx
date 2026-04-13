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
#include <chrono>

#include "DetectorsBase/GeometryManager.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/Configuration.h"
#include "Field/MagneticField.h"
#include "Field/MagFieldParam.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "TRKBase/GeometryTGeo.h"
#include "TRKBase/SegmentationChip.h"
#include "TRKSimulation/Hit.h"
#include "TRKReconstruction/TimeFrame.h"
#include "TRKWorkflow/TrackerSpec.h"
#include <TGeoGlobalMagField.h>

#ifdef O2_WITH_ACTS
#include "TRKReconstruction/TrackerACTS.h"
#endif

#include <TFile.h>
#include <TTree.h>

namespace o2
{
using namespace framework;
namespace trk
{

TrackerDPL::TrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr,
                       bool isMC,
                       const std::string& hitRecoConfigFileName,
                       o2::gpu::gpudatatypes::DeviceType dType)
{
  if (!hitRecoConfigFileName.empty()) {
    std::ifstream configFile(hitRecoConfigFileName);
    mHitRecoConfig = nlohmann::json::parse(configFile);
  }

  // mITSTrackingInterface.setTrackingMode(trMode);
}

void TrackerDPL::init(InitContext& ic)
{
  // mTimer.Stop();
  // mTimer.Reset();
  // o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  // mChainITS.reset(mRecChain->AddChain<o2::gpu::GPUChainITS>());
  // mITSTrackingInterface.setTraitsFromProvider(mChainITS->GetITSVertexerTraits(),
  //                                             mChainITS->GetITSTrackerTraits(),
  //                                             mChainITS->GetITSTimeframe());

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

  if (!mHitRecoConfig.contains("trackingparams") || !mHitRecoConfig["trackingparams"].is_array()) {
    LOGP(fatal, "No trackingparams field found in configuration or it is not an array. Returning empty vector.");
    return trackingParams;
  }

  for (const auto& paramConfig : mHitRecoConfig["trackingparams"]) {
    o2::its::TrackingParameters params;

    // Parse integer parameters
    if (paramConfig.contains("NLayers")) {
      params.NLayers = paramConfig["NLayers"].get<int>();
    }
    if (paramConfig.contains("ZBins")) {
      params.ZBins = paramConfig["ZBins"].get<int>();
    }
    if (paramConfig.contains("PhiBins")) {
      params.PhiBins = paramConfig["PhiBins"].get<int>();
    }
    if (paramConfig.contains("ClusterSharing")) {
      params.ClusterSharing = paramConfig["ClusterSharing"].get<int>();
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

    // Parse float parameters
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
    // if (paramConfig.contains("TrackFollowerNSigmaCutZ")) {
    //   params.TrackFollowerNSigmaCutZ = paramConfig["TrackFollowerNSigmaCutZ"].get<float>();
    // }
    // if (paramConfig.contains("TrackFollowerNSigmaCutPhi")) {
    //   params.TrackFollowerNSigmaCutPhi = paramConfig["TrackFollowerNSigmaCutPhi"].get<float>();
    // }

    // Parse boolean parameters
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
    // if (paramConfig.contains("FindShortTracks")) {
    //   params.FindShortTracks = paramConfig["FindShortTracks"].get<bool>();
    // }
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
    // if (paramConfig.contains("UseTrackFollower")) {
    //   params.UseTrackFollower = paramConfig["UseTrackFollower"].get<bool>();
    // }
    // if (paramConfig.contains("UseTrackFollowerTop")) {
    //   params.UseTrackFollowerTop = paramConfig["UseTrackFollowerTop"].get<bool>();
    // }
    // if (paramConfig.contains("UseTrackFollowerBot")) {
    //   params.UseTrackFollowerBot = paramConfig["UseTrackFollowerBot"].get<bool>();
    // }
    // if (paramConfig.contains("UseTrackFollowerMix")) {
    //   params.UseTrackFollowerMix = paramConfig["UseTrackFollowerMix"].get<bool>();
    // }
    if (paramConfig.contains("createArtefactLabels")) {
      params.createArtefactLabels = paramConfig["createArtefactLabels"].get<bool>();
    }
    if (paramConfig.contains("PrintMemory")) {
      params.PrintMemory = paramConfig["PrintMemory"].get<bool>();
    }
    if (paramConfig.contains("DropTFUponFailure")) {
      params.DropTFUponFailure = paramConfig["DropTFUponFailure"].get<bool>();
    }

    // Parse vector parameters
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

    // Parse Diamond array
    if (paramConfig.contains("Diamond") && paramConfig["Diamond"].is_array() && paramConfig["Diamond"].size() == 3) {
      params.Diamond[0] = paramConfig["Diamond"][0].get<float>();
      params.Diamond[1] = paramConfig["Diamond"][1].get<float>();
      params.Diamond[2] = paramConfig["Diamond"][2].get<float>();
    }

    // Parse size_t parameter
    if (paramConfig.contains("MaxMemory")) {
      params.MaxMemory = paramConfig["MaxMemory"].get<size_t>();
    }

    // Parse CorrType enum
    if (paramConfig.contains("CorrType")) {
      int corrTypeInt = paramConfig["CorrType"].get<int>();
      params.CorrType = static_cast<o2::base::PropagatorImpl<float>::MatCorrType>(corrTypeInt);
    }

    trackingParams.push_back(params);
  }

  LOGP(info, "Loaded {} tracking parameter sets from configuration", trackingParams.size());
  return trackingParams;
}

void TrackerDPL::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  auto realt = mTimer.RealTime();
  mTimer.Start(false);

  if (!mHitRecoConfig.empty()) {
    TFile hitsFile(mHitRecoConfig["inputfiles"]["hits"].get<std::string>().c_str(), "READ");
    TFile mcHeaderFile(mHitRecoConfig["inputfiles"]["mcHeader"].get<std::string>().c_str(), "READ");
    TTree* hitsTree = hitsFile.Get<TTree>("o2sim");
    std::vector<o2::trk::Hit>* trkHit = nullptr;
    hitsTree->SetBranchAddress("TRKHit", &trkHit);

    TTree* mcHeaderTree = mcHeaderFile.Get<TTree>("o2sim");
    auto mcheader = new o2::dataformats::MCEventHeader;
    mcHeaderTree->SetBranchAddress("MCEventHeader.", &mcheader);

    o2::base::GeometryManager::loadGeometry(mHitRecoConfig["inputfiles"]["geometry"].get<std::string>().c_str(), false, true);
    auto* gman = o2::trk::GeometryTGeo::Instance();

    const Long64_t nEvents{hitsTree->GetEntries()};
    LOGP(info, "Starting reconstruction from hits for {} events", nEvents);

    if (mMemoryPool.get() == nullptr) {
      mMemoryPool = std::make_shared<its::BoundedMemoryResource>();
    }
    if (mTaskArena.get() == nullptr) {
      mTaskArena = std::make_shared<tbb::task_arena>(1); /// TODO: make it configurable
    }

    o2::trk::TimeFrame<11> timeFrame;
    o2::its::TrackerTraits<11> itsTrackerTraits;
    o2::its::Tracker<11> itsTracker(&itsTrackerTraits);
    timeFrame.setMemoryPool(mMemoryPool);
    itsTrackerTraits.setMemoryPool(mMemoryPool);
    itsTrackerTraits.setNThreads(mTaskArena->max_concurrency(), mTaskArena);
    itsTrackerTraits.adoptTimeFrame(static_cast<o2::its::TimeFrame<11>*>(&timeFrame));
    itsTrackerTraits.setBz(mHitRecoConfig["geometry"]["bz"].get<float>());
    auto field = new field::MagneticField("ALICE3Mag", "ALICE 3 Magnetic Field", mHitRecoConfig["geometry"]["bz"].get<float>() / 5.f, 0.0, o2::field::MagFieldParam::k5kGUniform);
    TGeoGlobalMagField::Instance()->SetField(field);
    TGeoGlobalMagField::Instance()->Lock();
    itsTracker.adoptTimeFrame(timeFrame);

    const int nRofs = timeFrame.loadROFsFromHitTree(hitsTree, gman, mHitRecoConfig);
    const int inROFpileup{mHitRecoConfig.contains("inROFpileup") ? mHitRecoConfig["inROFpileup"].get<int>() : 1};

    // Add primary vertices from MC headers for each ROF
    timeFrame.getPrimaryVerticesFromMC(mcHeaderTree, nRofs, nEvents, inROFpileup);
    // Create tracking parameters from config and set them in the time frame
    auto trackingParams = createTrackingParamsFromConfig();

    itsTrackerTraits.updateTrackingParameters(trackingParams);

#ifdef O2_WITH_ACTS
    if (mUseACTS) {
      LOG(info) << "Running the tracking with ACTS";
      o2::trk::TrackerACTS<11> actsTracker;
      actsTracker.setBz(mHitRecoConfig["geometry"]["bz"].get<float>());
      actsTracker.adoptTimeFrame(timeFrame);
      actsTracker.clustersToTracks();
    }
#endif

    const auto trackingLoopStart = std::chrono::steady_clock::now();
    for (size_t iter{0}; iter < trackingParams.size(); ++iter) {
      LOGP(info, "{}", trackingParams[iter].asString());
      timeFrame.initialise(iter, trackingParams[iter], 11, false);
      itsTrackerTraits.computeLayerTracklets(iter, -1);
      LOGP(info, "Number of tracklets in iteration {}: {}", iter, timeFrame.getNumberOfTracklets());
      itsTrackerTraits.computeLayerCells(iter);
      LOGP(info, "Number of cells in iteration {}: {}", iter, timeFrame.getNumberOfCells());
      itsTrackerTraits.findCellsNeighbours(iter);
      LOGP(info, "Number of cell neighbours in iteration {}: {}", iter, timeFrame.getNumberOfNeighbours());
      itsTrackerTraits.findRoads(iter);
      LOGP(info, "Number of tracks in iteration {}: {}", iter, timeFrame.getNumberOfTracks());
    }
    const auto trackingLoopElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - trackingLoopStart).count();
    LOGP(info, "Tracking iterations block took {} ms", trackingLoopElapsedMs);

    itsTracker.computeTracksMClabels();

    // Collect tracks and labels (flat vectors in the new interface)
    const auto& tracks = timeFrame.getTracks();
    const auto& labels = timeFrame.getTracksLabel();

    // Copy to output vectors (TrackITSExt -> TrackITS slicing for output compatibility)
    std::vector<o2::its::TrackITS> allTracks(tracks.begin(), tracks.end());
    std::vector<o2::MCCompLabel> allLabels(labels.begin(), labels.end());

    int totalTracks = allTracks.size();
    int goodTracks = 0;
    int fakeTracks = 0;

    for (const auto& label : allLabels) {
      if (label.isFake()) {
        fakeTracks++;
      } else {
        goodTracks++;
      }
    }

    LOGP(info, "=== Tracking Summary ===");
    LOGP(info, "Total tracks reconstructed: {}", totalTracks);
    LOGP(info, "Good tracks: {} ({:.1f}%)", goodTracks, totalTracks > 0 ? 100.0 * goodTracks / totalTracks : 0);
    LOGP(info, "Fake tracks: {} ({:.1f}%)", fakeTracks, totalTracks > 0 ? 100.0 * fakeTracks / totalTracks : 0);

    // Stream tracks and labels to DPL output
    pc.outputs().snapshot(o2::framework::Output{"TRK", "TRACKS", 0}, allTracks);
    pc.outputs().snapshot(o2::framework::Output{"TRK", "TRACKSMCTR", 0}, allLabels);

    LOGP(info, "Tracks and MC labels streamed to output");

    pc.services().get<o2::framework::ControlService>().endOfStream();
    pc.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
  }

  mTimer.Stop();
  LOGP(info, "CPU Reconstruction time for this TF {} s (cpu), {} s (wall)", mTimer.CpuTime() - cput, mTimer.RealTime() - realt);
}

// void TrackerDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
// {
//   // mITSTrackingInterface.finaliseCCDB(matcher, obj);
// }

void TrackerDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TRK CA-Tracker total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTrackerSpec(bool useMC, const std::string& hitRecoConfig, o2::gpu::gpudatatypes::DeviceType dType)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TRK", "TRACKS", 0, Lifetime::Timeframe);
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                              false,                          // GRPECS=true
                                                              false,                          // GRPLHCIF
                                                              false,                          // GRPMagField
                                                              false,                          // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry, but ignored until it will be put in the CCDB
                                                              inputs,
                                                              true);

  if (!hitRecoConfig.empty()) {
    outputs.emplace_back("TRK", "TRACKSMCTR", 0, Lifetime::Timeframe);
    return DataProcessorSpec{
      "trk-hits-tracker",
      {},
      outputs,
      AlgorithmSpec{adaptFromTask<TrackerDPL>(ggRequest,
                                              useMC,
                                              hitRecoConfig,
                                              dType)},
      Options{ConfigParamSpec{"max-loops", VariantType::Int, 1, {"max number of loops"}}
#ifdef O2_WITH_ACTS
              ,
              {"useACTS", o2::framework::VariantType::Bool, false, {"Use ACTS for tracking"}}
#endif
      }};
  }

  inputs.emplace_back("dummy", "TRK", "DUMMY", 0, Lifetime::Timeframe);

  constexpr bool expectClusterInputs = false;
  if (expectClusterInputs) {
    inputs.pop_back();
    inputs.emplace_back("compClusters", "TRK", "COMPCLUSTERS", 0, Lifetime::Timeframe);
    inputs.emplace_back("patterns", "TRK", "PATTERNS", 0, Lifetime::Timeframe);
    inputs.emplace_back("ROframes", "TRK", "CLUSTERSROF", 0, Lifetime::Timeframe);
  }

  // inputs.emplace_back("itscldict", "TRK", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/ClusterDictionary"));
  // inputs.emplace_back("TRK_almiraparam", "TRK", "ALMIRAPARAM", 0, Lifetime::Condition, ccdbParamSpec("TRK/Config/AlmiraParam"));

  // outputs.emplace_back("TRK", "TRACKCLSID", 0, Lifetime::Timeframe);
  // outputs.emplace_back("TRK", "TRKTrackROF", 0, Lifetime::Timeframe);
  // outputs.emplace_back("TRK", "VERTICES", 0, Lifetime::Timeframe);
  // outputs.emplace_back("TRK", "VERTICESROF", 0, Lifetime::Timeframe);
  // outputs.emplace_back("TRK", "IRFRAMES", 0, Lifetime::Timeframe);

  if (useMC) {
    // inputs.emplace_back("trkmclabels", "TRK", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    // inputs.emplace_back("TRKMC2ROframes", "TRK", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
    // outputs.emplace_back("TRK", "VERTICESMCTR", 0, Lifetime::Timeframe);
    // outputs.emplace_back("TRK", "VERTICESMCPUR", 0, Lifetime::Timeframe);
    // outputs.emplace_back("TRK", "TRACKSMCTR", 0, Lifetime::Timeframe);
    // outputs.emplace_back("TRK", "TRKTrackMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "trk-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackerDPL>(ggRequest,
                                            useMC,
                                            hitRecoConfig,
                                            dType)},
    Options{}};
}

} // namespace trk
} // namespace o2
