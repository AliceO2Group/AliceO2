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
#ifdef TRK_HAS_GPU_TRACKING
#include "ALICE3GlobalReconstruction/TimeFrameGPU.h"
#include "ALICE3GlobalReconstruction/GPUExternalAllocator.h"
#include "ITStrackingGPU/TrackerTraitsGPU.h"
#endif
#include "ALICE3GlobalReconstructionWorkflow/TrackerSpec.h"
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

TrackerDPL::TrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr,
                       bool isMC,
                       const std::string& hitRecoConfigFileName,
                       const std::string& clusterRecoConfigFileName,
                       o2::gpu::gpudatatypes::DeviceType dType)
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
  auto loadTrackingParamsFromJson = [](std::vector<o2::its::TrackingParameters>& trackingParams, const nlohmann::json& paramConfigJson) {
    for (const auto& paramConfig : paramConfigJson) {
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
    mTaskArena = std::make_shared<tbb::task_arena>(1); /// TODO: make it configurable
  }

  // Create tracking parameters from config and set them in the time frame
  auto trackingParams = createTrackingParamsFromConfig();

  auto cput = mTimer.CpuTime();
  auto realt = mTimer.RealTime();
  mTimer.Start(false);

  const bool useGPU = mDeviceType != o2::gpu::gpudatatypes::DeviceType::CPU;
#ifndef TRK_HAS_GPU_TRACKING
  if (useGPU) {
    LOGP(fatal, "TRK GPU tracking was requested but this build has no TRK GPU tracking backend");
  }
#else
#ifdef TRK_HAS_CUDA_TRACKING
  if (useGPU && mDeviceType != o2::gpu::gpudatatypes::DeviceType::CUDA) {
    LOGP(fatal, "This build provides the CUDA TRK tracking backend only, but device type {} was requested", static_cast<int>(mDeviceType));
  }
#elif defined(TRK_HAS_HIP_TRACKING)
  if (useGPU && mDeviceType != o2::gpu::gpudatatypes::DeviceType::HIP) {
    LOGP(fatal, "This build provides the HIP TRK tracking backend only, but device type {} was requested", static_cast<int>(mDeviceType));
  }
#endif
#endif

  auto runTracking = [&](auto& timeFrame, auto& trackerTraits) {
    o2::its::Tracker<11> itsTracker(&trackerTraits);
    timeFrame.setMemoryPool(mMemoryPool);
    trackerTraits.setMemoryPool(mMemoryPool);
    trackerTraits.setNThreads(mTaskArena->max_concurrency(), mTaskArena);
    trackerTraits.adoptTimeFrame(static_cast<o2::its::TimeFrame<11>*>(&timeFrame));
    itsTracker.adoptTimeFrame(timeFrame);
    trackerTraits.updateTrackingParameters(trackingParams);

    int nRofs{0};
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
      LOGP(info, "Starting {} reconstruction from hits for {} events", trackerTraits.getName(), nEvents);

      trackerTraits.setBz(mHitRecoConfig["geometry"]["bz"].get<float>());
      auto field = new field::MagneticField("ALICE3Mag", "ALICE 3 Magnetic Field", mHitRecoConfig["geometry"]["bz"].get<float>() / 5.f, 0.0, o2::field::MagFieldParam::k5kGUniform);
      TGeoGlobalMagField::Instance()->SetField(field);
      TGeoGlobalMagField::Instance()->Lock();

      nRofs = timeFrame.loadROFsFromHitTree(hitsTree, gman, mHitRecoConfig);
      const int inROFpileup{mHitRecoConfig.contains("inROFpileup") ? mHitRecoConfig["inROFpileup"].get<int>() : 1};
      timeFrame.getPrimaryVerticesFromMC(mcHeaderTree, nRofs, nEvents, inROFpileup);
    } else if (!mClusterRecoConfig.empty()) {
      LOGP(info, "Starting {} reconstruction from clusters", trackerTraits.getName());

      o2::base::GeometryManager::loadGeometry(mClusterRecoConfig["inputfiles"]["geometry"].get<std::string>().c_str(), false, true);
      o2::trk::GeometryTGeo::Instance();

      trackerTraits.setBz(mClusterRecoConfig["geometry"]["bz"].get<float>());
      auto field = new field::MagneticField("ALICE3Mag", "ALICE 3 Magnetic Field", mClusterRecoConfig["geometry"]["bz"].get<float>() / 5.f, 0.0, o2::field::MagFieldParam::k5kGUniform);
      TGeoGlobalMagField::Instance()->SetField(field);
      TGeoGlobalMagField::Instance()->Lock();

      constexpr int nLayers{11};
      std::array<gsl::span<const o2::trk::Cluster>, nLayers> layerClusters;
      std::array<gsl::span<const unsigned char>, nLayers> layerPatterns;
      std::array<gsl::span<const o2::trk::ROFRecord>, nLayers> layerROFs;
      std::array<const dataformats::MCTruthContainer<MCCompLabel>*, nLayers> layerLabels{};

      size_t nInputRofs{0};
      for (int iLayer = 0; iLayer < nLayers; ++iLayer) {
        layerClusters[iLayer] = pc.inputs().get<gsl::span<o2::trk::Cluster>>(std::format("compClusters_{}", iLayer));
        layerPatterns[iLayer] = pc.inputs().get<gsl::span<unsigned char>>(std::format("patterns_{}", iLayer));
        layerROFs[iLayer] = pc.inputs().get<gsl::span<o2::trk::ROFRecord>>(std::format("ROframes_{}", iLayer));
        nInputRofs = std::max(nInputRofs, layerROFs[iLayer].size());
        if (mIsMC) {
          layerLabels[iLayer] = pc.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>(std::format("trkmclabels_{}", iLayer)).release();
        }
      }

      // Build per-layer LayerTiming from each layer's own ROF BCData. This is
      // a precondition for loadROFrameData(): timing must be in place before
      // clusters are loaded. Idempotent across TFs.
      timeFrame.deriveAndInitTiming(layerROFs);

      std::vector<o2::trk::ROFRecord> rofRecords;
      std::vector<o2::trk::Cluster> compClusters;
      std::vector<unsigned char> patterns;
      dataformats::MCTruthContainer<MCCompLabel> labels;
      std::array<std::vector<size_t>, nLayers> patternOffsets;
      rofRecords.reserve(nInputRofs);
      for (int iLayer = 0; iLayer < nLayers; ++iLayer) {
        compClusters.reserve(compClusters.size() + layerClusters[iLayer].size());
        patterns.reserve(patterns.size() + layerPatterns[iLayer].size());

        auto& offsets = patternOffsets[iLayer];
        offsets.reserve(layerClusters[iLayer].size() + 1);
        offsets.push_back(0);
        size_t patternOffset{0};
        for (size_t iCluster = 0; iCluster < layerClusters[iLayer].size(); ++iCluster) {
          if (patternOffset + 2 > layerPatterns[iLayer].size()) {
            LOGP(error, "Pattern stream exhausted while indexing TRK layer {} cluster {}", iLayer, iCluster);
            offsets.push_back(layerPatterns[iLayer].size());
            continue;
          }
          const auto* pattForCluster = layerPatterns[iLayer].data() + patternOffset;
          const size_t nBytes = (pattForCluster[0] * pattForCluster[1] + 7) / 8;
          patternOffset = std::min(patternOffset + 2 + nBytes, layerPatterns[iLayer].size());
          offsets.push_back(patternOffset);
        }
      }

      for (size_t iRof = 0; iRof < nInputRofs; ++iRof) {
        o2::trk::ROFRecord mergedROF;
        mergedROF.setFirstEntry(compClusters.size());
        mergedROF.setROFrame(iRof);

        bool hasBCData{false};
        for (int iLayer = 0; iLayer < nLayers; ++iLayer) {
          if (iRof >= layerROFs[iLayer].size()) {
            continue;
          }
          const auto& layerROF = layerROFs[iLayer][iRof];
          if (!hasBCData) {
            mergedROF.setBCData(layerROF.getBCData());
            hasBCData = true;
          }

          const auto first = layerROF.getFirstEntry();
          const auto last = first + layerROF.getNEntries();
          if (first < 0 || last < first || last >= static_cast<int>(patternOffsets[iLayer].size())) {
            LOGP(warning, "Skipping TRK clusters for invalid range [{}, {}) on layer {}", first, last, iLayer);
            continue;
          }
          const auto patternFirst = patternOffsets[iLayer][first];
          const auto patternLast = patternOffsets[iLayer][last];
          for (int iCluster = first; iCluster < last; ++iCluster) {
            if (iCluster < 0 || iCluster >= static_cast<int>(layerClusters[iLayer].size())) {
              LOGP(warning, "Skipping out-of-range TRK cluster {} on layer {}", iCluster, iLayer);
              continue;
            }
            const auto mergedClusterIndex = compClusters.size();
            compClusters.push_back(layerClusters[iLayer][iCluster]);
            if (mIsMC) {
              if (layerLabels[iLayer] != nullptr) {
                labels.addElements(mergedClusterIndex, layerLabels[iLayer]->getLabels(iCluster));
              } else {
                labels.addNoLabelIndex(mergedClusterIndex);
              }
            }
          }
          patterns.insert(patterns.end(), layerPatterns[iLayer].begin() + patternFirst, layerPatterns[iLayer].begin() + patternLast);
        }

        mergedROF.setNEntries(compClusters.size() - mergedROF.getFirstEntry());
        rofRecords.push_back(mergedROF);
      }

      const float yPlaneMLOT = 0.0010f;
      nRofs = timeFrame.loadROFrameData(rofRecords, compClusters, patterns, mIsMC ? &labels : nullptr, yPlaneMLOT);
      timeFrame.addTruthSeedingVertices(rofRecords);
    }

    const auto trackingLoopStart = std::chrono::steady_clock::now();
    for (size_t iter{0}; iter < trackingParams.size(); ++iter) {
      LOGP(info, "{}", trackingParams[iter].asString());
      trackerTraits.initialiseTimeFrame(iter);
      trackerTraits.computeLayerTracklets(iter, -1);
      LOGP(info, "Number of tracklets in iteration {}: {}", iter, timeFrame.getNumberOfTracklets());
      trackerTraits.computeLayerCells(iter);
      LOGP(info, "Number of cells in iteration {}: {}", iter, timeFrame.getNumberOfCells());
      trackerTraits.findCellsNeighbours(iter);
      LOGP(info, "Number of cell neighbours in iteration {}: {}", iter, timeFrame.getNumberOfNeighbours());
      trackerTraits.findRoads(iter);
      LOGP(info, "Number of roads in iteration {}: {}", iter, timeFrame.getNumberOfTracks());
    }
    const auto trackingLoopElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - trackingLoopStart).count();
    LOGP(info, "Tracking iterations block took {} ms", trackingLoopElapsedMs);

    if (mIsMC) {
      itsTracker.computeTracksMClabels();
    }

    const auto& tracks = timeFrame.getTracks();
    const auto& labels = timeFrame.getTracksLabel();
    std::vector<o2::its::TrackITS> allTracks(tracks.begin(), tracks.end());
    std::vector<o2::MCCompLabel> allLabels;

    int totalTracks = allTracks.size();
    int goodTracks = 0;
    int fakeTracks = 0;

    if (mIsMC) {
      allLabels.assign(labels.begin(), labels.end());
      for (const auto& label : allLabels) {
        if (label.isFake()) {
          ++fakeTracks;
        } else {
          ++goodTracks;
        }
      }
    }

    LOGP(info, "=== Tracking Summary ===");
    LOGP(info, "Total tracks reconstructed: {}", totalTracks);
    LOGP(info, "Good tracks: {} ({:.1f}%)", goodTracks, totalTracks > 0 ? 100.0 * goodTracks / totalTracks : 0);
    LOGP(info, "Fake tracks: {} ({:.1f}%)", fakeTracks, totalTracks > 0 ? 100.0 * fakeTracks / totalTracks : 0);

    // Build per-ROF track records and IR frames in the clock-layer's frame.
    // The clock layer is the fastest TRK layer (max mNROFsTF) — see
    // ROFOverlapTableView::getClock(). Each track's TimeStamp deterministically
    // maps to a clock-layer ROF; track ordering from the tracker is by
    // increasing time, so the per-ROF firstEntry/nEntries pair indexes a
    // contiguous range of allTracks. Mirrors ITS TrackingInterface.cxx.
    const auto& rofView = timeFrame.getROFOverlapTableView();
    const auto& clockLayer = rofView.getClockLayer();
    const int clockLayerId = rofView.getClock();
    const int64_t anchorBC = timeFrame.getTFAnchorIR().toLong();

    int highestROF = static_cast<int>(clockLayer.mNROFsTF);
    for (const auto& trc : allTracks) {
      highestROF = std::max(highestROF, static_cast<int>(clockLayer.getROF(trc.getTimeStamp())));
    }
    for (const auto& vtx : timeFrame.getPrimaryVertices()) {
      highestROF = std::max(highestROF, static_cast<int>(clockLayer.getROF(vtx.getTimeStamp().lower())));
    }

    std::vector<o2::trk::ROFRecord> allTrackROFs(highestROF);
    for (size_t iROF = 0; iROF < allTrackROFs.size(); ++iROF) {
      auto& rof = allTrackROFs[iROF];
      o2::InteractionRecord ir;
      ir.setFromLong(anchorBC + static_cast<int64_t>(clockLayer.getROFStartInBC(iROF)));
      rof.setBCData(ir);
      rof.setROFrame(iROF);
      rof.setFirstEntry(0);
      rof.setNEntries(0);
    }

    std::vector<int> rofEntries(highestROF + 1, 0);
    for (const auto& trc : allTracks) {
      const int rof = static_cast<int>(clockLayer.getROF(trc.getTimeStamp()));
      if (rof >= 0 && rof < highestROF) {
        ++rofEntries[rof];
      }
    }
    std::exclusive_scan(rofEntries.begin(), rofEntries.end(), rofEntries.begin(), 0);

    std::vector<o2::dataformats::IRFrame> irFrames;
    irFrames.reserve(allTrackROFs.size());
    const auto& maskView = timeFrame.getROFMaskView();
    const auto rofLenMinus1 = clockLayer.mROFLength > 0 ? clockLayer.mROFLength - 1 : 0;
    for (size_t iROF = 0; iROF < allTrackROFs.size(); ++iROF) {
      allTrackROFs[iROF].setFirstEntry(rofEntries[iROF]);
      allTrackROFs[iROF].setNEntries(rofEntries[iROF + 1] - rofEntries[iROF]);
      if (maskView.isROFEnabled(clockLayerId, static_cast<int>(iROF))) {
        const auto& bcStart = allTrackROFs[iROF].getBCData();
        auto& irFrame = irFrames.emplace_back(bcStart, bcStart + rofLenMinus1);
        irFrame.info = allTrackROFs[iROF].getNEntries();
      }
    }

    pc.outputs().snapshot(o2::framework::Output{"TRK", "TRACKS", 0}, allTracks);
    pc.outputs().snapshot(o2::framework::Output{"TRK", "TRACKSROF", 0}, allTrackROFs);
    pc.outputs().snapshot(o2::framework::Output{"TRK", "IRFRAMES", 0}, irFrames);
    if (mIsMC) {
      pc.outputs().snapshot(o2::framework::Output{"TRK", "TRACKSMCTR", 0}, allLabels);
    }

    LOGP(info, "TRK pushed {} tracks in {} ROFs and {} IR frames{}",
         allTracks.size(), allTrackROFs.size(), irFrames.size(),
         mIsMC ? " (with MC labels)" : "");
  };

#ifdef TRK_HAS_GPU_TRACKING
  if (useGPU) {
    o2::trk::TimeFrameGPU<11> timeFrame;
    o2::its::TrackerTraitsGPU<11> itsTrackerTraits;
    if (!mGPUAllocator) {
      mGPUAllocator = std::make_shared<o2::trk::GPUExternalAllocator>();
    }
    timeFrame.setFrameworkAllocator(mGPUAllocator.get());
    runTracking(timeFrame, itsTrackerTraits);
  } else
#endif
  {
    o2::trk::TimeFrame<11> timeFrame;
    o2::its::TrackerTraits<11> itsTrackerTraits;
    runTracking(timeFrame, itsTrackerTraits);
  }

  pc.services().get<o2::framework::ControlService>().endOfStream();
  pc.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);

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

DataProcessorSpec getTrackerSpec(bool useMC, const std::string& hitRecoConfig, const std::string& clusterRecoConfig, o2::gpu::gpudatatypes::DeviceType dType)
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
                                              dType)},
      Options{ConfigParamSpec{"max-loops", VariantType::Int, 1, {"max number of loops"}}
#ifdef O2_WITH_ACTS
              ,
              {"useACTS", o2::framework::VariantType::Bool, false, {"Use ACTS for tracking"}}
#endif
      }};
  }

  inputs.emplace_back("dummy", "TRK", "DUMMY", 0, Lifetime::Timeframe);

  /// Keep momentarily both the hit-based and cluster-based reconstruction in the code, but if a cluster reco config is provided, we assume that the input will be clusters and not hits, and we set the inputs accordingly. This is to avoid having to change the workflow too much for the moment, but ideally we should drop the hit-based reconstruction asap.
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

  // inputs.emplace_back("itscldict", "TRK", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/ClusterDictionary"));
  // inputs.emplace_back("itsalppar", "TRK", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("ITS/Config/AlpideParam"));

  // outputs.emplace_back("TRK", "TRACKCLSID", 0, Lifetime::Timeframe);
  // outputs.emplace_back("TRK", "TRKTrackROF", 0, Lifetime::Timeframe);
  // outputs.emplace_back("TRK", "VERTICES", 0, Lifetime::Timeframe);
  // outputs.emplace_back("TRK", "VERTICESROF", 0, Lifetime::Timeframe);
  // outputs.emplace_back("TRK", "IRFRAMES", 0, Lifetime::Timeframe);

  if (useMC) {
    // outputs.emplace_back("TRK", "VERTICESMCTR", 0, Lifetime::Timeframe);
    // outputs.emplace_back("TRK", "VERTICESMCPUR", 0, Lifetime::Timeframe);
    outputs.emplace_back("TRK", "TRACKSMCTR", 0, Lifetime::Timeframe);
    // outputs.emplace_back("TRK", "TRKTrackMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "trk-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackerDPL>(ggRequest,
                                            useMC,
                                            hitRecoConfig,
                                            clusterRecoConfig,
                                            dType)},
    Options{}};
}

} // namespace trk
} // namespace o2
