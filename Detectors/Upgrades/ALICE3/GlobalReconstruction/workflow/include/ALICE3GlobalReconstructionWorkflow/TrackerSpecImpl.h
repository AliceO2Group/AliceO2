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

#ifndef O2_TRK_TRACKERSPECIMPL_H
#define O2_TRK_TRACKERSPECIMPL_H

#include "ALICE3GlobalReconstructionWorkflow/TrackerSpec.h"

#include "CommonDataFormat/IRFrame.h"
#include "DataFormatsTRK/Cluster.h"
#include "DataFormatsTRK/ROFRecord.h"
#include "DetectorsBase/GeometryManager.h"
#include "Field/MagFieldParam.h"
#include "Field/MagneticField.h"
#include "Framework/ControlService.h"
#include "ITStracking/Tracker.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TRKBase/GeometryTGeo.h"
#include "TRKSimulation/Hit.h"

#include <TFile.h>
#include <TGeoGlobalMagField.h>
#include <TTree.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <numeric>
#include <vector>

namespace o2::trk
{

template <typename TimeFrameT, typename TrackerTraitsT>
void TrackerDPL::runTracking(framework::ProcessingContext& pc, TimeFrameT& timeFrame, TrackerTraitsT& trackerTraits)
{
  o2::its::Tracker<11> itsTracker(&trackerTraits);
  timeFrame.setMemoryPool(mMemoryPool);
  trackerTraits.setMemoryPool(mMemoryPool);
  trackerTraits.setNThreads(mTaskArena->max_concurrency(), mTaskArena);
  trackerTraits.adoptTimeFrame(static_cast<o2::its::TimeFrame<11>*>(&timeFrame));
  itsTracker.adoptTimeFrame(timeFrame);
  trackerTraits.updateTrackingParameters(mTrackingParams);
  timeFrame.initTrackerTopologies(mTrackingParams, 11);

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

    timeFrame.deriveAndInitTiming(layerROFs);

    const float yPlaneMLOT = 0.0010f;
    nRofs = timeFrame.loadROFrameData(layerROFs, layerClusters, layerPatterns, mIsMC ? &layerLabels : nullptr, yPlaneMLOT);
    timeFrame.addTruthSeedingVertices();
  }

  const auto trackingLoopStart = std::chrono::steady_clock::now();
  for (size_t iter{0}; iter < mTrackingParams.size(); ++iter) {
    LOGP(info, "{}", mTrackingParams[iter].asString());
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

  timeFrame.wipe();
}

} // namespace o2::trk

#endif // O2_TRK_TRACKERSPECIMPL_H
