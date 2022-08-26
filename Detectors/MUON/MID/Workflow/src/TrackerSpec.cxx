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

/// \file   MID/Workflow/src/TrackerSpec.cxx
/// \brief  Data processor spec for MID tracker device
/// \author Gabriele G. Fronze <gfronze at cern.ch>
/// \date   9 July 2018

#include "MIDWorkflow/TrackerSpec.h"

#include <chrono>
#include "Framework/DataRefUtils.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "DataFormatsMID/Cluster.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/Track.h"
#include "DataFormatsMID/MCClusterLabel.h"
#include "DetectorsBase/GeometryManager.h"
#include "MIDTracking/HitMapBuilder.h"
#include "MIDTracking/Tracker.h"
#include "MIDSimulation/TrackLabeler.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsBase/GRPGeomHelper.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{
class TrackerDeviceDPL
{
 public:
  TrackerDeviceDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr, bool isMC) : mGGCCDBRequest(gr), mIsMC(isMC) {}
  ~TrackerDeviceDPL() = default;

  void init(o2::framework::InitContext& ic)
  {
    o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
    mKeepAll = !ic.options().get<bool>("mid-tracker-keep-best");

    auto stop = [this]() {
      double scaleFactor = (mNROFs == 0) ? 0. : 1.e6 / mNROFs;
      LOG(info) << "Processing time / " << mNROFs << " ROFs: full: " << mTimer.count() * scaleFactor << " us  tracking: " << mTimerTracker.count() * scaleFactor << " us  hitMapBuilder: " << mTimerBuilder.count() << " us";
    };
    ic.services().get<of::CallbackService>().set(of::CallbackService::Id::Stop, stop);
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    auto tStart = std::chrono::high_resolution_clock::now();
    updateTimeDependentParams(pc);

    auto clusters = pc.inputs().get<gsl::span<Cluster>>("mid_clusters");

    auto inROFRecords = pc.inputs().get<gsl::span<ROFRecord>>("mid_clusters_rof");

    auto tAlgoStart = std::chrono::high_resolution_clock::now();
    mTracker->process(clusters, inROFRecords);
    mTimerTracker += std::chrono::high_resolution_clock::now() - tAlgoStart;

    tAlgoStart = std::chrono::high_resolution_clock::now();
    std::vector<Track> tracks = mTracker->getTracks();
    mHitMapBuilder->process(tracks, clusters);
    mTimerBuilder += std::chrono::high_resolution_clock::now() - tAlgoStart;

    if (mIsMC) {
      std::unique_ptr<const o2::dataformats::MCTruthContainer<MCClusterLabel>> labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<MCClusterLabel>*>("mid_clusterlabels");
      mTrackLabeler.process(mTracker->getClusters(), tracks, *labels);
      pc.outputs().snapshot(of::Output{"MID", "TRACKLABELS", 0, of::Lifetime::Timeframe}, mTrackLabeler.getTracksLabels());
      LOG(debug) << "Sent " << mTrackLabeler.getTracksLabels().size() << " indexed tracks.";
      pc.outputs().snapshot(of::Output{"MID", "TRCLUSLABELS", 0, of::Lifetime::Timeframe}, mTrackLabeler.getTrackClustersLabels());
      LOG(debug) << "Sent " << mTrackLabeler.getTrackClustersLabels().getIndexedSize() << " indexed track clusters.";
    }

    pc.outputs().snapshot(of::Output{"MID", "TRACKS", 0, of::Lifetime::Timeframe}, tracks);
    LOG(debug) << "Sent " << tracks.size() << " tracks.";
    pc.outputs().snapshot(of::Output{"MID", "TRACKCLUSTERS", 0, of::Lifetime::Timeframe}, mTracker->getClusters());
    LOG(debug) << "Sent " << mTracker->getClusters().size() << " track clusters.";

    pc.outputs().snapshot(of::Output{"MID", "TRACKROFS", 0, of::Lifetime::Timeframe}, mTracker->getTrackROFRecords());
    LOG(debug) << "Sent " << mTracker->getTrackROFRecords().size() << " ROFs.";
    pc.outputs().snapshot(of::Output{"MID", "TRCLUSROFS", 0, of::Lifetime::Timeframe}, mTracker->getClusterROFRecords());
    LOG(debug) << "Sent " << mTracker->getClusterROFRecords().size() << " ROFs.";

    mTimer += std::chrono::high_resolution_clock::now() - tStart;
    mNROFs += inROFRecords.size();
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
  {
    if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
      return;
    }
  }

 private:
  void updateTimeDependentParams(o2::framework::ProcessingContext& pc)
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    static bool initOnceDone = false;
    if (!initOnceDone) {
      initOnceDone = true;
      auto geoTrans = createTransformationFromManager(gGeoManager);
      mTracker = std::make_unique<Tracker>(geoTrans);
      if (!mTracker->init(mKeepAll)) {
        LOG(error) << "Initialization of MID tracker device failed";
      }
      mHitMapBuilder = std::make_unique<HitMapBuilder>(geoTrans);
    }
  }

  bool mIsMC = false;
  bool mKeepAll = false;
  TrackLabeler mTrackLabeler{};
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::unique_ptr<Tracker> mTracker{nullptr};
  std::unique_ptr<HitMapBuilder> mHitMapBuilder{nullptr};
  std::chrono::duration<double> mTimer{0};        ///< full timer
  std::chrono::duration<double> mTimerTracker{0}; ///< tracker timer
  std::chrono::duration<double> mTimerBuilder{0}; ///< hit map builder timer
  unsigned int mNROFs{0};                         /// Total number of processed ROFs
};

framework::DataProcessorSpec getTrackerSpec(bool isMC)
{
  std::vector<of::InputSpec> inputSpecs{of::InputSpec{"mid_clusters", "MID", "CLUSTERS"}, of::InputSpec{"mid_clusters_rof", "MID", "CLUSTERSROF"}};
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              false,                             // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              false,                             // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              inputSpecs,
                                                              true);
  std::vector<of::OutputSpec> outputSpecs{
    of::OutputSpec{"MID", "TRACKS"},
    of::OutputSpec{"MID", "TRACKCLUSTERS"},
    of::OutputSpec{"MID", "TRACKROFS"},
    of::OutputSpec{"MID", "TRCLUSROFS"}};

  if (isMC) {
    inputSpecs.emplace_back(of::InputSpec{"mid_clusterlabels", "MID", "CLUSTERSLABELS"});

    outputSpecs.emplace_back(of::OutputSpec{"MID", "TRACKLABELS"});
    outputSpecs.emplace_back(of::OutputSpec{"MID", "TRCLUSLABELS"});
  }

  return of::DataProcessorSpec{
    "MIDTracker",
    {inputSpecs},
    {outputSpecs},
    of::adaptFromTask<o2::mid::TrackerDeviceDPL>(ggRequest, isMC),
    of::Options{{"mid-tracker-keep-best", of::VariantType::Bool, false, {"Keep only best track (default is keep all)"}}}};
}
} // namespace mid
} // namespace o2
