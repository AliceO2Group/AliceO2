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

/// \file TrackFinderSpec.cxx
/// \brief Implementation of a data processor to read clusters, reconstruct tracks and send them
///
/// \author Philippe Pillot, Subatech

#include "TrackFinderSpec.h"

#include <chrono>
#include <unordered_map>
#include <list>
#include <stdexcept>
#include <string>

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "CommonUtils/ConfigurableParam.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "MCHBase/ClusterBlock.h"
#include "MCHTracking/TrackParam.h"
#include "MCHTracking/Cluster.h"
#include "MCHTracking/Track.h"
#include "MCHTracking/TrackFinder.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TrackFinderTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the track extrapolation tools

    LOG(INFO) << "initializing track finder";

    auto l3Current = ic.options().get<float>("l3Current");
    auto dipoleCurrent = ic.options().get<float>("dipoleCurrent");
    auto config = ic.options().get<std::string>("config");
    if (!config.empty()) {
      o2::conf::ConfigurableParam::updateFromFile(config, "MCHTracking", true);
    }
    mTrackFinder.init(l3Current, dipoleCurrent);

    auto debugLevel = ic.options().get<int>("debug");
    mTrackFinder.debug(debugLevel);

    auto stop = [this]() {
      mTrackFinder.printStats();
      mTrackFinder.printTimers();
      LOG(INFO) << "tracking duration = " << mElapsedTime.count() << " s";
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// for each event in the current TF, read the clusters and find tracks, then send them all

    // get the input messages with clusters
    auto clusterROFs = pc.inputs().get<gsl::span<ROFRecord>>("clusterrofs");
    auto clustersIn = pc.inputs().get<gsl::span<ClusterStruct>>("clusters");

    //LOG(INFO) << "received time frame with " << clusterROFs.size() << " interactions";

    // create the output messages for tracks and attached clusters
    auto& trackROFs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"trackrofs"});
    auto& mchTracks = pc.outputs().make<std::vector<TrackMCH>>(OutputRef{"tracks"});
    auto& usedClusters = pc.outputs().make<std::vector<ClusterStruct>>(OutputRef{"trackclusters"});

    trackROFs.reserve(clusterROFs.size());
    for (const auto& clusterROF : clusterROFs) {

      //LOG(INFO) << "processing interaction: " << clusterROF.getBCData() << "...";

      // get the input clusters of the current event
      std::unordered_map<int, std::list<Cluster>> clusters{};
      for (const auto& cluster : clustersIn.subspan(clusterROF.getFirstIdx(), clusterROF.getNEntries())) {
        clusters[cluster.getDEId()].emplace_back(cluster);
      }

      // run the track finder
      auto tStart = std::chrono::high_resolution_clock::now();
      const auto& tracks = mTrackFinder.findTracks(clusters);
      auto tEnd = std::chrono::high_resolution_clock::now();
      mElapsedTime += tEnd - tStart;

      // fill the ouput messages
      trackROFs.emplace_back(clusterROF.getBCData(), mchTracks.size(), tracks.size());
      writeTracks(tracks, mchTracks, usedClusters);
    }
  }

 private:
  //_________________________________________________________________________________________________
  void writeTracks(const std::list<Track>& tracks,
                   std::vector<TrackMCH, o2::pmr::polymorphic_allocator<TrackMCH>>& mchTracks,
                   std::vector<ClusterStruct, o2::pmr::polymorphic_allocator<ClusterStruct>>& usedClusters) const
  {
    /// fill the output messages with tracks and attached clusters

    for (const auto& track : tracks) {

      const auto& param = track.first();
      mchTracks.emplace_back(param.getZ(), param.getParameters(), param.getCovariances(),
                             param.getTrackChi2(), usedClusters.size(), track.getNClusters());

      for (const auto& param : track) {
        usedClusters.emplace_back(param.getClusterPtr()->getClusterStruct());
      }
    }
  }

  TrackFinder mTrackFinder{};                   ///< track finder
  std::chrono::duration<double> mElapsedTime{}; ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackFinderSpec()
{
  return DataProcessorSpec{
    "TrackFinder",
    Inputs{InputSpec{"clusterrofs", "MCH", "CLUSTERROFS", 0, Lifetime::Timeframe},
           InputSpec{"clusters", "MCH", "GLOBALCLUSTERS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{{"trackrofs"}, "MCH", "TRACKROFS", 0, Lifetime::Timeframe},
            OutputSpec{{"tracks"}, "MCH", "TRACKS", 0, Lifetime::Timeframe},
            OutputSpec{{"trackclusters"}, "MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TrackFinderTask>()},
    Options{{"l3Current", VariantType::Float, -30000.0f, {"L3 current"}},
            {"dipoleCurrent", VariantType::Float, -6000.0f, {"Dipole current"}},
            {"config", VariantType::String, "", {"JSON or INI file with tracking parameters"}},
            {"debug", VariantType::Int, 0, {"debug level"}}}};
}

} // namespace mch
} // namespace o2
