// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFinderOriginalSpec.cxx
/// \brief Implementation of a data processor to read clusters, reconstruct tracks and send them
///
/// \author Philippe Pillot, Subatech

#include "TrackFinderOriginalSpec.h"

#include <chrono>
#include <array>
#include <list>
#include <stdexcept>

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "DataFormatsMCH/TrackMCH.h"
#include "MCHBase/ClusterBlock.h"
#include "MCHTracking/TrackParam.h"
#include "MCHTracking/Cluster.h"
#include "MCHTracking/Track.h"
#include "MCHTracking/TrackFinderOriginal.h"

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
    mTrackFinder.init(l3Current, dipoleCurrent);

    auto moreCandidates = ic.options().get<bool>("moreCandidates");
    mTrackFinder.findMoreTrackCandidates(moreCandidates);

    auto refineTracks = !ic.options().get<bool>("noRefinement");
    mTrackFinder.refineTracks(refineTracks);

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
    /// read the clusters of the current event, find tracks and send them

    // get the input clusters
    auto clustersIn = pc.inputs().get<gsl::span<ClusterStruct>>("clusters");
    std::array<std::list<Cluster>, 10> clusters{};
    for (const auto& cluster : clustersIn) {
      clusters[cluster.getChamberId()].emplace_back(cluster);
    }

    // run the track finder
    auto tStart = std::chrono::high_resolution_clock::now();
    const auto& tracks = mTrackFinder.findTracks(&clusters);
    auto tEnd = std::chrono::high_resolution_clock::now();
    mElapsedTime += tEnd - tStart;

    // create the output messages for tracks and attached clusters
    auto& mchTracks = pc.outputs().make<std::vector<TrackMCH>>(Output{"MCH", "TRACKS", 0, Lifetime::Timeframe});
    auto& usedClusters = pc.outputs().make<std::vector<ClusterStruct>>(Output{"MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe});

    // write the messages
    if (tracks.size() > 0) {
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

    // avoid memory reallocation
    mchTracks.reserve(tracks.size());
    usedClusters.reserve(20. * tracks.size());

    for (const auto& track : tracks) {

      const auto& param = track.first();
      mchTracks.emplace_back(param.getZ(), param.getParameters(), param.getCovariances(),
                             param.getTrackChi2(), usedClusters.size(), track.getNClusters());

      for (const auto& param : track) {
        usedClusters.emplace_back(param.getClusterPtr()->getClusterStruct());
      }
    }
  }

  TrackFinderOriginal mTrackFinder{};           ///< track finder
  std::chrono::duration<double> mElapsedTime{}; ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackFinderOriginalSpec()
{
  return DataProcessorSpec{
    "TrackFinderOriginal",
    Inputs{InputSpec{"clusters", "MCH", "CLUSTERS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{"MCH", "TRACKS", 0, Lifetime::Timeframe},
            OutputSpec{"MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TrackFinderTask>()},
    Options{{"l3Current", VariantType::Float, -30000.0f, {"L3 current"}},
            {"dipoleCurrent", VariantType::Float, -6000.0f, {"Dipole current"}},
            {"moreCandidates", VariantType::Bool, false, {"Find more track candidates"}},
            {"noRefinement", VariantType::Bool, false, {"Disable the track refinement"}},
            {"debug", VariantType::Int, 0, {"debug level"}}}};
}

} // namespace mch
} // namespace o2
