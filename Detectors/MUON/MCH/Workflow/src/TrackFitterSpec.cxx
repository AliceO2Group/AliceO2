// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFitterSpec.cxx
/// \brief Implementation of a data processor to read, refit and send tracks with attached clusters
///
/// \author Philippe Pillot, Subatech

#include "TrackFitterSpec.h"

#include <stdexcept>
#include <list>
#include <vector>

#include <gsl/span>

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
#include "MCHTracking/TrackExtrap.h"
#include "MCHTracking/TrackFitter.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TrackFitterTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the track extrapolation tools
    LOG(INFO) << "initializing track fitter";
    auto l3Current = ic.options().get<float>("l3Current");
    auto dipoleCurrent = ic.options().get<float>("dipoleCurrent");
    mTrackFitter.initField(l3Current, dipoleCurrent);
    mTrackFitter.smoothTracks(true);
    TrackExtrap::useExtrapV2();
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// read the tracks with attached clusters, refit them and send the new version

    // get the input tracks and attached clusters
    auto tracksIn = pc.inputs().get<gsl::span<TrackMCH>>("tracks");
    auto clustersIn = pc.inputs().get<gsl::span<ClusterStruct>>("clusters");

    // create the output messages for refitted tracks and attached clusters
    auto& tracksOut = pc.outputs().make<std::vector<TrackMCH>>(Output{"MCH", "TRACKS", 0, Lifetime::Timeframe});
    auto& clustersOut = pc.outputs().make<std::vector<ClusterStruct>>(Output{"MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe});

    // avoid memory reallocation
    tracksOut.reserve(tracksIn.size());
    clustersOut.reserve(clustersIn.size());

    for (const auto& mchTrack : tracksIn) {

      // get the clusters attached to the track
      auto trackClusters = clustersIn.subspan(mchTrack.getFirstClusterIdx(), mchTrack.getNClusters());

      // create the internal track
      Track track{};
      std::list<Cluster> clusters{};
      for (const auto& cluster : trackClusters) {
        clusters.emplace_back(cluster);
        track.createParamAtCluster(clusters.back());
      }

      // refit the track
      try {
        mTrackFitter.fit(track);
      } catch (exception const& e) {
        LOG(ERROR) << "Track fit failed: " << e.what();
        continue;
      }

      // write the refitted track and attached clusters (same as those of the input track)
      const auto& param = track.first();
      tracksOut.emplace_back(param.getZ(), param.getParameters(), param.getCovariances(),
                             param.getTrackChi2(), clustersOut.size(), track.getNClusters());
      clustersOut.insert(clustersOut.end(), trackClusters.begin(), trackClusters.end());
    }
  }

 private:
  TrackFitter mTrackFitter{}; ///< track fitter
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackFitterSpec()
{
  return DataProcessorSpec{
    "TrackFitter",
    Inputs{InputSpec{"tracks", "MCH", "TRACKSIN", 0, Lifetime::Timeframe},
           InputSpec{"clusters", "MCH", "TRACKCLUSTERSIN", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{"MCH", "TRACKS", 0, Lifetime::Timeframe},
            OutputSpec{"MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TrackFitterTask>()},
    Options{{"l3Current", VariantType::Float, -30000.0f, {"L3 current"}},
            {"dipoleCurrent", VariantType::Float, -6000.0f, {"Dipole current"}}}};
}

} // namespace mch
} // namespace o2
