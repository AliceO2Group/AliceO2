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

/// \file TrackFinderOriginalSpec.cxx
/// \brief Implementation of a data processor to read clusters, reconstruct tracks and send them
///
/// \author Philippe Pillot, Subatech

#include "TrackFinderOriginalSpec.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DetectorsBase/Propagator.h"
#include "MCHTracking/TrackParam.h"
#include "MCHTracking/Track.h"
#include "MCHTracking/TrackFinderOriginal.h"
#include "MCHTracking/TrackExtrap.h"

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
  TrackFinderTask(std::shared_ptr<base::GRPGeomRequest> req) : mCCDBRequest(req) {}

  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the track extrapolation tools

    LOG(info) << "initializing track finder";

    if (mCCDBRequest) {
      base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    } else {
      auto grpFile = ic.options().get<std::string>("grp-file");
      if (std::filesystem::exists(grpFile)) {
        const auto grp = parameters::GRPObject::loadFrom(grpFile);
        base::Propagator::initFieldFromGRP(grp);
        TrackExtrap::setField();
      } else {
        float l3Current = ic.options().get<float>("l3Current");
        float dipoleCurrent = ic.options().get<float>("dipoleCurrent");
        mTrackFinder.initField(l3Current, dipoleCurrent);
      }
    }

    auto config = ic.options().get<std::string>("mch-config");
    if (!config.empty()) {
      o2::conf::ConfigurableParam::updateFromFile(config, "MCHTracking", true);
    }
    mTrackFinder.init();

    auto debugLevel = ic.options().get<int>("mch-debug");
    mTrackFinder.debug(debugLevel);

    auto stop = [this]() {
      mTrackFinder.printStats();
      mTrackFinder.printTimers();
      LOG(info) << "tracking duration = " << mElapsedTime.count() << " s";
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj)
  {
    /// finalize the track extrapolation setting
    if (mCCDBRequest && base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
      if (matcher == framework::ConcreteDataMatcher("GLO", "GRPMAGFIELD", 0)) {
        TrackExtrap::setField();
      }
    }
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// for each event in the current TF, read the clusters and find tracks, then send them all

    if (mCCDBRequest) {
      base::GRPGeomHelper::instance().checkUpdates(pc);
    }

    // get the input messages with clusters
    auto clusterROFs = pc.inputs().get<gsl::span<ROFRecord>>("clusterrofs");
    auto clustersIn = pc.inputs().get<gsl::span<Cluster>>("clusters");

    // LOG(info) << "received time frame with " << clusterROFs.size() << " interactions";

    // create the output messages for tracks and attached clusters
    auto& trackROFs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"trackrofs"});
    auto& mchTracks = pc.outputs().make<std::vector<TrackMCH>>(OutputRef{"tracks"});
    auto& usedClusters = pc.outputs().make<std::vector<Cluster>>(OutputRef{"trackclusters"});

    trackROFs.reserve(clusterROFs.size());
    for (const auto& clusterROF : clusterROFs) {

      // LOG(info) << "processing interaction: " << clusterROF.getBCData() << "...";

      // sort the input clusters of the current event per chamber
      std::array<std::list<const Cluster*>, 10> clusters{};
      for (const auto& cluster : clustersIn.subspan(clusterROF.getFirstIdx(), clusterROF.getNEntries())) {
        clusters[cluster.getChamberId()].emplace_back(&cluster);
      }

      // run the track finder
      auto tStart = std::chrono::high_resolution_clock::now();
      const auto& tracks = mTrackFinder.findTracks(clusters);
      auto tEnd = std::chrono::high_resolution_clock::now();
      mElapsedTime += tEnd - tStart;

      // fill the ouput messages
      int trackOffset(mchTracks.size());
      writeTracks(tracks, mchTracks, usedClusters);
      trackROFs.emplace_back(clusterROF.getBCData(), trackOffset, mchTracks.size() - trackOffset,
                             clusterROF.getBCWidth());
    }
  }

 private:
  //_________________________________________________________________________________________________
  void writeTracks(const std::list<Track>& tracks,
                   std::vector<TrackMCH, o2::pmr::polymorphic_allocator<TrackMCH>>& mchTracks,
                   std::vector<Cluster, o2::pmr::polymorphic_allocator<Cluster>>& usedClusters) const
  {
    /// fill the output messages with tracks and attached clusters

    for (const auto& track : tracks) {

      TrackParam paramAtMID(track.last());
      if (!TrackExtrap::extrapToMID(paramAtMID)) {
        LOG(warning) << "propagation to MID failed --> track discarded";
        continue;
      }

      const auto& param = track.first();
      mchTracks.emplace_back(param.getZ(), param.getParameters(), param.getCovariances(),
                             param.getTrackChi2(), usedClusters.size(), track.getNClusters(),
                             paramAtMID.getZ(), paramAtMID.getParameters(), paramAtMID.getCovariances());

      for (const auto& param : track) {
        usedClusters.emplace_back(*param.getClusterPtr());
      }
    }
  }

  std::shared_ptr<base::GRPGeomRequest> mCCDBRequest{}; ///< pointer to the CCDB requests
  TrackFinderOriginal mTrackFinder{};                   ///< track finder
  std::chrono::duration<double> mElapsedTime{};         ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackFinderOriginalSpec(const char* specName, bool disableCCDBMagField)
{
  std::vector<InputSpec> inputSpecs{};
  inputSpecs.emplace_back(InputSpec{"clusterrofs", "MCH", "CLUSTERROFS", 0, Lifetime::Timeframe});
  inputSpecs.emplace_back(InputSpec{"clusters", "MCH", "GLOBALCLUSTERS", 0, Lifetime::Timeframe});

  std::vector<OutputSpec> outputSpecs{};
  outputSpecs.emplace_back(OutputSpec{{"trackrofs"}, "MCH", "TRACKROFS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"tracks"}, "MCH", "TRACKS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"trackclusters"}, "MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe});

  auto ccdbRequest = disableCCDBMagField ? nullptr
                                         : std::make_shared<base::GRPGeomRequest>(false,                      // orbitResetTime
                                                                                  false,                      // GRPECS=true
                                                                                  false,                      // GRPLHCIF
                                                                                  true,                       // GRPMagField
                                                                                  false,                      // askMatLUT
                                                                                  base::GRPGeomRequest::None, // geometry
                                                                                  inputSpecs);

  return DataProcessorSpec{
    specName,
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TrackFinderTask>(ccdbRequest)},
    Options{{"l3Current", VariantType::Float, -30000.0f, {"L3 current"}},
            {"dipoleCurrent", VariantType::Float, -6000.0f, {"Dipole current"}},
            {"grp-file", VariantType::String, o2::base::NameConf::getGRPFileName(), {"Name of the grp file"}},
            {"mch-config", VariantType::String, "", {"JSON or INI file with tracking parameters"}},
            {"mch-debug", VariantType::Int, 0, {"debug level"}}}};
}

} // namespace mch
} // namespace o2
