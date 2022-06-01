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

/// \file TrackMatcherSpec.cxx
/// \brief Implementation of a data processor to match MCH and MID tracks
///
/// \author Philippe Pillot, Subatech

#include "TrackMatcherSpec.h"

#include <chrono>
#include <string>
#include <vector>

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/Track.h"
#include "ReconstructionDataFormats/TrackMCHMID.h"
#include "MUONMatching/TrackMatcher.h"

namespace o2
{
namespace muon
{

using namespace o2::framework;

class TrackMatcherTask
{
 public:
  //_________________________________________________________________________________________________
  /// prepare the track matching
  void init(InitContext& ic)
  {
    LOG(info) << "initializing track matching";

    auto config = ic.options().get<std::string>("mch-config");
    if (!config.empty()) {
      conf::ConfigurableParam::updateFromFile(config, "MUONMatching", true);
    }
    mTrackMatcher.init();

    auto stop = [this]() {
      LOG(info) << "track matching duration = " << mElapsedTime.count() << " s";
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  /// run the track matching
  void run(ProcessingContext& pc)
  {
    auto mchROFs = pc.inputs().get<gsl::span<mch::ROFRecord>>("mchrofs");
    auto mchTracks = pc.inputs().get<gsl::span<mch::TrackMCH>>("mchtracks");
    auto midROFs = pc.inputs().get<gsl::span<mid::ROFRecord>>("midrofs");
    auto midTracks = pc.inputs().get<gsl::span<mid::Track>>("midtracks");

    auto tStart = std::chrono::high_resolution_clock::now();
    mTrackMatcher.match(mchROFs, mchTracks, midROFs, midTracks);
    auto tEnd = std::chrono::high_resolution_clock::now();
    mElapsedTime += tEnd - tStart;

    pc.outputs().snapshot(OutputRef{"muontracks"}, mTrackMatcher.getMuons());
  }

 private:
  TrackMatcher mTrackMatcher{};                 ///< MCH-MID track matcher
  std::chrono::duration<double> mElapsedTime{}; ///< timer
};

//_________________________________________________________________________________________________
DataProcessorSpec getTrackMatcherSpec(const char* name)
{
  return DataProcessorSpec{
    name,
    Inputs{InputSpec{"mchrofs", "MCH", "TRACKROFS", 0, Lifetime::Timeframe},
           InputSpec{"mchtracks", "MCH", "TRACKS", 0, Lifetime::Timeframe},
           InputSpec{"midrofs", "MID", "TRACKROFS", 0, Lifetime::Timeframe},
           InputSpec{"midtracks", "MID", "TRACKS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{{"muontracks"}, "GLO", "MTC_MCHMID", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TrackMatcherTask>()},
    Options{{"mch-config", VariantType::String, "", {"JSON or INI file with matching parameters"}}}};
}

} // namespace muon
} // namespace o2
