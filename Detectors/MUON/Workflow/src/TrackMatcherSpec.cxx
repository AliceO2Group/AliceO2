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
#include "SimulationDataFormat/MCCompLabel.h"
#include "MUONMatching/TrackMatcher.h"

namespace o2
{
namespace muon
{

using namespace o2::framework;

class TrackMatcherTask
{
 public:
  TrackMatcherTask(bool useMC = false) : mUseMC(useMC) {}

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
    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(stop);
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

    if (mUseMC) {
      auto mchTrackLabels = pc.inputs().get<gsl::span<MCCompLabel>>("mchtracklabels");
      auto midTrackLabels = pc.inputs().get<gsl::span<MCCompLabel>>("midtracklabels");

      std::vector<MCCompLabel> muonLabels;
      for (const auto& muon : mTrackMatcher.getMuons()) {
        const auto& mchTrackLabel = mchTrackLabels[muon.getMCHRef().getIndex()];
        const auto& midTrackLabel = midTrackLabels[muon.getMIDRef().getIndex()];
        muonLabels.push_back(mchTrackLabel);
        // tag fake matching (different labels or at least one of them is fake)
        muonLabels.back().setFakeFlag(midTrackLabel.compare(mchTrackLabel) != 1);
      }

      pc.outputs().snapshot(OutputRef{"muontracklabels"}, muonLabels);
    }
  }

 private:
  bool mUseMC = false;                          ///< MC flag
  TrackMatcher mTrackMatcher{};                 ///< MCH-MID track matcher
  std::chrono::duration<double> mElapsedTime{}; ///< timer
};

//_________________________________________________________________________________________________
DataProcessorSpec getTrackMatcherSpec(bool useMC, const char* name)
{
  std::vector<InputSpec> inputSpecs{InputSpec{"mchrofs", "MCH", "TRACKROFS", 0, Lifetime::Timeframe},
                                    InputSpec{"mchtracks", "MCH", "TRACKS", 0, Lifetime::Timeframe},
                                    InputSpec{"midrofs", "MID", "TRACKROFS", 0, Lifetime::Timeframe},
                                    InputSpec{"midtracks", "MID", "TRACKS", 0, Lifetime::Timeframe}};

  std::vector<OutputSpec> outputSpecs{OutputSpec{{"muontracks"}, "GLO", "MTC_MCHMID", 0, Lifetime::Timeframe}};

  if (useMC) {
    inputSpecs.emplace_back(InputSpec{"mchtracklabels", "MCH", "TRACKLABELS", 0, Lifetime::Timeframe});
    inputSpecs.emplace_back(InputSpec{"midtracklabels", "MID", "TRACKLABELS", 0, Lifetime::Timeframe});

    outputSpecs.emplace_back(OutputSpec{{"muontracklabels"}, "GLO", "MCMTC_MCHMID", 0, Lifetime::Timeframe});
  }

  return DataProcessorSpec{
    name,
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TrackMatcherTask>(useMC)},
    Options{{"mch-config", VariantType::String, "", {"JSON or INI file with matching parameters"}}}};
}

} // namespace muon
} // namespace o2
