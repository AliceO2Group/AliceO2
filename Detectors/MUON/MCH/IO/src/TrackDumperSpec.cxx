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

#include "TrackDumperSpec.h"

#include "CommonUtils/StringUtils.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Lifetime.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <iostream>
#include <vector>

using namespace o2::framework;

namespace o2::mch
{
void dump(std::ostream& os, const o2::mch::TrackMCH& t)
{
  auto pt = std::sqrt(t.getPx() * t.getPx() + t.getPy() * t.getPy());
  os << fmt::format("({:s}) p {:7.2f} pt {:7.2f} nclusters: {} \n", t.getSign() == -1 ? "-" : "+", t.getP(), pt, t.getNClusters());
}

struct TrackDumper {

  bool mUseMC{false};

  TrackDumper(bool useMC) : mUseMC(useMC) {}

  void init(InitContext& ic)
  {
  }

  void process(gsl::span<const o2::mch::ROFRecord> rofs,
               gsl::span<const o2::mch::TrackMCH> tracks,
               gsl::span<const o2::MCCompLabel> labels)
  {
    for (const auto& rof : rofs) {
      const auto& tracksInRof = tracks.subspan(rof.getFirstIdx(), rof.getNEntries());
      for (auto i = 0; i < tracksInRof.size(); i++) {
        const auto& t = tracksInRof[i];
        std::cout << fmt::format("Track {:4d}/{:4d} : ", i + 1, tracksInRof.size());
        dump(std::cout, t);
        if (!labels.empty()) {
          const auto& labelsInRof = labels.subspan(rof.getFirstIdx(), rof.getNEntries());
          labelsInRof[i].print();
        }
      }
    }
  }

  void run(ProcessingContext& pc)
  {
    auto tracks = pc.inputs().get<gsl::span<o2::mch::TrackMCH>>("tracks");
    auto rofs = pc.inputs().get<gsl::span<o2::mch::ROFRecord>>("trackrofs");
    if (mUseMC) {
      auto labels = pc.inputs().get<gsl::span<o2::MCCompLabel>>("tracklabels");
      process(rofs, tracks, labels);
    } else {
      process(rofs, tracks, {});
    }
  }
};

DataProcessorSpec getTrackDumperSpec(bool useMC, const char* specName)
{
  Inputs inputs{};
  inputs.emplace_back("trackrofs", "MCH", "TRACKROFS", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracks", "MCH", "TRACKS", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("tracklabels", "MCH", "TRACKLABELS", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    specName,
    inputs,
    {},
    adaptFromTask<TrackDumper>(useMC),
    {}};
}
} // namespace o2::mch
