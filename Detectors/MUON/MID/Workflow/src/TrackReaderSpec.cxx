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

#include "MIDWorkflow/TrackReaderSpec.h"

#include "DPLUtils/RootTreeReader.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Lifetime.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/Cluster.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/Track.h"
#include "DataFormatsMID/MCClusterLabel.h"
#include <iostream>
#include <vector>

using namespace o2::framework;

namespace o2::mid
{

template <typename T>
void printBranch(char* data, const char* what)
{
  auto tdata = reinterpret_cast<std::vector<T>*>(data);
  LOGP(info, "MID {:d} {:s}", tdata->size(), what);
}

RootTreeReader::SpecialPublishHook logging{
  [](std::string_view name, ProcessingContext&, Output const&, char* data) -> bool {
    if (name == "MIDTrackROF") {
      printBranch<ROFRecord>(data, "TRACKROFS");
    }
    if (name == "MIDTrackClusterROF") {
      printBranch<ROFRecord>(data, "TRCLUSROFS");
    }
    if (name == "MIDTrack") {
      printBranch<Track>(data, "TRACKS");
    }
    if (name == "MIDTrackCluster") {
      printBranch<Cluster>(data, "TRACKCLUSTERS");
    }
    if (name == "MIDTrackLabels") {
      printBranch<MCCompLabel>(data, "TRACKLABELS");
    }
    if (name == "MIDTrackClusterLabels") {
      auto tdata = reinterpret_cast<o2::dataformats::MCTruthContainer<MCClusterLabel>*>(data);
      LOGP(info, "MID {:d} {:s}", tdata->getNElements(), "TRCLUSLABELS");
    }
    return false;
  }};

struct TrackReader {
  std::unique_ptr<RootTreeReader> mTreeReader;
  bool mUseMC = false;
  TrackReader(bool useMC = false) : mUseMC(useMC) {}
  void init(InitContext& ic)
  {
    if (!mUseMC) {
      LOGP(warning, "Not reading MID Track Labels");
    }
    auto treeName = "midreco";
    auto fileName = ic.options().get<std::string>("infile");
    auto nofEntries{-1};
    if (mUseMC) {
      mTreeReader = std::make_unique<RootTreeReader>(
        treeName,
        fileName.c_str(),
        nofEntries,
        RootTreeReader::PublishingMode::Single,
        RootTreeReader::BranchDefinition<std::vector<Track>>{Output{"MID", "TRACKS", 0}, "MIDTrack"},
        RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MID", "TRACKROFS", 0}, "MIDTrackROF"},
        RootTreeReader::BranchDefinition<std::vector<Cluster>>{Output{"MID", "TRACKCLUSTERS", 0}, "MIDTrackCluster"},
        RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MID", "TRCLUSROFS", 0}, "MIDTrackClusterROF"},
        RootTreeReader::BranchDefinition<std::vector<MCCompLabel>>{Output{"MID", "TRACKLABELS", 0}, "MIDTrackLabels"},
        RootTreeReader::BranchDefinition<o2::dataformats::MCTruthContainer<MCClusterLabel>>{Output{"MID", "TRCLUSLABELS", 0}, "MIDTrackClusterLabels"},
        &logging);
    } else {
      mTreeReader = std::make_unique<RootTreeReader>(
        treeName,
        fileName.c_str(),
        nofEntries,
        RootTreeReader::PublishingMode::Single,
        RootTreeReader::BranchDefinition<std::vector<Track>>{Output{"MID", "TRACKS", 0}, "MIDTrack"},
        RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MID", "TRACKROFS", 0}, "MIDTrackROF"},
        RootTreeReader::BranchDefinition<std::vector<Cluster>>{Output{"MID", "TRACKCLUSTERS", 0}, "MIDTrackCluster"},
        RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MID", "TRCLUSROFS", 0}, "MIDTrackClusterROF"},
        &logging);
    }
  }

  void run(ProcessingContext& pc)
  {
    if (mTreeReader->next()) {
      (*mTreeReader)(pc);
    } else {
      pc.services().get<ControlService>().endOfStream();
    }
  }
};

DataProcessorSpec getTrackReaderSpec(bool useMC, const char* specName)
{
  std::vector<OutputSpec> outputSpecs;
  outputSpecs.emplace_back(OutputSpec{{"tracks"}, "MID", "TRACKS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"tracksrof"}, "MID", "TRACKROFS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"trackclusters"}, "MID", "TRACKCLUSTERS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"trclusrof"}, "MID", "TRCLUSROFS", 0, Lifetime::Timeframe});
  if (useMC) {
    outputSpecs.emplace_back(OutputSpec{{"tracklabels"}, "MID", "TRACKLABELS", 0, Lifetime::Timeframe});
    outputSpecs.emplace_back(OutputSpec{{"trcluslabels"}, "MID", "TRCLUSLABELS", 0, Lifetime::Timeframe});
  }

  auto options = Options{
    {"infile", VariantType::String, "mid-reco.root", {"name of the input track file"}},
  };

  return DataProcessorSpec{
    specName,
    Inputs{},
    outputSpecs,
    adaptFromTask<TrackReader>(useMC),
    options};
}
} // namespace o2::mid
