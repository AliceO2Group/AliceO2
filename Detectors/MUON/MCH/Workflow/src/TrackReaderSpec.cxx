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

#include "MCHWorkflow/TrackReaderSpec.h"

#include "DPLUtils/RootTreeReader.h"
#include "CommonUtils/StringUtils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Lifetime.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include <iostream>
#include <vector>

using namespace o2::framework;

namespace o2::mch
{

template <typename T>
void printBranch(char* data, const char* what)
{
  auto tdata = reinterpret_cast<std::vector<T>*>(data);
  LOGP(info, "MCH {:d} {:s}", tdata->size(), what);
}

RootTreeReader::SpecialPublishHook logging{
  [](std::string_view name, ProcessingContext&, Output const&, char* data) -> bool {
    if (name == "trackrofs") {
      printBranch<ROFRecord>(data, "ROFS");
    }
    if (name == "trackclusters") {
      printBranch<Cluster>(data, "CLUSTERS");
    }
    if (name == "tracks") {
      printBranch<TrackMCH>(data, "TRACKS");
    }
    if (name == "trackdigits") {
      printBranch<Digit>(data, "DIGITS");
    }
    if (name == "tracklabels") {
      auto tdata = reinterpret_cast<std::vector<o2::MCCompLabel>*>(data);
      LOGP(info, "MCH {:d} {:s}", tdata->size(), "LABELS");
    }
    return false;
  }};

struct TrackReader {
  std::unique_ptr<RootTreeReader> mTreeReader;
  bool mUseMC = false;
  bool mDigits = false;
  TrackReader(bool useMC, bool digits) : mUseMC(useMC), mDigits(digits) {}
  void init(InitContext& ic)
  {
    if (!mUseMC) {
      LOGP(warning, "Not reading MCH Track Labels");
    }
    auto treeName = "o2sim";
    auto fileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")), ic.options().get<std::string>("infile"));
    auto nofEntries{-1};
    if (mUseMC) {
      if (mDigits) {
        mTreeReader = std::make_unique<RootTreeReader>(
          treeName,
          fileName.c_str(),
          nofEntries,
          RootTreeReader::PublishingMode::Single,
          RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MCH", "TRACKROFS", 0}, "trackrofs"},
          RootTreeReader::BranchDefinition<std::vector<TrackMCH>>{Output{"MCH", "TRACKS", 0}, "tracks"},
          RootTreeReader::BranchDefinition<std::vector<Cluster>>{Output{"MCH", "TRACKCLUSTERS", 0}, "trackclusters"},
          RootTreeReader::BranchDefinition<std::vector<Digit>>{Output{"MCH", "TRACKDIGITS", 0}, "trackdigits"},
          RootTreeReader::BranchDefinition<std::vector<o2::MCCompLabel>>{Output{"MCH", "TRACKLABELS", 0}, "tracklabels"},
          &logging);
      } else {
        mTreeReader = std::make_unique<RootTreeReader>(
          treeName,
          fileName.c_str(),
          nofEntries,
          RootTreeReader::PublishingMode::Single,
          RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MCH", "TRACKROFS", 0}, "trackrofs"},
          RootTreeReader::BranchDefinition<std::vector<TrackMCH>>{Output{"MCH", "TRACKS", 0}, "tracks"},
          RootTreeReader::BranchDefinition<std::vector<Cluster>>{Output{"MCH", "TRACKCLUSTERS", 0}, "trackclusters"},
          RootTreeReader::BranchDefinition<std::vector<o2::MCCompLabel>>{Output{"MCH", "TRACKLABELS", 0}, "tracklabels"},
          &logging);
      }
    } else {
      if (mDigits) {
        mTreeReader = std::make_unique<RootTreeReader>(
          treeName,
          fileName.c_str(),
          nofEntries,
          RootTreeReader::PublishingMode::Single,
          RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MCH", "TRACKROFS", 0}, "trackrofs"},
          RootTreeReader::BranchDefinition<std::vector<TrackMCH>>{Output{"MCH", "TRACKS", 0}, "tracks"},
          RootTreeReader::BranchDefinition<std::vector<Cluster>>{Output{"MCH", "TRACKCLUSTERS", 0}, "trackclusters"},
          RootTreeReader::BranchDefinition<std::vector<Digit>>{Output{"MCH", "TRACKDIGITS", 0}, "trackdigits"},
          &logging);
      } else {
        mTreeReader = std::make_unique<RootTreeReader>(
          treeName,
          fileName.c_str(),
          nofEntries,
          RootTreeReader::PublishingMode::Single,
          RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MCH", "TRACKROFS", 0}, "trackrofs"},
          RootTreeReader::BranchDefinition<std::vector<TrackMCH>>{Output{"MCH", "TRACKS", 0}, "tracks"},
          RootTreeReader::BranchDefinition<std::vector<Cluster>>{Output{"MCH", "TRACKCLUSTERS", 0}, "trackclusters"},
          &logging);
      }
    }
  }

  void
    run(ProcessingContext& pc)
  {
    if (mTreeReader->next()) {
      (*mTreeReader)(pc);
    } else {
      pc.services().get<ControlService>().endOfStream();
    }
  }
};

DataProcessorSpec getTrackReaderSpec(bool useMC, const char* specName, bool digits)
{
  std::vector<OutputSpec> outputSpecs;
  outputSpecs.emplace_back(OutputSpec{{"tracks"}, "MCH", "TRACKS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"trackrofs"}, "MCH", "TRACKROFS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"trackclusters"}, "MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe});
  if (digits) {
    outputSpecs.emplace_back(OutputSpec{{"trackdigits"}, "MCH", "TRACKDIGITS", 0, Lifetime::Timeframe});
  }
  if (useMC) {
    outputSpecs.emplace_back(OutputSpec{{"tracklabels"}, "MCH", "TRACKLABELS", 0, Lifetime::Timeframe});
  }

  auto options = Options{
    {"infile", VariantType::String, "mchtracks.root", {"name of the input track file"}},
    {"input-dir", VariantType::String, "none", {"Input directory"}}};

  return DataProcessorSpec{
    specName,
    Inputs{},
    outputSpecs,
    adaptFromTask<TrackReader>(useMC, digits),
    options};
}
} // namespace o2::mch
