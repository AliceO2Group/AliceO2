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

#include "MCHWorkflow/PreClusterReaderSpec.h"

#include <iostream>
#include <vector>
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Lifetime.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "DPLUtils/RootTreeReader.h"
#include "CommonUtils/StringUtils.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include "MCHBase/PreCluster.h"

using namespace o2::framework;

namespace o2::mch
{

struct PreClusterReader {
  std::unique_ptr<RootTreeReader> mTreeReader;
  bool mUseMC = false;

  PreClusterReader(bool useMC) : mUseMC(useMC) {}

  void init(InitContext& ic)
  {
    auto treeName = "o2sim";
    auto fileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")), ic.options().get<std::string>("infile"));
    auto nofEntries{-1};
    if (mUseMC) {
      // not available (yet?)
      LOG(warn) << "Disabling preclusters MC labels reading";
      mUseMC = false;
    }
    if (mUseMC) {
      mTreeReader = std::make_unique<RootTreeReader>(
        treeName,
        fileName.c_str(),
        nofEntries,
        RootTreeReader::PublishingMode::Single,
        RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MCH", "PRECLUSTERROFS", 0}, "preclusterrofs"},
        RootTreeReader::BranchDefinition<std::vector<PreCluster>>{Output{"MCH", "PRECLUSTERS", 0}, "preclusters"},
        RootTreeReader::BranchDefinition<std::vector<Digit>>{Output{"MCH", "PRECLUSTERDIGITS", 0}, "preclusterdigits"},
        RootTreeReader::BranchDefinition<dataformats::MCTruthContainer<MCCompLabel>>{Output{"MCH", "PRECLUSTERLABELS", 0}, "preclusterlabels"});
    } else {
      mTreeReader = std::make_unique<RootTreeReader>(
        treeName,
        fileName.c_str(),
        nofEntries,
        RootTreeReader::PublishingMode::Single,
        RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MCH", "PRECLUSTERROFS", 0}, "preclusterrofs"},
        RootTreeReader::BranchDefinition<std::vector<PreCluster>>{Output{"MCH", "PRECLUSTERS", 0}, "preclusters"},
        RootTreeReader::BranchDefinition<std::vector<Digit>>{Output{"MCH", "PRECLUSTERDIGITS", 0}, "preclusterdigits"});
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

DataProcessorSpec getPreClusterReaderSpec(bool useMC, const char* specName)
{
  std::vector<OutputSpec> outputSpecs;
  outputSpecs.emplace_back(OutputSpec{{"preclusterrofs"}, "MCH", "PRECLUSTERROFS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"preclusters"}, "MCH", "PRECLUSTERS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"preclusterdigits"}, "MCH", "PRECLUSTERDIGITS", 0, Lifetime::Timeframe});

  auto options = Options{
    {"infile", VariantType::String, "mchpreclusters.root", {"name of the input precluster file"}},
    {"input-dir", VariantType::String, "none", {"Input directory"}}};

  return DataProcessorSpec{
    specName,
    Inputs{},
    outputSpecs,
    adaptFromTask<PreClusterReader>(useMC),
    options};
}
} // namespace o2::mch
