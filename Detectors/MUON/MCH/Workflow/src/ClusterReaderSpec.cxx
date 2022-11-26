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

#include "MCHWorkflow/ClusterReaderSpec.h"

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
#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsMCH/Digit.h"

using namespace o2::framework;

namespace o2::mch
{

struct ClusterReader {
  std::unique_ptr<RootTreeReader> mTreeReader;
  bool mUseMC = false;
  bool mGlobal = false;
  bool mDigits = false;

  ClusterReader(bool useMC, bool global, bool digits) : mUseMC(useMC), mGlobal(global), mDigits(digits) {}

  void init(InitContext& ic)
  {
    auto treeName = "o2sim";
    auto fileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")), ic.options().get<std::string>("infile"));
    auto nofEntries{-1};
    auto clusterDescription = mGlobal ? header::DataDescription{"GLOBALCLUSTERS"} : header::DataDescription{"CLUSTERS"};
    if (mUseMC) {
      // not clear where the MClabels come from
      LOG(warn) << "Disabling clusters MC labels reading";
      mUseMC = false;
    }
    if (mUseMC) {
      if (mDigits) {
        mTreeReader = std::make_unique<RootTreeReader>(
          treeName,
          fileName.c_str(),
          nofEntries,
          RootTreeReader::PublishingMode::Single,
          RootTreeReader::BranchDefinition<std::vector<Cluster>>{Output{"MCH", clusterDescription, 0}, "clusters"},
          RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MCH", "CLUSTERROFS", 0}, "clusterrofs"},
          RootTreeReader::BranchDefinition<std::vector<Digit>>{Output{"MCH", "CLUSTERDIGITS", 0}, "clusterdigits"},
          RootTreeReader::BranchDefinition<dataformats::MCTruthContainer<MCCompLabel>>{Output{"MCH", "CLUSTERLABELS", 0}, "clusterlabels"});
      } else {
        mTreeReader = std::make_unique<RootTreeReader>(
          treeName,
          fileName.c_str(),
          nofEntries,
          RootTreeReader::PublishingMode::Single,
          RootTreeReader::BranchDefinition<std::vector<Cluster>>{Output{"MCH", clusterDescription, 0}, "clusters"},
          RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MCH", "CLUSTERROFS", 0}, "clusterrofs"},
          RootTreeReader::BranchDefinition<dataformats::MCTruthContainer<MCCompLabel>>{Output{"MCH", "CLUSTERLABELS", 0}, "clusterlabels"});
      }
    } else {
      if (mDigits) {
        mTreeReader = std::make_unique<RootTreeReader>(
          treeName,
          fileName.c_str(),
          nofEntries,
          RootTreeReader::PublishingMode::Single,
          RootTreeReader::BranchDefinition<std::vector<Cluster>>{Output{"MCH", clusterDescription, 0}, "clusters"},
          RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MCH", "CLUSTERROFS", 0}, "clusterrofs"},
          RootTreeReader::BranchDefinition<std::vector<Digit>>{Output{"MCH", "CLUSTERDIGITS", 0}, "clusterdigits"});
      } else {
        mTreeReader = std::make_unique<RootTreeReader>(
          treeName,
          fileName.c_str(),
          nofEntries,
          RootTreeReader::PublishingMode::Single,
          RootTreeReader::BranchDefinition<std::vector<Cluster>>{Output{"MCH", clusterDescription, 0}, "clusters"},
          RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{Output{"MCH", "CLUSTERROFS", 0}, "clusterrofs"});
      }
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

DataProcessorSpec getClusterReaderSpec(bool useMC, const char* specName, bool global, bool digits)
{
  auto clusterDescription = global ? header::DataDescription{"GLOBALCLUSTERS"} : header::DataDescription{"CLUSTERS"};
  std::vector<OutputSpec> outputSpecs;
  outputSpecs.emplace_back(OutputSpec{{"clusters"}, "MCH", clusterDescription, 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"clusterrofs"}, "MCH", "CLUSTERROFS", 0, Lifetime::Timeframe});
  if (digits) {
    outputSpecs.emplace_back(OutputSpec{{"clusterdigits"}, "MCH", "CLUSTERDIGITS", 0, Lifetime::Timeframe});
  }
  if (useMC) {
    outputSpecs.emplace_back(OutputSpec{{"clusterlabels"}, "MCH", "CLUSTERLABELS", 0, Lifetime::Timeframe});
  }

  auto options = Options{
    {"infile", VariantType::String, "mchclusters.root", {"name of the input cluster file"}},
    {"input-dir", VariantType::String, "none", {"Input directory"}}};

  return DataProcessorSpec{
    specName,
    Inputs{},
    outputSpecs,
    adaptFromTask<ClusterReader>(useMC, global, digits),
    options};
}
} // namespace o2::mch
