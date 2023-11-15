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

/// \file   MID/Workflow/src/reco-workflow.cxx
/// \brief  MID reconstruction workflow from digits
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   23 October 2020

#include <array>
#include <string>
#include <vector>
#include "Framework/Variant.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/Cluster.h"
#include "DataFormatsMID/Track.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/MCClusterLabel.h"
#include "MIDWorkflow/ClusterizerSpec.h"
#include "MIDWorkflow/FilteringSpec.h"
#include "MIDWorkflow/FilteringBCSpec.h"
#include "MIDWorkflow/TimingSpec.h"
#include "MIDWorkflow/TrackerSpec.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:MID|mid).*[W,w]riter.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec>
    options{
      {"disable-mc", VariantType::Bool, false, {"Do not propagate MC labels"}},
      {"disable-tracking", VariantType::Bool, false, {"Only run clustering"}},
      {"disable-root-output", VariantType::Bool, false, {"Do not write output to file"}},
      {"change-local-to-BC", VariantType::Int, 0, {"Change the delay between the MID local clock and the BC"}},
      {"disable-filtering", VariantType::Bool, false, {"Do not remove bad channels"}},
      {"enable-filter-BC", VariantType::Bool, false, {"Filter BCs"}},
      {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  bool disableMC = cfgc.options().get<bool>("disable-mc");
  bool disableTracking = cfgc.options().get<bool>("disable-tracking");
  bool disableFile = cfgc.options().get<bool>("disable-root-output");
  auto disableFiltering = cfgc.options().get<bool>("disable-filtering");
  auto localToBC = cfgc.options().get<int>("change-local-to-BC");
  auto filterBC = cfgc.options().get<bool>("enable-filter-BC");
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  WorkflowSpec specs;
  std::string baseDataDesc = "DATA";
  std::string baseROFDesc = "DATAROF";
  std::string baseLabelsDesc = "DATALABELS";
  std::string dataDesc = baseDataDesc;
  std::string rofDesc = baseROFDesc;
  std::string labelsDesc = baseLabelsDesc;
  if (!disableFiltering) {
    std::string prefix = "F";
    std::string outDataDesc = prefix + baseDataDesc;
    specs.emplace_back(o2::mid::getFilteringSpec(!disableMC, dataDesc, outDataDesc));
    dataDesc = outDataDesc;
    rofDesc = prefix + baseROFDesc;
    labelsDesc = prefix + baseLabelsDesc;
  }
  if (localToBC != 0) {
    specs.emplace_back(o2::mid::getTimingSpec(localToBC, rofDesc));
    std::string prefix = "T";
    rofDesc = prefix + baseROFDesc;
  }
  if (filterBC) {
    specs.emplace_back(o2::mid::getFilteringBCSpec(!disableMC, dataDesc));
    std::string prefix = "B";
    dataDesc = prefix + baseDataDesc;
    rofDesc = prefix + baseROFDesc;
    labelsDesc = prefix + baseLabelsDesc;
  }
  specs.emplace_back(o2::mid::getClusterizerSpec(!disableMC, dataDesc, rofDesc, labelsDesc));
  if (!disableTracking) {
    specs.emplace_back(o2::mid::getTrackerSpec(!disableMC, !disableFiltering));
  }
  if (!disableFile) {
    if (disableTracking) {
      specs.emplace_back(MakeRootTreeWriterSpec("MIDRecoWriter",
                                                "mid-reco.root",
                                                "midreco",
                                                MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::Cluster>>{InputSpec{"mid_clusters", o2::header::gDataOriginMID, "CLUSTERS"}, "MIDCluster"},
                                                MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ROFRecord>>{InputSpec{"mid_clusters_rof", o2::header::gDataOriginMID, "CLUSTERSROF"}, "MIDClusterROF"},
                                                MakeRootTreeWriterSpec::BranchDefinition<o2::dataformats::MCTruthContainer<o2::mid::MCClusterLabel>>{InputSpec{"mid_clusters_labels", o2::header::gDataOriginMID, "CLUSTERSLABELS"}, "MIDClusterLabels", disableMC ? 0 : 1})());

    } else {
      specs.emplace_back(MakeRootTreeWriterSpec("MIDRecoWriter",
                                                "mid-reco.root",
                                                "midreco",
                                                MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::Track>>{InputSpec{"mid_tracks", o2::header::gDataOriginMID, "TRACKS"}, "MIDTrack"},
                                                MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::Cluster>>{InputSpec{"mid_trclus", o2::header::gDataOriginMID, "TRACKCLUSTERS"}, "MIDTrackCluster"},
                                                MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ROFRecord>>{InputSpec{"mid_tracks_rof", o2::header::gDataOriginMID, "TRACKROFS"}, "MIDTrackROF"},
                                                MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ROFRecord>>{InputSpec{"mid_trclus_rof", o2::header::gDataOriginMID, "TRCLUSROFS"}, "MIDTrackClusterROF"},
                                                MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::MCCompLabel>>{InputSpec{"mid_track_labels", o2::header::gDataOriginMID, "TRACKLABELS"}, "MIDTrackLabels", disableMC ? 0 : 1},
                                                MakeRootTreeWriterSpec::BranchDefinition<o2::dataformats::MCTruthContainer<o2::mid::MCClusterLabel>>{InputSpec{"mid_trclus_labels", o2::header::gDataOriginMID, "TRCLUSLABELS"}, "MIDTrackClusterLabels", disableMC ? 0 : 1})());
    }
  }

  return specs;
}
