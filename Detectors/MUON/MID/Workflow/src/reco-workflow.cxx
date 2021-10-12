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
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/Cluster2D.h"
#include "DataFormatsMID/Cluster3D.h"
#include "DataFormatsMID/Track.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/MCClusterLabel.h"
#include "MIDWorkflow/ClusterizerMCSpec.h"
#include "MIDWorkflow/ClusterizerSpec.h"
#include "MIDWorkflow/TrackerMCSpec.h"
#include "MIDWorkflow/TrackerSpec.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec>
    options{
      {"disable-mc", VariantType::Bool, false, {"Do not propagate MC labels"}},
      {"disable-tracking", VariantType::Bool, false, {"Only run clustering"}},
      {"disable-root-output", VariantType::Bool, false, {"Do not write output to file"}},
      {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  bool disableMC = cfgc.options().get<bool>("disable-mc");
  bool disableTracking = cfgc.options().get<bool>("disable-tracking");
  bool disableFile = cfgc.options().get<bool>("disable-root-output");
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  WorkflowSpec specs;
  specs.emplace_back(disableMC ? o2::mid::getClusterizerSpec() : o2::mid::getClusterizerMCSpec());
  if (!disableTracking) {
    specs.emplace_back(disableMC ? o2::mid::getTrackerSpec() : o2::mid::getTrackerMCSpec());
  }
  if (!disableFile) {
    if (disableTracking) {
      specs.emplace_back(MakeRootTreeWriterSpec("MIDRecoWriter",
                                                "mid-reco.root",
                                                "midreco",
                                                MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::Cluster2D>>{InputSpec{"mid_clusters", o2::header::gDataOriginMID, "CLUSTERS"}, "MIDCluster"},
                                                MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ROFRecord>>{InputSpec{"mid_clusters_rof", o2::header::gDataOriginMID, "CLUSTERSROF"}, "MIDClusterROF"},
                                                MakeRootTreeWriterSpec::BranchDefinition<o2::dataformats::MCTruthContainer<o2::mid::MCClusterLabel>>{InputSpec{"mid_clusters_labels", o2::header::gDataOriginMID, "CLUSTERSLABELS"}, "MIDClusterLabels", disableMC ? 0 : 1})());

    } else {
      specs.emplace_back(MakeRootTreeWriterSpec("MIDRecoWriter",
                                                "mid-reco.root",
                                                "midreco",
                                                MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::Track>>{InputSpec{"mid_tracks", o2::header::gDataOriginMID, "TRACKS"}, "MIDTrack"},
                                                MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::Cluster3D>>{InputSpec{"mid_trclus", o2::header::gDataOriginMID, "TRACKCLUSTERS"}, "MIDTrackCluster"},
                                                MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ROFRecord>>{InputSpec{"mid_tracks_rof", o2::header::gDataOriginMID, "TRACKSROF"}, "MIDTrackROF"},
                                                MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ROFRecord>>{InputSpec{"mid_trclus_rof", o2::header::gDataOriginMID, "TRCLUSROF"}, "MIDTrackClusterROF"},
                                                MakeRootTreeWriterSpec::BranchDefinition<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>{InputSpec{"mid_track_labels", o2::header::gDataOriginMID, "TRACKLABELS"}, "MIDTrackLabels", disableMC ? 0 : 1},
                                                MakeRootTreeWriterSpec::BranchDefinition<o2::dataformats::MCTruthContainer<o2::mid::MCClusterLabel>>{InputSpec{"mid_trclus_labels", o2::header::gDataOriginMID, "TRCLUSLABELS"}, "MIDTrackClusterLabels", disableMC ? 0 : 1})());
    }
  }

  return specs;
}
