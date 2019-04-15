// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/RecoWorkflow.cxx
/// \brief  Definition of MID reconstruction workflow
/// \author Gabriele G. Fronze <gfronze at cern.ch>
/// \date   11 July 2018

#include "MIDWorkflow/RecoWorkflow.h"
#include "DPLUtils/Utils.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "MIDWorkflow/ClusterizerSpec.h"
#include "MIDWorkflow/ClusterLabelerSpec.h"
#include "MIDWorkflow/DigitReaderSpec.h"
#include "MIDWorkflow/TrackerSpec.h"
#include "MIDWorkflow/TrackLabelerSpec.h"
#include "MIDSimulation/MCClusterLabel.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

of::WorkflowSpec getRecoWorkflow(bool useMC)
{

  auto checkReady = [](o2::framework::DataRef const& ref) {
    // The default checkReady function is not defined in MakeRootTreeWriterSpec:
    // this leads to a std::exception in DPL.
    // While the exception seems harmless (the processing goes on without apparent consequence),
    // it is quite ugly to see.
    // So, let us define checkReady here.
    // FIXME: to be fixed in MakeRootTreeWriterSpec
    return false;
  };

  of::WorkflowSpec specs;

  specs.emplace_back(getDigitReaderSpec());
  specs.emplace_back(getClusterizerSpec(useMC));
  if (useMC) {
    specs.emplace_back(getClusterLabelerSpec());
    specs.emplace_back(o2::workflows::defineBroadcaster("ClustersBcast", of::InputSpec{ "mid_clusters_data", "MID", "CLUSTERS" }, of::Outputs{ of::OutputSpec{ "MID", "CLUSTERS_DATA" }, of::OutputSpec{ "MID", "CLUSTERS_MC" } }));
  }
  specs.emplace_back(getTrackerSpec(useMC));
  if (useMC) {
    specs.emplace_back(o2::workflows::defineBroadcaster("TracksBcast", of::InputSpec{ "mid_tracks_data", "MID", "TRACKS" }, of::Outputs{ of::OutputSpec{ "MID", "TRACKS_DATA" }, of::OutputSpec{ "MID", "TRACKS_MC" } }));
    specs.emplace_back(o2::workflows::defineBroadcaster("TrackClustersBcast", of::InputSpec{ "mid_trackClusters_data", "MID", "TRACKCLUSTERS" }, of::Outputs{ of::OutputSpec{ "MID", "TRCLUS_DATA" }, of::OutputSpec{ "MID", "TRCLUS_MC" } }));
    specs.emplace_back(getTrackLabelerSpec());
    specs.emplace_back(of::MakeRootTreeWriterSpec("MIDTrackLabelsWriter",
                                                  "mid-track-labels.root",
                                                  "midtracklabels",
                                                  of::MakeRootTreeWriterSpec::TerminationPolicy::Workflow,
                                                  of::MakeRootTreeWriterSpec::TerminationCondition{ checkReady },
                                                  of::MakeRootTreeWriterSpec::BranchDefinition<const char*>{ of::InputSpec{ "mid_tracks", "MID", "TRACKS_DATA" }, "MIDTrack" },
                                                  of::MakeRootTreeWriterSpec::BranchDefinition<const char*>{ of::InputSpec{ "mid_trackClusters", "MID", "TRCLUS_DATA" }, "MIDTrackClusters" },
                                                  of::MakeRootTreeWriterSpec::BranchDefinition<dataformats::MCTruthContainer<MCCompLabel>>{ of::InputSpec{ "mid_track_labels", "MID", "TRACKSLABELS" }, "MIDTrackLabels" },
                                                  of::MakeRootTreeWriterSpec::BranchDefinition<dataformats::MCTruthContainer<MCClusterLabel>>{ of::InputSpec{ "mid_trclus_labels", "MID", "TRCLUSLABELS" }, "MIDTrackClusterLabels" })());
  }

  return specs;
}
} // namespace mid
} // namespace o2
