// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/RecoWorkflowMC.cxx
/// \brief  Definition of MID reconstruction workflow for MC
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   27 September 2019

#include "MIDWorkflow/RecoWorkflowMC.h"

#include "DPLUtils/Utils.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDWorkflow/ClusterizerMCSpec.h"
#include "MIDWorkflow/DigitReaderSpec.h"
#include "MIDWorkflow/TrackerMCSpec.h"
#include "MIDSimulation/MCClusterLabel.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

of::WorkflowSpec getRecoWorkflowMC()
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
  specs.emplace_back(getClusterizerMCSpec());
  specs.emplace_back(getTrackerMCSpec());
  specs.emplace_back(of::MakeRootTreeWriterSpec("MIDTrackLabelsWriter",
                                                "mid-track-labels.root",
                                                "midtracklabels",
                                                of::MakeRootTreeWriterSpec::TerminationPolicy::Workflow,
                                                of::MakeRootTreeWriterSpec::TerminationCondition{checkReady},
                                                of::MakeRootTreeWriterSpec::BranchDefinition<const char*>{of::InputSpec{"mid_tracks", "MID", "TRACKS"}, "MIDTrack"},
                                                of::MakeRootTreeWriterSpec::BranchDefinition<const char*>{of::InputSpec{"mid_trackClusters", "MID", "TRACKCLUSTERS"}, "MIDTrackClusters"},
                                                of::MakeRootTreeWriterSpec::BranchDefinition<std::vector<ROFRecord>>{of::InputSpec{"mid_tracks_rof", "MID", "TRACKSROF"}, "MIDTrackROF"},
                                                of::MakeRootTreeWriterSpec::BranchDefinition<std::vector<ROFRecord>>{of::InputSpec{"mid_trclus_rof", "MID", "TRCLUSROF"}, "MIDTrackClusterROF"},
                                                of::MakeRootTreeWriterSpec::BranchDefinition<dataformats::MCTruthContainer<MCCompLabel>>{of::InputSpec{"mid_track_labels", "MID", "TRACKSLABELS"}, "MIDTrackLabels"},
                                                of::MakeRootTreeWriterSpec::BranchDefinition<dataformats::MCTruthContainer<MCClusterLabel>>{of::InputSpec{"mid_trclus_labels", "MID", "TRCLUSLABELS"}, "MIDTrackClusterLabels"})());

  return specs;
}
} // namespace mid
} // namespace o2
