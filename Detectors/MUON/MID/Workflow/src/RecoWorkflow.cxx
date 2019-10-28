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
#include "DataFormatsMID/ROFRecord.h"
#include "MIDWorkflow/ClusterizerSpec.h"
#include "MIDWorkflow/RawReaderSpec.h"
#include "MIDWorkflow/TrackerSpec.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

of::WorkflowSpec getRecoWorkflow()
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

  specs.emplace_back(getRawReaderSpec());
  specs.emplace_back(getClusterizerSpec());
  specs.emplace_back(getTrackerSpec());
  specs.emplace_back(of::MakeRootTreeWriterSpec("MIDTracksWriter",
                                                "mid-tracks.root",
                                                "midtracks",
                                                of::MakeRootTreeWriterSpec::TerminationPolicy::Process,
                                                of::MakeRootTreeWriterSpec::TerminationCondition{checkReady},
                                                of::MakeRootTreeWriterSpec::BranchDefinition<const char*>{of::InputSpec{"mid_tracks", "MID", "TRACKS"}, "MIDTrack"},
                                                of::MakeRootTreeWriterSpec::BranchDefinition<const char*>{of::InputSpec{"mid_trackClusters", "MID", "TRACKCLUSTERS"}, "MIDTrackClusters"},
                                                of::MakeRootTreeWriterSpec::BranchDefinition<std::vector<ROFRecord>>{of::InputSpec{"mid_tracks_rof", "MID", "TRACKSROF"}, "MIDTrackROF"},
                                                of::MakeRootTreeWriterSpec::BranchDefinition<std::vector<ROFRecord>>{of::InputSpec{"mid_trclus_rof", "MID", "TRCLUSROF"}, "MIDTrackClusterROF"})());

  return specs;
}
} // namespace mid
} // namespace o2
