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

#include "ITSMFTCAWriter/ITSCATrackWriterSpec.h"

#include <vector>

#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2::its::ca
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using LabelsType = std::vector<o2::MCCompLabel>;

DataProcessorSpec getTrackWriterSpec(bool useMC)
{
  // Spectators for logging; mirrors ITSWorkflow/TrackWriterSpec.cxx.
  auto tracksSize = std::make_shared<size_t>(0);
  auto tracksSizeGetter = [tracksSize](std::vector<o2::its::TrackITS> const& tracks) {
    *tracksSize = tracks.size();
  };
  auto logger = [tracksSize](std::vector<o2::itsmft::ROFRecord> const& rofs) {
    LOG(info) << "ITSCATrackWriter pulled " << *tracksSize << " tracks, in " << rofs.size() << " RO frames";
  };
  // Deliberately no VERTICES/VERTICESROF/VERTICESMCTR/VERTICESMCPUR branch:
  // this opt-in tracker-only workflow never publishes those OutputSpecs (see
  // CATrackerSpec.cxx), so a writer branch consuming them would simply never
  // fire.
  return MakeRootTreeWriterSpec("its-ca-track-writer",
                                "o2trac_its_ca.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with ITS common-CA tracks"},
                                BranchDefinition<std::vector<o2::its::TrackITS>>{InputSpec{"tracks", "ITS", "TRACKS", 0},
                                                                                 "ITSTrack",
                                                                                 tracksSizeGetter},
                                BranchDefinition<std::vector<int>>{InputSpec{"trackClIdx", "ITS", "TRACKCLSID", 0},
                                                                   "ITSTrackClusIdx"},
                                BranchDefinition<std::vector<o2::itsmft::ROFRecord>>{InputSpec{"ROframes", "ITS", "ITSTrackROF", 0},
                                                                                     "ITSTracksROF",
                                                                                     logger},
                                BranchDefinition<LabelsType>{InputSpec{"labels", "ITS", "TRACKSMCTR", 0},
                                                             "ITSTrackMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""})();
}

} // namespace o2::its::ca
