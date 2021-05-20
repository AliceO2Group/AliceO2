// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackWriterSpec.cxx

#include <vector>

#include "MFTWorkflow/TrackWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "MFTTracking/TrackCA.h"

#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;
using LabelsType = std::vector<o2::MCCompLabel>;
using ROFRecLblT = std::vector<o2::itsmft::MC2ROFRecord>;

namespace o2
{
namespace mft
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using namespace o2::header;

DataProcessorSpec getTrackWriterSpec(bool useMC)
{
  // Spectators for logging
  // this is only to restore the original behavior
  auto tracksSize = std::make_shared<int>(0);
  auto tracksSizeGetter = [tracksSize](std::vector<o2::mft::TrackMFT> const& tracks) {
    *tracksSize = tracks.size();
  };
  auto logger = [tracksSize](std::vector<o2::itsmft::ROFRecord> const& rofs) {
    LOG(INFO) << "MFTTrackWriter pulled " << *tracksSize << " tracks, in " << rofs.size() << " RO frames";
  };
  return MakeRootTreeWriterSpec("mft-track-writer",
                                "mfttracks.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with MFT tracks"},
                                BranchDefinition<std::vector<o2::mft::TrackMFT>>{InputSpec{"tracks", "MFT", "TRACKS", 0},
                                                                                 "MFTTrack",
                                                                                 tracksSizeGetter},
                                BranchDefinition<std::vector<int>>{InputSpec{"trackClIdx", "MFT", "TRACKCLSID", 0},
                                                                   "MFTTrackClusIdx"},
                                BranchDefinition<LabelsType>{InputSpec{"labels", "MFT", "TRACKSMCTR", 0},
                                                             "MFTTrackMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""},
                                BranchDefinition<std::vector<o2::itsmft::ROFRecord>>{InputSpec{"ROframes", "MFT", "MFTTrackROF", 0},
                                                                                     "MFTTracksROF",
                                                                                     logger},
                                BranchDefinition<ROFRecLblT>{InputSpec{"MC2ROframes", "MFT", "TRACKSMC2ROF", 0},
                                                             "MFTTracksMC2ROF",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""})();
}

} // namespace mft
} // namespace o2
