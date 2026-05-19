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

/// @file   TrackWriterSpec.cxx

#include <vector>

#include "ALICE3GlobalReconstructionWorkflow/TrackWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"

using namespace o2::framework;

namespace o2
{
namespace trk
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using LabelsType = std::vector<o2::MCCompLabel>;
using namespace o2::header;

DataProcessorSpec getTrackWriterSpec(bool useMC)
{
  // Spectators for logging
  auto tracksSize = std::make_shared<int>(0);
  auto tracksSizeGetter = [tracksSize](std::vector<o2::its::TrackITS> const& tracks) {
    *tracksSize = tracks.size();
  };
  auto logger = [tracksSize]() {
    LOG(info) << "TRKTrackWriter pulled " << *tracksSize << " tracks";
  };

  return MakeRootTreeWriterSpec("trk-track-writer",
                                "o2trac_trk.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with TRK tracks"},
                                BranchDefinition<std::vector<o2::its::TrackITS>>{InputSpec{"tracks", "TRK", "TRACKS", 0},
                                                                                 "TRKTrack",
                                                                                 tracksSizeGetter},
                                BranchDefinition<LabelsType>{InputSpec{"labels", "TRK", "TRACKSMCTR", 0},
                                                             "TRKTrackMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""})();
}

} // namespace trk
} // namespace o2
