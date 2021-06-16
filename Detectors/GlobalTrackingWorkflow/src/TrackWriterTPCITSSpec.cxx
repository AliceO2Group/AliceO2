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

/// @file   TrackWriterTPCITSSpec.cxx

#include <vector>
#include "GlobalTrackingWorkflow/TrackWriterTPCITSSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using TracksType = std::vector<o2::dataformats::TrackTPCITS>;
using LabelsType = std::vector<o2::MCCompLabel>;

DataProcessorSpec getTrackWriterTPCITSSpec(bool useMC)
{
  // A spectator for logging
  auto logger = [](TracksType const& tracks) {
    LOG(INFO) << "Writing " << tracks.size() << " TPC-ITS matches";
  };
  return MakeRootTreeWriterSpec("itstpc-track-writer",
                                "o2match_itstpc.root",
                                "matchTPCITS",
                                BranchDefinition<TracksType>{InputSpec{"match", "GLO", "TPCITS", 0},
                                                             "TPCITS",
                                                             "TPCITS-branch-name",
                                                             1,
                                                             logger},
                                BranchDefinition<LabelsType>{InputSpec{"matchMC", "GLO", "TPCITS_MC", 0},
                                                             "MatchMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""})();
}

} // namespace globaltracking
} // namespace o2
