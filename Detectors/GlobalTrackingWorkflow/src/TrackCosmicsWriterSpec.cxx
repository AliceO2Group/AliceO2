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

/// @file   TrackCosmicsWriterSpec.cxx

#include <vector>
#include "GlobalTrackingWorkflow/TrackCosmicsWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "ReconstructionDataFormats/TrackCosmics.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsGlobalTracking/CosmicTrack.h"
#include "CommonDataFormat/TFIDInfo.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using TracksType = std::vector<o2::dataformats::TrackCosmics>;
using LabelsType = std::vector<o2::MCCompLabel>;

DataProcessorSpec getTrackCosmicsWriterSpec(bool useMC)
{
  // A spectator for logging
  auto logger = [](TracksType const& tracks) {
    LOG(info) << "Writing " << tracks.size() << " Cosmics Tracks";
  };
  return MakeRootTreeWriterSpec("cosmic-track-writer",
                                "cosmics.root",
                                "cosmics",
                                -1,  // do not limit number of events to store
                                100, // periodically autosave
                                BranchDefinition<TracksType>{InputSpec{"tracks", "GLO", "COSMICTRC", 0},
                                                             "tracks",
                                                             "tracks-branch-name",
                                                             1,
                                                             logger},
                                BranchDefinition<LabelsType>{InputSpec{"tracksMC", "GLO", "COSMICTRC_MC", 0},
                                                             "MCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""})();
}

DataProcessorSpec getCosmicsFullWriterSpec()
{
  auto logger = [](std::vector<o2::dataformats::CosmicTrack> const& cosmics) {
    LOG(info) << "Writing " << cosmics.size() << " cosmics with clusters";
  };
  return MakeRootTreeWriterSpec("cosmics-full-writer",
                                "o2_cosmics_full.root",
                                "cosmicsFull",
                                -1,  // do not limit number of events to store
                                100, // periodically autosave
                                BranchDefinition<std::vector<o2::dataformats::CosmicTrack>>{InputSpec{"cosmics", "GLO", "COSMFULL", 0}, "cosmics", 1, logger},
                                BranchDefinition<o2::dataformats::CosmicsTFInfo>{InputSpec{"tfinfo", "GLO", "COSMFULLTF", 0}, "tfInfo", 1},
                                BranchDefinition<o2::dataformats::TFIDInfo>{InputSpec{"tfid", "GLO", "COSMFULLTFID", 0}, "tfID", 1})();
}

} // namespace globaltracking
} // namespace o2
