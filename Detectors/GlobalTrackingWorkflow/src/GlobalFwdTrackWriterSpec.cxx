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

/// @file   GlobalFwdTrackWriterSpec.cxx

#include <vector>
#include "GlobalTrackingWorkflow/GlobalFwdTrackWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "ReconstructionDataFormats/GlobalFwdTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "GlobalTracking/MatchGlobalFwdParam.h"
#include "DataFormatsMFT/TrackMFT.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using TracksType = std::vector<o2::dataformats::GlobalFwdTrack>;
using FwdTracksType = std::vector<o2::track::TrackParCovFwd>;
using MFTTracksType = std::vector<o2::mft::TrackMFT>;
using MatchesType = std::vector<o2::dataformats::MatchInfoFwd>;
using LabelsType = std::vector<o2::MCCompLabel>;

DataProcessorSpec getGlobalFwdTrackWriterSpec(bool useMC)
{
  const auto& matchingParam = GlobalFwdMatchingParam::Instance();
  if (matchingParam.saveMode != kSaveTrainingData) {
    // A spectator for logging
    auto logger = [](TracksType const& tracks) {
      LOG(info) << "Writing " << tracks.size() << " GlobalForward Tracks";
    };
    return MakeRootTreeWriterSpec("globalfwd-track-writer",
                                  "globalfwdtracks.root",
                                  "GlobalFwdTracks",
                                  BranchDefinition<TracksType>{InputSpec{"fwdtracks", "GLO", "GLFWD", 0}, "fwdtracks", logger},
                                  BranchDefinition<LabelsType>{InputSpec{"labels", "GLO", "GLFWD_MC", 0}, "MCTruth", (useMC ? 1 : 0), ""})();
  } else {
    // A spectator for logging
    auto logger = [](MFTTracksType const& tracks) {
      LOG(info) << "Writing " << tracks.size() << " MFT and MCH track parameters at matching plane";
    };
    return MakeRootTreeWriterSpec("globalfwd-track-writer",
                                  "globalfwdparamsatmatchingplane.root",
                                  "MatchingPlaneParams",
                                  BranchDefinition<MFTTracksType>{InputSpec{"mfttracks", "GLO", "GLFWDMFT", 0}, "mfttracks", logger},
                                  BranchDefinition<FwdTracksType>{InputSpec{"mchtracks", "GLO", "GLFWDMCH", 0}, "mchtracks"},
                                  BranchDefinition<MatchesType>{InputSpec{"matchinfo", "GLO", "GLFWDINF", 0}, "matchinfo"},
                                  BranchDefinition<LabelsType>{InputSpec{"labels", "GLO", "GLFWD_MC", 0}, "MCTruth", (useMC ? 1 : 0), ""})();
  }
}

} // namespace globaltracking
} // namespace o2
