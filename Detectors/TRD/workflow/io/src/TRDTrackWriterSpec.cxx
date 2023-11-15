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

/// @file  TRDTrackWriterSpec.cxx

#include <vector>
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/TrackTriggerRecord.h"
#include "ReconstructionDataFormats/MatchingType.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "TRDWorkflowIO/TRDTrackWriterSpec.h"
#include "SimulationDataFormat/MCCompLabel.h"

using namespace o2::framework;
using namespace o2::gpu;

using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace trd
{
template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getTRDGlobalTrackWriterSpec(bool useMC)
{
  using LabelsType = std::vector<o2::MCCompLabel>;

  // A spectator to store the size of the data array for the logger below
  auto tracksSize = std::make_shared<int>();
  auto tracksLogger = [tracksSize](std::vector<o2::trd::TrackTRD> const& tracks) {
    *tracksSize = tracks.size();
  };

  return MakeRootTreeWriterSpec("trd-track-writer-tpcits",
                                "trdmatches_itstpc.root",
                                "tracksTRD",
                                BranchDefinition<std::vector<o2::trd::TrackTRD>>{InputSpec{"tracks", o2::header::gDataOriginTRD, "MATCH_ITSTPC", 0},
                                                                                 "tracks",
                                                                                 "tracks-branch-name",
                                                                                 1,
                                                                                 tracksLogger},
                                BranchDefinition<std::vector<o2::trd::TrackTriggerRecord>>{InputSpec{"trackTrig", o2::header::gDataOriginTRD, "TRGREC_ITSTPC", 0},
                                                                                           "trgrec",
                                                                                           "trgrec-branch-name",
                                                                                           1},
                                BranchDefinition<LabelsType>{InputSpec{"trdlabels", o2::header::gDataOriginTRD, "MCLB_ITSTPC_TRD", 0},
                                                             "labelsTRD",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             "trdlabels-branch-name"},
                                BranchDefinition<LabelsType>{InputSpec{"matchitstpclabels", o2::header::gDataOriginTRD, "MCLB_ITSTPC", 0},
                                                             "labels",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             "labels-branch-name"})();
}

DataProcessorSpec getTRDTPCTrackWriterSpec(bool useMC, bool strictMode)
{
  using LabelsType = std::vector<o2::MCCompLabel>;

  // A spectator to store the size of the data array for the logger below
  auto tracksSize = std::make_shared<int>();
  auto tracksLogger = [tracksSize](std::vector<o2::trd::TrackTRD> const& tracks) {
    *tracksSize = tracks.size();
  };
  uint32_t ss = o2::globaltracking::getSubSpec(strictMode ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);
  return MakeRootTreeWriterSpec("trd-track-writer-tpc",
                                "trdmatches_tpc.root",
                                "tracksTRD",
                                BranchDefinition<std::vector<o2::trd::TrackTRD>>{InputSpec{"tracks", o2::header::gDataOriginTRD, "MATCH_TPC", ss},
                                                                                 "tracks",
                                                                                 "tracks-branch-name",
                                                                                 1,
                                                                                 tracksLogger},
                                BranchDefinition<std::vector<o2::trd::TrackTriggerRecord>>{InputSpec{"trackTrig", o2::header::gDataOriginTRD, "TRGREC_TPC", ss},
                                                                                           "trgrec",
                                                                                           "trgrec-branch-name",
                                                                                           1},
                                BranchDefinition<LabelsType>{InputSpec{"trdlabels", o2::header::gDataOriginTRD, "MCLB_TPC_TRD", ss},
                                                             "labelsTRD",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             "trdlabels-branch-name"},
                                BranchDefinition<LabelsType>{InputSpec{"matchtpclabels", o2::header::gDataOriginTRD, "MCLB_TPC", ss},
                                                             "labels",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             "labels-branch-name"})();
}

} // namespace trd
} // namespace o2
