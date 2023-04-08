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

/// @file   TOFMatchableWriterSpec.cxx

#include "TOFWorkflowIO/TOFMatchableWriterSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Headers/DataHeader.h"
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "ReconstructionDataFormats/MatchInfoTOFReco.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using MatchableType = std::vector<o2::dataformats::MatchInfoTOFReco>;

DataProcessorSpec getTOFMatchableWriterSpec(const char* outdef)
{
  // A spectator for logging
  auto logger = [](MatchableType const& indata) {
    LOG(debug) << "RECEIVED MATCHABLE SIZE " << indata.size();
  };
  o2::header::DataDescription ddMatchable0{"MATCHABLES_0"};
  o2::header::DataDescription ddMatchable1{"MATCHABLES_1"};
  o2::header::DataDescription ddMatchable2{"MATCHABLES_2"};
  o2::header::DataDescription ddMatchable3{"MATCHABLES_3"};
  o2::header::DataDescription ddMatchable4{"MATCHABLES_4"};
  o2::header::DataDescription ddMatchable5{"MATCHABLES_5"};
  o2::header::DataDescription ddMatchable6{"MATCHABLES_6"};
  o2::header::DataDescription ddMatchable7{"MATCHABLES_7"};
  o2::header::DataDescription ddMatchable8{"MATCHABLES_8"};
  o2::header::DataDescription ddMatchable9{"MATCHABLES_9"};
  o2::header::DataDescription ddMatchable10{"MATCHABLES_10"};
  o2::header::DataDescription ddMatchable11{"MATCHABLES_11"};
  o2::header::DataDescription ddMatchable12{"MATCHABLES_12"};
  o2::header::DataDescription ddMatchable13{"MATCHABLES_13"};
  o2::header::DataDescription ddMatchable14{"MATCHABLES_14"};
  o2::header::DataDescription ddMatchable15{"MATCHABLES_15"};
  o2::header::DataDescription ddMatchable16{"MATCHABLES_16"};
  o2::header::DataDescription ddMatchable17{"MATCHABLES_17"};
  return MakeRootTreeWriterSpec("TOFMatchableWriter",
                                outdef,
                                "matchableTOF",
                                BranchDefinition<MatchableType>{InputSpec{"input0", o2::header::gDataOriginTOF, ddMatchable0, 0}, "TOFMatchableInfo0", "matchableinfo0-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input1", o2::header::gDataOriginTOF, ddMatchable1, 0}, "TOFMatchableInfo1", "matchableinfo1-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input2", o2::header::gDataOriginTOF, ddMatchable2, 0}, "TOFMatchableInfo2", "matchableinfo2-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input3", o2::header::gDataOriginTOF, ddMatchable3, 0}, "TOFMatchableInfo3", "matchableinfo3-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input4", o2::header::gDataOriginTOF, ddMatchable4, 0}, "TOFMatchableInfo4", "matchableinfo4-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input5", o2::header::gDataOriginTOF, ddMatchable5, 0}, "TOFMatchableInfo5", "matchableinfo5-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input6", o2::header::gDataOriginTOF, ddMatchable6, 0}, "TOFMatchableInfo6", "matchableinfo6-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input7", o2::header::gDataOriginTOF, ddMatchable7, 0}, "TOFMatchableInfo7", "matchableinfo7-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input8", o2::header::gDataOriginTOF, ddMatchable8, 0}, "TOFMatchableInfo8", "matchableinfo8-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input9", o2::header::gDataOriginTOF, ddMatchable9, 0}, "TOFMatchableInfo9", "matchableinfo9-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input10", o2::header::gDataOriginTOF, ddMatchable10, 0}, "TOFMatchableInfo10", "matchableinfo10-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input11", o2::header::gDataOriginTOF, ddMatchable11, 0}, "TOFMatchableInfo11", "matchableinfo11-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input12", o2::header::gDataOriginTOF, ddMatchable12, 0}, "TOFMatchableInfo12", "matchableinfo12-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input13", o2::header::gDataOriginTOF, ddMatchable13, 0}, "TOFMatchableInfo13", "matchableinfo13-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input14", o2::header::gDataOriginTOF, ddMatchable14, 0}, "TOFMatchableInfo14", "matchableinfo14-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input15", o2::header::gDataOriginTOF, ddMatchable15, 0}, "TOFMatchableInfo15", "matchableinfo15-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input16", o2::header::gDataOriginTOF, ddMatchable16, 0}, "TOFMatchableInfo16", "matchableinfo16-branch-name", 1, logger},
                                BranchDefinition<MatchableType>{InputSpec{"input17", o2::header::gDataOriginTOF, ddMatchable17, 0}, "TOFMatchableInfo17", "matchableinfo17-branch-name", 1, logger})();
}

} // namespace tof
} // namespace o2
