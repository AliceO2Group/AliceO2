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

/// @file   MatchedMFTMCHWriterSpec.cxx

#include <vector>
#include "GlobalTrackingWorkflow/MatchedMFTMCHWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "ReconstructionDataFormats/MatchInfoFwd.h"
#include "SimulationDataFormat/MCCompLabel.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using MatchesType = std::vector<o2::dataformats::MatchInfoFwd>;
using LabelsType = std::vector<o2::MCCompLabel>;

DataProcessorSpec getMFTMCHMatchesWriterSpec(bool useMC)
{
  // A spectator for logging
  auto logger = [](MatchesType const& tracks) {
    LOG(info) << "Writing " << tracks.size() << " MFTMCH Matches";
  };
  return MakeRootTreeWriterSpec("mftmch-matches-writer",
                                "mftmchmatches.root",
                                "o2sim",
                                BranchDefinition<MatchesType>{InputSpec{"MFTMCHMatches", "GLO", "MTC_MFTMCH", 0}, "matches", logger})();
}

} // namespace globaltracking
} // namespace o2
