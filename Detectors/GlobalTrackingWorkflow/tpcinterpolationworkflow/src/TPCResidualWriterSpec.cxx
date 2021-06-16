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

/// @file  TPCResidualWriterSpec.cxx

#include <vector>

#include "SpacePoints/TrackInterpolation.h"
#include "TPCInterpolationWorkflow/TPCResidualWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{
template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getTPCResidualWriterSpec(bool useMC)
{
  // TODO: not clear if the writer is supposed to write MC labels at some point
  // this is just a dummy definition for the template branch definition below
  // define the correct type and the input specs
  using LabelsType = std::vector<int>;
  // force, this will disable the branch for now, can be adjusted in the future
  useMC = false;

  // A spectator to store the size of the data array for the logger below
  auto tracksSize = std::make_shared<int>();
  auto tracksLogger = [tracksSize](std::vector<TrackData> const& tracks) {
    *tracksSize = tracks.size();
  };
  // A spectator for logging
  auto residualsLogger = [tracksSize](std::vector<TPCClusterResiduals> const& residuals) {
    LOG(INFO) << "ResidualWriterTPC pulled " << *tracksSize << " reference tracks and " << residuals.size() << " TPC cluster residuals";
  };
  return MakeRootTreeWriterSpec("tpc-residuals-writer",
                                "o2residuals_tpc.root",
                                "residualsTPC",
                                BranchDefinition<std::vector<TrackData>>{InputSpec{"tracks", "GLO", "TPCINT_TRK", 0},
                                                                         "tracks",
                                                                         "tracks-branch-name",
                                                                         1,
                                                                         tracksLogger},
                                BranchDefinition<std::vector<TPCClusterResiduals>>{InputSpec{"residuals", "GLO", "TPCINT_RES", 0},
                                                                                   "residuals",
                                                                                   "residuals-branch-name",
                                                                                   1,
                                                                                   residualsLogger},
                                // NOTE: this branch template is to show how the conditional MC labels can
                                // be defined, the '0' disables the branch for the moment
                                BranchDefinition<LabelsType>{InputSpec{"matchtpclabels", "GLO", "SOME_LABELS", 0},
                                                             "labels",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             "labels-branch-name"})();
}

} // namespace tpc
} // namespace o2
