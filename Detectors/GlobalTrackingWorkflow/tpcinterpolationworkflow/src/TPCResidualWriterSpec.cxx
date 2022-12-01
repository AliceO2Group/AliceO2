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
#include "SpacePoints/TrackResiduals.h"
#include "SpacePoints/SpacePointsCalibConfParam.h"
#include "TPCInterpolationWorkflow/TPCResidualWriterSpec.h"
#include "DataFormatsCTP/LumiInfo.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{
template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getTPCResidualWriterSpec(bool writeTrackData)
{
  bool writeUnfiltered = SpacePointsCalibConfParam::Instance().writeUnfiltered;
  return MakeRootTreeWriterSpec("tpc-residuals-writer",
                                "o2residuals_tpc.root",
                                "residualsTPC",
                                BranchDefinition<std::vector<TrackData>>{InputSpec{"tracksUnfiltered", "GLO", "TPCINT_TRK", 0}, "tracksUnfiltered", ((writeUnfiltered && writeTrackData) ? 1 : 0)},
                                BranchDefinition<std::vector<TPCClusterResiduals>>{InputSpec{"residualsUnfiltered", "GLO", "TPCINT_RES", 0}, "residualsUnfiltered", (writeUnfiltered ? 1 : 0)},
                                BranchDefinition<std::vector<UnbinnedResid>>{InputSpec{"residuals", "GLO", "UNBINNEDRES"}, "residuals"},
                                BranchDefinition<std::vector<TrackDataCompact>>{InputSpec{"trackRefs", "GLO", "TRKREFS"}, "trackRefs"},
                                BranchDefinition<std::vector<TrackData>>{InputSpec{"tracks", "GLO", "TRKDATA"}, "tracks", (writeTrackData ? 1 : 0)})();
}

} // namespace tpc
} // namespace o2
