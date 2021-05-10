// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  TRDTrackWriterSpec.cxx

#include <vector>
#include "DataFormatsTRD/TrackTRD.h"

#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "TRDWorkflow/TRDTrackWriterSpec.h"

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
  // TODO: not clear if the writer is supposed to write MC labels at some point
  // this is just a dummy definition for the template branch definition below
  // define the correct type and the input specs
  using LabelsType = std::vector<int>;
  // force, this will disable the branch for now, can be adjusted in the future
  useMC = false;

  // A spectator to store the size of the data array for the logger below
  auto tracksSize = std::make_shared<int>();
  auto tracksLogger = [tracksSize](std::vector<o2::trd::TrackTRD> const& tracks) {
    *tracksSize = tracks.size();
  };

  return MakeRootTreeWriterSpec("trd-track-writer-tpcits",
                                "trdmatches_itstpc.root",
                                "tracksTRD",
                                BranchDefinition<std::vector<o2::trd::TrackTRD>>{InputSpec{"tracks", o2::header::gDataOriginTRD, "MATCHTRD_GLO", 0},
                                                                                 "tracks",
                                                                                 "tracks-branch-name",
                                                                                 1,
                                                                                 tracksLogger},
                                // NOTE: this branch template is to show how the conditional MC labels can
                                // be defined, the '0' disables the branch for the moment
                                BranchDefinition<LabelsType>{InputSpec{"matchtpclabels", "GLO", "SOME_LABELS", 0},
                                                             "labels",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             "labels-branch-name"})();
}

DataProcessorSpec getTRDTPCTrackWriterSpec(bool useMC)
{
  // TODO: not clear if the writer is supposed to write MC labels at some point
  // this is just a dummy definition for the template branch definition below
  // define the correct type and the input specs
  using LabelsType = std::vector<int>;
  // force, this will disable the branch for now, can be adjusted in the future
  useMC = false;

  // A spectator to store the size of the data array for the logger below
  auto tracksSize = std::make_shared<int>();
  auto tracksLogger = [tracksSize](std::vector<o2::trd::TrackTRD> const& tracks) {
    *tracksSize = tracks.size();
  };

  return MakeRootTreeWriterSpec("trd-track-writer-tpc",
                                "trdmatches_tpc.root",
                                "tracksTRD",
                                BranchDefinition<std::vector<o2::trd::TrackTRD>>{InputSpec{"tracks", o2::header::gDataOriginTRD, "MATCHTRD_TPC", 0},
                                                                                 "tracks",
                                                                                 "tracks-branch-name",
                                                                                 1,
                                                                                 tracksLogger},
                                // NOTE: this branch template is to show how the conditional MC labels can
                                // be defined, the '0' disables the branch for the moment
                                BranchDefinition<LabelsType>{InputSpec{"matchtpclabels", "GLO", "SOME_LABELS", 0},
                                                             "labels",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             "labels-branch-name"})();
}

} // namespace trd
} // namespace o2
