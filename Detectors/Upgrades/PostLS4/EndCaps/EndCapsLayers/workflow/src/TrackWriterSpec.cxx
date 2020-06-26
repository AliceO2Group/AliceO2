// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackWriterSpec.cxx

#include <vector>

#include "EC0Workflow/TrackWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "ReconstructionDataFormats/Vertex.h"

using namespace o2::framework;

namespace o2
{
namespace ecl
{
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using LabelsType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using ROFRecLblT = std::vector<o2::itsmft::MC2ROFRecord>;
using namespace o2::header;

DataProcessorSpec getTrackWriterSpec(bool useMC)
{
  // Spectators for logging
  // this is only to restore the original behavior
  auto tracksSize = std::make_shared<int>(0);
  auto tracksSizeGetter = [tracksSize](std::vector<o2::its::TrackITS> const& tracks) {
    *tracksSize = tracks.size();
  };
  auto logger = [tracksSize](std::vector<o2::itsmft::ROFRecord> const& rofs) {
    LOG(INFO) << "EC0TrackWriter pulled " << *tracksSize << " tracks, in " << rofs.size() << " RO frames";
  };
  return MakeRootTreeWriterSpec("ecl-track-writer",
                                "o2trac_ecl.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with EC0 tracks"},
                                BranchDefinition<std::vector<o2::its::TrackITS>>{InputSpec{"tracks", "EC0", "TRACKS", 0},
                                                                                 "EC0Track",
                                                                                 tracksSizeGetter},
                                BranchDefinition<std::vector<int>>{InputSpec{"trackClIdx", "EC0", "TRACKCLSID", 0},
                                                                   "EC0TrackClusIdx"},
                                BranchDefinition<std::vector<Vertex>>{InputSpec{"vertices", "EC0", "VERTICES", 0},
                                                                      "Vertices"},
                                BranchDefinition<std::vector<o2::itsmft::ROFRecord>>{InputSpec{"vtxROF", "EC0", "VERTICESROF", 0},
                                                                                     "VerticesROF"},
                                BranchDefinition<std::vector<o2::itsmft::ROFRecord>>{InputSpec{"ROframes", "EC0", "EC0TrackROF", 0},
                                                                                     "EC0TracksROF",
                                                                                     logger},
                                BranchDefinition<LabelsType>{InputSpec{"labels", "EC0", "TRACKSMCTR", 0},
                                                             "EC0TrackMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""},
                                BranchDefinition<ROFRecLblT>{InputSpec{"MC2ROframes", "EC0", "EC0TrackMC2ROF", 0},
                                                             "EC0TracksMC2ROF",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""})();
}

} // namespace ecl
} // namespace o2
