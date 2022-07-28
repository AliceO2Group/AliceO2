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

/// @file   TrackWriterSpec.cxx

#include <vector>

#include "ITSWorkflow/TrackWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "ReconstructionDataFormats/Vertex.h"

using namespace o2::framework;

namespace o2
{
namespace its
{
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using LabelsType = std::vector<o2::MCCompLabel>;
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
    LOG(info) << "ITSTrackWriter pulled " << *tracksSize << " tracks, in " << rofs.size() << " RO frames";
  };
  return MakeRootTreeWriterSpec("its-track-writer",
                                "o2trac_its.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with ITS tracks"},
                                BranchDefinition<std::vector<o2::its::TrackITS>>{InputSpec{"tracks", "ITS", "TRACKS", 0},
                                                                                 "ITSTrack",
                                                                                 tracksSizeGetter},
                                BranchDefinition<std::vector<int>>{InputSpec{"trackClIdx", "ITS", "TRACKCLSID", 0},
                                                                   "ITSTrackClusIdx"},
                                BranchDefinition<std::vector<Vertex>>{InputSpec{"vertices", "ITS", "VERTICES", 0},
                                                                      "Vertices"},
                                BranchDefinition<std::vector<o2::itsmft::ROFRecord>>{InputSpec{"vtxROF", "ITS", "VERTICESROF", 0},
                                                                                     "VerticesROF"},
                                BranchDefinition<std::vector<o2::itsmft::ROFRecord>>{InputSpec{"ROframes", "ITS", "ITSTrackROF", 0},
                                                                                     "ITSTracksROF",
                                                                                     logger},
                                BranchDefinition<LabelsType>{InputSpec{"labels", "ITS", "TRACKSMCTR", 0},
                                                             "ITSTrackMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""},
                                BranchDefinition<LabelsType>{InputSpec{"labelsVertices", "ITS", "VERTICESMCTR", 0},
                                                             "ITSVertexMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""},
                                BranchDefinition<ROFRecLblT>{InputSpec{"MC2ROframes", "ITS", "ITSTrackMC2ROF", 0},
                                                             "ITSTracksMC2ROF",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""})();
}

} // namespace its
} // namespace o2
