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

#include "ITS3Workflow/TrackWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "ReconstructionDataFormats/Vertex.h"

using namespace o2::framework;

namespace o2
{
namespace its3
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
    LOG(info) << "ITS3TrackWriter pulled " << *tracksSize << " tracks, in " << rofs.size() << " RO frames";
  };
  return MakeRootTreeWriterSpec("its3-track-writer",
                                "o2trac_its3.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with ITS3 tracks"},
                                BranchDefinition<std::vector<o2::its::TrackITS>>{InputSpec{"tracks", "IT3", "TRACKS", 0},
                                                                                 "ITS3Track",
                                                                                 tracksSizeGetter},
                                BranchDefinition<std::vector<int>>{InputSpec{"trackClIdx", "IT3", "TRACKCLSID", 0},
                                                                   "ITS3TrackClusIdx"},
                                BranchDefinition<std::vector<Vertex>>{InputSpec{"vertices", "IT3", "VERTICES", 0},
                                                                      "Vertices"},
                                BranchDefinition<std::vector<o2::itsmft::ROFRecord>>{InputSpec{"vtxROF", "IT3", "VERTICESROF", 0},
                                                                                     "VerticesROF"},
                                BranchDefinition<std::vector<o2::itsmft::ROFRecord>>{InputSpec{"ROframes", "IT3", "ITS3TrackROF", 0},
                                                                                     "ITS3TracksROF",
                                                                                     logger},
                                BranchDefinition<LabelsType>{InputSpec{"labels", "IT3", "TRACKSMCTR", 0},
                                                             "ITS3TrackMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""},
                                BranchDefinition<LabelsType>{InputSpec{"labelsVertices", "IT3", "VERTICESMCTR", 0},
                                                             "ITS3VertexMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""},
                                BranchDefinition<ROFRecLblT>{InputSpec{"MC2ROframes", "IT3", "ITS3TrackMC2ROF", 0},
                                                             "ITS3TracksMC2ROF",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""})();
}

} // namespace its3
} // namespace o2
