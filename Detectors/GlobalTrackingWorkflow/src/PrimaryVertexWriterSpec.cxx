// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   PrimaryVertexWriterSpec.cxx

#include <vector>

#include "GlobalTrackingWorkflow/PrimaryVertexWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "SimulationDataFormat/MCEventLabel.h"

using namespace o2::framework;

namespace o2
{
namespace vertexing
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

using TimeEst = o2::dataformats::TimeStampWithError<float, float>;
using Vertex = o2::dataformats::Vertex<TimeEst>;
using V2TRef = o2::dataformats::RangeReference<int, int>;
using Label = o2::MCEventLabel;

using namespace o2::header;

DataProcessorSpec getPrimaryVertexWriterSpec(bool useMC)
{
  auto logger = [](std::vector<Vertex> const& v) {
    LOG(INFO) << "PrimaryVertexWriter pulled " << v.size() << " vertices";
  };

  return MakeRootTreeWriterSpec("primary-vertex-writer",
                                "o2_primary_vertex.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with Primary Vertices"},
                                BranchDefinition<std::vector<Vertex>>{InputSpec{"vertices", "GLO", "PVERTEX", 0}, "PrimaryVertex", logger},
                                BranchDefinition<std::vector<V2TRef>>{InputSpec{"v2tref", "GLO", "PVERTEX_TRIDREFS", 0}, "PV2TrackRefs"},
                                BranchDefinition<std::vector<int>>{InputSpec{"vttrackID", "GLO", "PVERTEX_TRID", 0}, "PVTrackIndices"},
                                BranchDefinition<std::vector<Label>>{InputSpec{"labels", "GLO", "PVERTEX_MCTR", 0}, "PVMCTruth", (useMC ? 1 : 0), ""})();
}

} // namespace vertexing
} // namespace o2
