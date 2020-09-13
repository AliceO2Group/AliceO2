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
#include "DetectorsVertexing/PVertexerHelpers.h"
#include "CommonDataFormat/RangeReference.h"
#include "SimulationDataFormat/MCEventLabel.h"

using namespace o2::framework;

namespace o2
{
namespace vertexing
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

using Label = o2::MCEventLabel;

using namespace o2::header;

DataProcessorSpec getPrimaryVertexWriterSpec(bool disableMatching, bool useMC)
{
  auto logger = [](std::vector<PVertex> const& v) {
    LOG(INFO) << "PrimaryVertexWriter pulled " << v.size() << " vertices";
  };
  auto inpID = disableMatching ? InputSpec{"vttrackID", "GLO", "PVTX_CONTID", 0} : InputSpec{"vttrackID", "GLO", "PVTX_TRMTC", 0};
  auto inpIDRef = disableMatching ? InputSpec{"v2tref", "GLO", "PVTX_CONTIDREFS", 0} : InputSpec{"v2tref", "GLO", "PVTX_TRMTCREFS", 0};
  return MakeRootTreeWriterSpec("primary-vertex-writer",
                                "o2_primary_vertex.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with Primary Vertices"},
                                BranchDefinition<std::vector<PVertex>>{InputSpec{"vertices", "GLO", "PVTX", 0}, "PrimaryVertex", logger},
                                BranchDefinition<std::vector<V2TRef>>{inpIDRef, "PV2TrackRefs"},
                                BranchDefinition<std::vector<GIndex>>{inpID, "PVTrackIndices"},
                                BranchDefinition<std::vector<Label>>{InputSpec{"labels", "GLO", "PVTX_MCTR", 0}, "PVMCTruth", (useMC ? 1 : 0), ""})();
}

} // namespace vertexing
} // namespace o2
