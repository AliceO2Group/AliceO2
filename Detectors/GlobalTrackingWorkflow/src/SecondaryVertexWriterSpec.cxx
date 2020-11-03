// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   SecondaryVertexWriterSpec.cxx

#include <vector>

#include "GlobalTrackingWorkflow/SecondaryVertexWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/V0.h"

using namespace o2::framework;

namespace o2
{
namespace vertexing
{
using RRef = o2::dataformats::RangeReference<int, int>;
using V0 = o2::dataformats::V0;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

using namespace o2::header;

DataProcessorSpec getSecondaryVertexWriterSpec()
{
  auto logger = [](std::vector<V0> const& v) {
    LOG(INFO) << "SecondaryVertexWriter pulled " << v.size() << " v0s";
  };
  auto inpID = InputSpec{"v0s", "GLO", "V0s", 0};
  auto inpIDRef = InputSpec{"pv2v0ref", "GLO", "PVTX_V0REFS", 0};
  return MakeRootTreeWriterSpec("secondary-vertex-writer",
                                "o2_secondary_vertex.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with Secondary Vertices"},
                                BranchDefinition<std::vector<V0>>{inpID, "V0s", logger},
                                BranchDefinition<std::vector<RRef>>{inpIDRef, "PV2V0Refs"})();
}

} // namespace vertexing
} // namespace o2
