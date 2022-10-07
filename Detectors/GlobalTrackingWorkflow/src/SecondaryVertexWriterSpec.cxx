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

/// @file   SecondaryVertexWriterSpec.cxx

#include <vector>

#include "GlobalTrackingWorkflow/SecondaryVertexWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/DecayNbody.h"

using namespace o2::framework;

namespace o2
{
namespace vertexing
{
using RRef = o2::dataformats::RangeReference<int, int>;
using V0 = o2::dataformats::V0;
using Cascade = o2::dataformats::Cascade;
using DecayNbody = o2::dataformats::DecayNbody;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

using namespace o2::header;

DataProcessorSpec getSecondaryVertexWriterSpec()
{
  auto loggerV = [](std::vector<V0> const& v) {
    LOG(info) << "SecondaryVertexWriter pulled " << v.size() << " v0s";
  };

  auto loggerC = [](std::vector<Cascade> const& v) {
    LOG(info) << "SecondaryVertexWriter pulled " << v.size() << " cascades";
  };

  auto loggerD = [](std::vector<DecayNbody> const& v) {
    LOG(info) << "SecondaryVertexWriter pulled " << v.size() << " decays3body";
  };

  auto inpV0ID = InputSpec{"v0s", "GLO", "V0S", 0};
  auto inpV0RefID = InputSpec{"pv2v0ref", "GLO", "PVTX_V0REFS", 0};
  auto inpCascID = InputSpec{"cascs", "GLO", "CASCS", 0};
  auto inpCascRefID = InputSpec{"pv2cascref", "GLO", "PVTX_CASCREFS", 0};
  auto inp3BodyID = InputSpec{"decays3body", "GLO", "DECAYS3BODY", 0};
  auto inp3BodyRefID = InputSpec{"pv23bodyref", "GLO", "PVTX_3BODYREFS", 0};

  return MakeRootTreeWriterSpec("secondary-vertex-writer",
                                "o2_secondary_vertex.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with Secondary Vertices"},
                                BranchDefinition<std::vector<V0>>{inpV0ID, "V0s", loggerV},
                                BranchDefinition<std::vector<RRef>>{inpV0RefID, "PV2V0Refs"},
                                BranchDefinition<std::vector<Cascade>>{inpCascID, "Cascades", loggerC},
                                BranchDefinition<std::vector<RRef>>{inpCascRefID, "PV2CascRefs"},
                                BranchDefinition<std::vector<DecayNbody>>{inp3BodyID, "Decays3Body", loggerD},
                                BranchDefinition<std::vector<RRef>>{inp3BodyRefID, "PV23BodyRefs"})();
}

} // namespace vertexing
} // namespace o2
