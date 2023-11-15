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
#include "ReconstructionDataFormats/Decay3Body.h"

using namespace o2::framework;

namespace o2
{
namespace vertexing
{
using RRef = o2::dataformats::RangeReference<int, int>;
using V0 = o2::dataformats::V0;
using V0Index = o2::dataformats::V0Index;
using Cascade = o2::dataformats::Cascade;
using CascadeIndex = o2::dataformats::CascadeIndex;
using Decay3Body = o2::dataformats::Decay3Body;
using Decay3BodyIndex = o2::dataformats::Decay3BodyIndex;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

using namespace o2::header;

DataProcessorSpec getSecondaryVertexWriterSpec()
{
  auto loggerV = [](std::vector<V0Index> const& v) {
    LOG(info) << "SecondaryVertexWriter pulled " << v.size() << " v0s";
  };

  auto loggerC = [](std::vector<CascadeIndex> const& v) {
    LOG(info) << "SecondaryVertexWriter pulled " << v.size() << " cascades";
  };

  auto loggerD = [](std::vector<Decay3BodyIndex> const& v) {
    LOG(info) << "SecondaryVertexWriter pulled " << v.size() << " decays3body";
  };

  auto inpV0 = InputSpec{"v0s", "GLO", "V0S", 0};
  auto inpV0ID = InputSpec{"v0sIdx", "GLO", "V0S_IDX", 0};
  auto inpV0Ref = InputSpec{"pv2v0ref", "GLO", "PVTX_V0REFS", 0};
  auto inpCasc = InputSpec{"cascs", "GLO", "CASCS", 0};
  auto inpCascID = InputSpec{"cascsIdx", "GLO", "CASCS_IDX", 0};
  auto inpCascRef = InputSpec{"pv2cascref", "GLO", "PVTX_CASCREFS", 0};
  auto inp3Body = InputSpec{"decays3body", "GLO", "DECAYS3BODY", 0};
  auto inp3BodyID = InputSpec{"decays3bodyIdx", "GLO", "DECAYS3BODY_IDX", 0};
  auto inp3BodyRef = InputSpec{"pv23bodyref", "GLO", "PVTX_3BODYREFS", 0};

  return MakeRootTreeWriterSpec("secondary-vertex-writer",
                                "o2_secondary_vertex.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with Secondary Vertices"},
                                BranchDefinition<std::vector<V0Index>>{inpV0ID, "V0sID", loggerV},
                                BranchDefinition<std::vector<V0>>{inpV0, "V0s"},
                                BranchDefinition<std::vector<RRef>>{inpV0Ref, "PV2V0Refs"},

                                BranchDefinition<std::vector<CascadeIndex>>{inpCascID, "CascadesID", loggerC},
                                BranchDefinition<std::vector<Cascade>>{inpCasc, "Cascades"},
                                BranchDefinition<std::vector<RRef>>{inpCascRef, "PV2CascRefs"},

                                BranchDefinition<std::vector<Decay3BodyIndex>>{inp3BodyID, "Decays3BodyID", loggerD},
                                BranchDefinition<std::vector<Decay3Body>>{inp3Body, "Decays3Body"},
                                BranchDefinition<std::vector<RRef>>{inp3BodyRef, "PV23BodyRefs"})();
}

} // namespace vertexing
} // namespace o2
