// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  VertexTrackMatcherSpec.cxx
/// @brief Specs for vertex track association device
/// @author ruben.shahoyan@cern.ch

#include "GlobalTrackingWorkflow/VertexTrackMatcherSpec.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

using namespace o2::framework;
using DetID = o2::detectors::DetID;

namespace o2
{
namespace vertexing
{
o2::globaltracking::DataRequest dataRequestV2T;

void VertexTrackMatcherSpec::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  mTimer.Stop();
  mTimer.Reset();

  mMatcher.init();
}

void VertexTrackMatcherSpec::run(ProcessingContext& pc)
{
  double timeCPU0 = mTimer.CpuTime(), timeReal0 = mTimer.RealTime();
  mTimer.Start(false);

  o2::globaltracking::RecoContainer recoData;

  // RS FIXME this will not have effect until the 1st orbit is propagated, until that will work only for TF starting at orbit 0
  const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().getByPos(0).header);
  mMatcher.setStartIR({0, dh->firstTForbit});
  recoData.collectData(pc, dataRequestV2T);

  const auto vertices = pc.inputs().get<gsl::span<o2::dataformats::PrimaryVertex>>("vertices");
  const auto vtxTracks = pc.inputs().get<gsl::span<o2::dataformats::VtxTrackIndex>>("vtxTracks");
  const auto vtxTrackRefs = pc.inputs().get<gsl::span<o2::dataformats::VtxTrackRef>>("vtxTrackRefs");

  std::vector<o2::dataformats::VtxTrackIndex> trackIndex;
  std::vector<o2::dataformats::VtxTrackRef> vtxRefs;

  mMatcher.process(vertices, vtxTracks, vtxTrackRefs, recoData, trackIndex, vtxRefs);

  pc.outputs().snapshot(Output{"GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe}, trackIndex);
  pc.outputs().snapshot(Output{"GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe}, vtxRefs);

  mTimer.Stop();
  LOG(INFO) << "Made " << trackIndex.size() << " track associationgs for " << vertices.size() << " vertices, timing: CPU: "
            << mTimer.CpuTime() - timeCPU0 << " Real: " << mTimer.RealTime() - timeReal0 << " s";
}

void VertexTrackMatcherSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "Primary vertex - track matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getVertexTrackMatcherSpec(GTrackID::mask_t src)
{
  std::vector<OutputSpec> outputs;

  if (src[GTrackID::ITS]) {
    dataRequestV2T.requestITSTracks(false);
  }
  if (src[GTrackID::TPC]) {
    dataRequestV2T.requestTPCTracks(false);
  }
  if (src[GTrackID::ITSTPC] || src[GTrackID::ITSTPCTOF]) { // ITSTPCTOF does not provide tracks, only matchInfo
    dataRequestV2T.requestITSTPCTracks(false);
  }
  if (src[GTrackID::ITSTPCTOF]) {
    dataRequestV2T.requestTOFMatches(false);
    dataRequestV2T.requestTOFClusters(false);
  }
  if (src[GTrackID::TPCTOF]) {
    dataRequestV2T.requestTPCTOFTracks(false);
  }

  auto& inputs = dataRequestV2T.inputs;
  inputs.emplace_back("vertices", "GLO", "PVTX", 0, Lifetime::Timeframe);
  inputs.emplace_back("vtxTracks", "GLO", "PVTX_CONTID", 0, Lifetime::Timeframe);
  inputs.emplace_back("vtxTrackRefs", "GLO", "PVTX_CONTIDREFS", 0, Lifetime::Timeframe);

  outputs.emplace_back("GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "pvertex-track-matching",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<VertexTrackMatcherSpec>()},
    Options{}};
}

} // namespace vertexing
} // namespace o2
