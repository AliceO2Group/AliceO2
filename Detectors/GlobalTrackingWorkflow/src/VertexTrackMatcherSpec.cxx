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
#include "DetectorsVertexing/VertexTrackMatcher.h"
#include "TStopwatch.h"

using namespace o2::framework;
using DetID = o2::detectors::DetID;
using GTrackID = o2::dataformats::GlobalTrackID;
using DataRequest = o2::globaltracking::DataRequest;

namespace o2
{
namespace vertexing
{

class VertexTrackMatcherSpec : public Task
{
 public:
  VertexTrackMatcherSpec(std::shared_ptr<DataRequest> dr) : mDataRequest(dr){};
  ~VertexTrackMatcherSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

 private:
  std::shared_ptr<DataRequest> mDataRequest;
  o2::vertexing::VertexTrackMatcher mMatcher;
  TStopwatch mTimer;
};

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
  recoData.collectData(pc, *mDataRequest.get());

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
  auto dataRequest = std::make_shared<DataRequest>();

  if (src[GTrackID::ITS]) {
    dataRequest->requestITSTracks(false);
  }
  if (src[GTrackID::TPC]) {
    dataRequest->requestTPCTracks(false);
  }
  if (src[GTrackID::ITSTPC] || src[GTrackID::ITSTPCTOF]) { // ITSTPCTOF does not provide tracks, only matchInfo
    dataRequest->requestITSTPCTracks(false);
  }
  if (src[GTrackID::ITSTPCTOF]) {
    dataRequest->requestTOFMatches(false);
    dataRequest->requestTOFClusters(false);
  }
  if (src[GTrackID::TPCTOF]) {
    dataRequest->requestTPCTOFTracks(false);
  }

  auto& inputs = dataRequest->inputs;
  inputs.emplace_back("vertices", "GLO", "PVTX", 0, Lifetime::Timeframe);
  inputs.emplace_back("vtxTracks", "GLO", "PVTX_CONTID", 0, Lifetime::Timeframe);
  inputs.emplace_back("vtxTrackRefs", "GLO", "PVTX_CONTIDREFS", 0, Lifetime::Timeframe);

  outputs.emplace_back("GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "pvertex-track-matching",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<VertexTrackMatcherSpec>(dataRequest)},
    Options{}};
}

} // namespace vertexing
} // namespace o2
