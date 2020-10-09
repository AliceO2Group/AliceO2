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

using namespace o2::framework;

namespace o2
{
namespace vertexing
{

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

  // eventually, this should be set per TF from CCDB?
  if (mMatcher.getITSROFrameLengthInBC() == 0) {
    std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom(o2::base::NameConf::getGRPFileName())};
    const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    if (grp->isDetContinuousReadOut(o2::detectors::DetID::ITS)) {
      mMatcher.setITSROFrameLengthInBC(alpParams.roFrameLengthInBC);
    } else {
      mMatcher.setITSROFrameLengthInBC(alpParams.roFrameLengthTrig / o2::constants::lhc::LHCOrbitNS);
    }
  }

  // RS FIXME this will not have effect until the 1st orbit is propagated, until that will work only for TF starting at orbit 0
  const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().get("vertices").header);
  mMatcher.setStartIR({0, dh->firstTForbit});

  const auto tracksITSTPC = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("tpcits");
  const auto tracksTPC = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("tpc");
  const auto tracksITS = pc.inputs().get<gsl::span<o2::its::TrackITS>>("its");
  const auto tracksITSROF = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("itsROF");
  const auto vertices = pc.inputs().get<gsl::span<o2::dataformats::PrimaryVertex>>("vertices");
  const auto vtxTracks = pc.inputs().get<gsl::span<o2::dataformats::VtxTrackIndex>>("vtxTracks");
  const auto vtxTrackRefs = pc.inputs().get<gsl::span<o2::dataformats::VtxTrackRef>>("vtxTrackRefs");

  std::vector<o2::dataformats::VtxTrackIndex> trackIndex;
  std::vector<o2::dataformats::VtxTrackRef> vtxRefs;

  mMatcher.process(vertices, vtxTracks, vtxTrackRefs, tracksITSTPC, tracksITS, tracksITSROF, tracksTPC, trackIndex, vtxRefs);

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

DataProcessorSpec getVertexTrackMatcherSpec()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  inputs.emplace_back("tpcits", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("its", "ITS", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("itsROF", "ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("tpc", "TPC", "TRACKS", 0, Lifetime::Timeframe);

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
