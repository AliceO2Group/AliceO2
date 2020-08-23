// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  PrimaryVertexingSpec.cxx

#include <vector>
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DetectorsBase/Propagator.h"
#include "GlobalTrackingWorkflow/PrimaryVertexingSpec.h"
#include "SimulationDataFormat/MCEventLabel.h"

using namespace o2::framework;

namespace o2
{
namespace vertexing
{

void PrimaryVertexingSpec::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  o2::base::Propagator::initFieldFromGRP("o2sim_grp.root");
  mTimer.Stop();
  mTimer.Reset();
  mVertexer.init();
}

void PrimaryVertexingSpec::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  double timeCPU0 = mTimer.CpuTime(), timeReal0 = mTimer.RealTime();
  const auto tracksITSTPC = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("match");
  gsl::span<const o2::MCCompLabel> lblITS, lblTPC;
  if (mUseMC) {
    lblITS = pc.inputs().get<gsl::span<o2::MCCompLabel>>("lblITS");
    lblTPC = pc.inputs().get<gsl::span<o2::MCCompLabel>>("lblTPC");
  }
  std::vector<Vertex> vertices;
  std::vector<int> vertexTrackIDs;
  std::vector<V2TRef> v2tRefs;
  std::vector<o2::MCEventLabel> lblVtx;

  mVertexer.process(tracksITSTPC, vertices, vertexTrackIDs, v2tRefs, lblITS, lblTPC, lblVtx);

  pc.outputs().snapshot(Output{"GLO", "PVERTEX", 0, Lifetime::Timeframe}, vertices);
  pc.outputs().snapshot(Output{"GLO", "PVERTEX_TRIDREFS", 0, Lifetime::Timeframe}, v2tRefs);
  pc.outputs().snapshot(Output{"GLO", "PVERTEX_TRID", 0, Lifetime::Timeframe}, vertexTrackIDs);

  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "PVERTEX_MCTR", 0, Lifetime::Timeframe}, lblVtx);
  }

  mTimer.Stop();
  LOG(INFO) << "Found " << vertices.size() << " primary vertices, timing: CPU: "
            << mTimer.CpuTime() - timeCPU0 << " Real: " << mTimer.RealTime() - timeReal0 << " s";
}

void PrimaryVertexingSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "Primary vertexing total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getPrimaryVertexingSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  inputs.emplace_back("match", "GLO", "TPCITS", 0, Lifetime::Timeframe);

  outputs.emplace_back("GLO", "PVERTEX", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVERTEX_TRIDREFS", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVERTEX_TRID", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("lblITS", "GLO", "TPCITS_ITSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("lblTPC", "GLO", "TPCITS_TPCMC", 0, Lifetime::Timeframe);
    outputs.emplace_back("GLO", "PVERTEX_MCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "primary-vertexing",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<PrimaryVertexingSpec>(useMC)},
    Options{}};
}

} // namespace vertexing
} // namespace o2
