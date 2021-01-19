// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  SecondaryVertexingSpec.cxx

#include <vector>
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsITS/TrackITS.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "GlobalTrackingWorkflow/SecondaryVertexingSpec.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "CommonDataFormat/BunchFilling.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "Framework/ConfigParamRegistry.h"

using namespace o2::framework;

using GIndex = o2::dataformats::VtxTrackIndex;
using VRef = o2::dataformats::VtxTrackRef;
using PVertex = const o2::dataformats::PrimaryVertex;
using V0 = o2::dataformats::V0;
using RRef = o2::dataformats::RangeReference<int, int>;

namespace o2
{
namespace vertexing
{

void SecondaryVertexingSpec::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP("o2sim_grp.root");
  // this is a hack to provide Mat.LUT from the local file, in general will be provided by the framework from CCDB
  std::string matLUTPath = ic.options().get<std::string>("material-lut-path");
  std::string matLUTFile = o2::base::NameConf::getMatLUTFileName(matLUTPath);
  if (o2::base::NameConf::pathExists(matLUTFile)) {
    auto* lut = o2::base::MatLayerCylSet::loadFromFile(matLUTFile);
    o2::base::Propagator::Instance()->setMatLUT(lut);
    LOG(INFO) << "Loaded material LUT from " << matLUTFile;
  } else {
    LOG(INFO) << "Material LUT " << matLUTFile << " file is absent, only TGeo can be used";
  }
  mTimer.Stop();
  mTimer.Reset();

  mVertexer.init();
}

void SecondaryVertexingSpec::run(ProcessingContext& pc)
{
  double timeCPU0 = mTimer.CpuTime(), timeReal0 = mTimer.RealTime();
  mTimer.Start(false);

  const auto tracksITSTPC = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("tpcits");
  const auto tracksTPC = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("tpc");
  const auto tracksITS = pc.inputs().get<gsl::span<o2::its::TrackITS>>("its");

  const auto pvertices = pc.inputs().get<gsl::span<o2::dataformats::PrimaryVertex>>("pvtx");
  const auto pvtxTracks = pc.inputs().get<gsl::span<o2::dataformats::VtxTrackIndex>>("pvtx_cont");
  const auto pvtxTrackRefs = pc.inputs().get<gsl::span<o2::dataformats::VtxTrackRef>>("pvtx_tref");

  std::vector<V0> v0s;
  std::vector<RRef> pv2v0ref;

  o2::dataformats::GlobalTrackAccessor tracksPool;
  tracksPool.registerContainer(tracksITSTPC, GIndex::ITSTPC);
  tracksPool.registerContainer(tracksITS, GIndex::ITS);
  tracksPool.registerContainer(tracksTPC, GIndex::TPC);

  mVertexer.process(pvertices, pvtxTracks, pvtxTrackRefs, tracksPool, v0s, pv2v0ref);

  pc.outputs().snapshot(Output{"GLO", "V0s", 0, Lifetime::Timeframe}, v0s);
  pc.outputs().snapshot(Output{"GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe}, pv2v0ref);

  mTimer.Stop();
  LOG(INFO) << "Found " << v0s.size() << " V0s, timing: CPU: "
            << mTimer.CpuTime() - timeCPU0 << " Real: " << mTimer.RealTime() - timeReal0 << " s";
}

void SecondaryVertexingSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "Secondary vertexing total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getSecondaryVertexingSpec()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  inputs.emplace_back("pvtx", "GLO", "PVTX", 0, Lifetime::Timeframe);                // prim.vertices
  inputs.emplace_back("pvtx_cont", "GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe);     // global ids of associated tracks
  inputs.emplace_back("pvtx_tref", "GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe); // vertex - trackID refs
  //
  inputs.emplace_back("tpcits", "GLO", "TPCITS", 0, Lifetime::Timeframe); // matched ITS-TPC tracks
  inputs.emplace_back("its", "ITS", "TRACKS", 0, Lifetime::Timeframe);    // standalone ITS tracks
  inputs.emplace_back("tpc", "TPC", "TRACKS", 0, Lifetime::Timeframe);    // standalone TPC tracks
  //
  outputs.emplace_back("GLO", "V0s", 0, Lifetime::Timeframe);         // found V0s
  outputs.emplace_back("GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe); // prim.vertex -> V0s refs

  return DataProcessorSpec{
    "secondary-vertexing",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<SecondaryVertexingSpec>()},
    Options{{"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}}}};
}

} // namespace vertexing
} // namespace o2
