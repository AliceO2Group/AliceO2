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
#include "GlobalTracking/RecoContainer.h"
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

using GTrackID = o2::dataformats::GlobalTrackID;
using GIndex = o2::dataformats::VtxTrackIndex;
using VRef = o2::dataformats::VtxTrackRef;
using PVertex = const o2::dataformats::PrimaryVertex;
using V0 = o2::dataformats::V0;
using Cascade = o2::dataformats::Cascade;
using RRef = o2::dataformats::RangeReference<int, int>;

namespace o2
{
namespace vertexing
{

o2::globaltracking::DataRequest dataRequestSV;
namespace o2d = o2::dataformats;

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
  mVertexer.setEnableCascades(mEnableCascades);
  mVertexer.setNThreads(ic.options().get<int>("threads"));
  mTimer.Stop();
  mTimer.Reset();
  mVertexer.init();
}

void SecondaryVertexingSpec::run(ProcessingContext& pc)
{
  double timeCPU0 = mTimer.CpuTime(), timeReal0 = mTimer.RealTime();
  mTimer.Start(false);

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, dataRequestSV);

  const auto pvertices = pc.inputs().get<gsl::span<o2::dataformats::PrimaryVertex>>("pvtx");
  const auto pvtxTracks = pc.inputs().get<gsl::span<o2::dataformats::VtxTrackIndex>>("pvtx_cont");
  const auto pvtxTrackRefs = pc.inputs().get<gsl::span<o2::dataformats::VtxTrackRef>>("pvtx_tref");

  auto& v0s = pc.outputs().make<std::vector<V0>>(Output{"GLO", "V0S", 0, Lifetime::Timeframe});
  auto& v0Refs = pc.outputs().make<std::vector<RRef>>(Output{"GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe});
  auto& cascs = pc.outputs().make<std::vector<Cascade>>(Output{"GLO", "CASCS", 0, Lifetime::Timeframe});
  auto& cascRefs = pc.outputs().make<std::vector<RRef>>(Output{"GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe});

  mVertexer.process(pvertices, pvtxTracks, pvtxTrackRefs, recoData);
  mVertexer.extractSecondaryVertices(v0s, v0Refs, cascs, cascRefs);

  mTimer.Stop();
  LOG(INFO) << "Found " << v0s.size() << " V0s and " << cascs.size() << " cascades, timing: CPU: "
            << mTimer.CpuTime() - timeCPU0 << " Real: " << mTimer.RealTime() - timeReal0 << " s";
}

void SecondaryVertexingSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "Secondary vertexing total timing: Cpu: %.3e Real: %.3e s in %d slots, nThreads = %d",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1, mVertexer.getNThreads());
}

DataProcessorSpec getSecondaryVertexingSpec(GTrackID::mask_t src, bool enableCasc)
{
  std::vector<OutputSpec> outputs;
  bool useMC = false;
  dataRequestSV.requestTracks(src, false);
  auto& inputs = dataRequestSV.inputs;
  inputs.emplace_back("pvtx", "GLO", "PVTX", 0, Lifetime::Timeframe);                // prim.vertices
  inputs.emplace_back("pvtx_cont", "GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe);     // global ids of associated tracks
  inputs.emplace_back("pvtx_tref", "GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe); // vertex - trackID refs

  outputs.emplace_back("GLO", "V0S", 0, Lifetime::Timeframe);           // found V0s
  outputs.emplace_back("GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe);   // prim.vertex -> V0s refs
  outputs.emplace_back("GLO", "CASCS", 0, Lifetime::Timeframe);         // found Cascades
  outputs.emplace_back("GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe); // prim.vertex -> Cascades refs

  return DataProcessorSpec{
    "secondary-vertexing",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<SecondaryVertexingSpec>(enableCasc)},
    Options{{"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}},
            {"threads", VariantType::Int, 1, {"Number of threads"}}}};
}

} // namespace vertexing
} // namespace o2
