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
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "GlobalTrackingWorkflow/PrimaryVertexingSpec.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "CommonDataFormat/BunchFilling.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsFT0/RecPoints.h"
#include "Framework/ConfigParamRegistry.h"
#include "FT0Reconstruction/InteractionTag.h"

using namespace o2::framework;

namespace o2
{
namespace vertexing
{

void PrimaryVertexingSpec::init(InitContext& ic)
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
  mVertexer.setValidateWithIR(mValidateWithIR);

  // set bunch filling. Eventually, this should come from CCDB
  const auto* digctx = o2::steer::DigitizationContext::loadFromFile("collisioncontext.root");
  const auto& bcfill = digctx->getBunchFilling();
  mVertexer.setBunchFilling(bcfill);
  mVertexer.init();
}

void PrimaryVertexingSpec::run(ProcessingContext& pc)
{
  double timeCPU0 = mTimer.CpuTime(), timeReal0 = mTimer.RealTime();
  mTimer.Start(false);
  const auto tracksITSTPC = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("match");
  gsl::span<const o2::MCCompLabel> lblITS, lblTPC;
  if (mUseMC) {
    lblITS = pc.inputs().get<gsl::span<o2::MCCompLabel>>("lblITS");
    lblTPC = pc.inputs().get<gsl::span<o2::MCCompLabel>>("lblTPC");
  }
  std::vector<PVertex> vertices;
  std::vector<GIndex> vertexTrackIDs;
  std::vector<V2TRef> v2tRefs;
  std::vector<o2::MCEventLabel> lblVtx;

  // RS FIXME this will not have effect until the 1st orbit is propagated, until that will work only for TF starting at orbit 0
  const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().get("match").header);
  mVertexer.setStartIR({0, dh->firstTForbit});
  std::vector<o2::dataformats::GlobalTrackID> idxVec; // here we will the global IDs of all used tracks

  // eventually we will mix here the tracks from different sources
  idxVec.reserve(tracksITSTPC.size());
  for (unsigned i = 0; i < tracksITSTPC.size(); i++) {
    idxVec.emplace_back(i, o2::dataformats::GlobalTrackID::ITSTPC);
  }

  std::vector<o2::ft0::RecPoints> ft0Data;
  if (mValidateWithIR) { // select BCs for validation
    const o2::ft0::InteractionTag& ft0Params = o2::ft0::InteractionTag::Instance();
    auto ftall = pc.inputs().get<gsl::span<o2::ft0::RecPoints>>("fitInfo");
    for (const auto& ftRP : ftall) {
      if (ft0Params.isSelected(ftRP)) {
        ft0Data.push_back(ftRP);
      }
    }
  }

  mVertexer.process(tracksITSTPC, idxVec, ft0Data, vertices, vertexTrackIDs, v2tRefs, lblITS, lblTPC, lblVtx);
  pc.outputs().snapshot(Output{"GLO", "PVTX", 0, Lifetime::Timeframe}, vertices);
  pc.outputs().snapshot(Output{"GLO", "PVTX_CONTIDREFS", 0, Lifetime::Timeframe}, v2tRefs);
  pc.outputs().snapshot(Output{"GLO", "PVTX_CONTID", 0, Lifetime::Timeframe}, vertexTrackIDs);

  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "PVTX_MCTR", 0, Lifetime::Timeframe}, lblVtx);
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

DataProcessorSpec getPrimaryVertexingSpec(bool validateWithFT0, bool useMC)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  inputs.emplace_back("match", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  if (validateWithFT0) {
    inputs.emplace_back("fitInfo", "FT0", "RECPOINTS", 0, Lifetime::Timeframe);
  }

  outputs.emplace_back("GLO", "PVTX", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVTX_CONTID", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVTX_CONTIDREFS", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("lblITS", "GLO", "TPCITS_ITSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("lblTPC", "GLO", "TPCITS_TPCMC", 0, Lifetime::Timeframe);
    outputs.emplace_back("GLO", "PVTX_MCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "primary-vertexing",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<PrimaryVertexingSpec>(validateWithFT0, useMC)},
    Options{{"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}}}};
}

} // namespace vertexing
} // namespace o2
