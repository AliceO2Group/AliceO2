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
#include "GlobalTracking/RecoContainer.h"

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
#include "ITSMFTBase/DPLAlpideParam.h"

using namespace o2::framework;

namespace o2
{
namespace vertexing
{
o2::globaltracking::DataRequest dataRequest;
namespace o2d = o2::dataformats;

void PrimaryVertexingSpec::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP("o2sim_grp.root");

  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom(o2::base::NameConf::getGRPFileName())};
  const auto& alpParams = o2::itsmft::DPLAlpideParam<DetID::ITS>::Instance();
  if (!grp->isDetContinuousReadOut(DetID::ITS)) {
    mITSROFrameLengthMUS = alpParams.roFrameLengthTrig / 1.e3; // ITS ROFrame duration in \mus
  } else {
    mITSROFrameLengthMUS = alpParams.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3; // ITS ROFrame duration in \mus
  }

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

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, dataRequest);
  // select tracks of needed type, with minimal cuts, the real selected will be done in the vertexer

  std::vector<TrackWithTimeStamp> tracks;
  std::vector<o2::MCCompLabel> tracksMCInfo;
  std::vector<o2d::GlobalTrackID> gids;
  auto maxTrackTimeError = PVertexerParams::Instance().maxTimeErrorMUS;
  auto hw2ErrITS = 2.f / std::sqrt(12.f) * mITSROFrameLengthMUS; // conversion from half-width to error for ITS

  std::function<void(const o2::track::TrackParCov& _tr, float t0, float terr, GTrackID _origID)> creator =
    [maxTrackTimeError, hw2ErrITS, &tracks, &gids](const o2::track::TrackParCov& _tr, float t0, float terr, GTrackID _origID) {
      if (!_origID.includesDet(DetID::ITS)) {
        return; // just in case this selection was not done on RecoContainer filling level
      }
      if (_origID.getSource() == GTrackID::ITS) { // error is supplied a half-ROF duration, convert to \mus
        terr *= hw2ErrITS;
      }
      if (terr > maxTrackTimeError) {
        return;
      }
      tracks.emplace_back(TrackWithTimeStamp{_tr, {t0, terr}});
      gids.emplace_back(_origID);
    };

  recoData.createTracks(creator); // create track sample considered for vertexing
  if (mUseMC) {
    recoData.fillTrackMCLabels(gids, tracksMCInfo);
  }
  std::vector<PVertex> vertices;
  std::vector<GIndex> vertexTrackIDs;
  std::vector<V2TRef> v2tRefs;
  std::vector<o2::MCEventLabel> lblVtx;

  // RS FIXME this will not have effect until the 1st orbit is propagated, until that will work only for TF starting at orbit 0
  mVertexer.setStartIR({0, DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getByPos(0))->firstTForbit});

  std::vector<o2::InteractionRecord> ft0Data;
  if (mValidateWithIR) { // select BCs for validation
    const o2::ft0::InteractionTag& ft0Params = o2::ft0::InteractionTag::Instance();
    auto ft0all = recoData.getFT0RecPoints<o2::ft0::RecPoints>();
    for (const auto& ftRP : ft0all) {
      if (ft0Params.isSelected(ftRP)) {
        ft0Data.push_back(ftRP.getInteractionRecord());
      }
    }
  }

  mVertexer.process(tracks, gids, ft0Data, vertices, vertexTrackIDs, v2tRefs, tracksMCInfo, lblVtx);
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

DataProcessorSpec getPrimaryVertexingSpec(DetID::mask_t dets, bool validateWithFT0, bool useMC)
{
  std::vector<OutputSpec> outputs;
  if (dets[DetID::ITS]) {
    dataRequest.requestITSTracks(useMC);
  }
  if (dets[DetID::TPC]) {
    dataRequest.requestITSTPCTracks(useMC);
    if (dets[DetID::TRD]) {
      // RSTODO will add once TRD tracking available
    }
    if (dets[DetID::TOF]) {
      dataRequest.requestTOFMatches(useMC);
      dataRequest.requestTOFClusters(false);
    }
  }
  if (validateWithFT0 && dets[DetID::FT0]) {
    dataRequest.requestFT0RecPoints(false);
  }

  outputs.emplace_back("GLO", "PVTX", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVTX_CONTID", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVTX_CONTIDREFS", 0, Lifetime::Timeframe);

  if (useMC) {
    outputs.emplace_back("GLO", "PVTX_MCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "primary-vertexing",
    dataRequest.inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<PrimaryVertexingSpec>(validateWithFT0, useMC)},
    Options{{"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}}}};
}

} // namespace vertexing
} // namespace o2
