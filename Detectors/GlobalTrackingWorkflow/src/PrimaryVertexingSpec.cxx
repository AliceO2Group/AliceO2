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
#include <TStopwatch.h>
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
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
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsVertexing/PVertexer.h"

using namespace o2::framework;
using DetID = o2::detectors::DetID;
using DataRequest = o2::globaltracking::DataRequest;

namespace o2
{
namespace vertexing
{

namespace o2d = o2::dataformats;

class PrimaryVertexingSpec : public Task
{
 public:
  PrimaryVertexingSpec(std::shared_ptr<DataRequest> dr, bool validateWithIR, bool useMC)
    : mDataRequest(dr), mUseMC(useMC), mValidateWithIR(validateWithIR) {}
  ~PrimaryVertexingSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

 private:
  std::shared_ptr<DataRequest> mDataRequest;
  o2::vertexing::PVertexer mVertexer;
  bool mUseMC{false};          ///< MC flag
  bool mValidateWithIR{false}; ///< require vertex validation with IR (e.g. from FT0)
  float mITSROFrameLengthMUS = 0.;
  TStopwatch mTimer;
};

void PrimaryVertexingSpec::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP();

  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};
  const auto& alpParams = o2::itsmft::DPLAlpideParam<DetID::ITS>::Instance();
  if (!grp->isDetContinuousReadOut(DetID::ITS)) {
    mITSROFrameLengthMUS = alpParams.roFrameLengthTrig / 1.e3; // ITS ROFrame duration in \mus
  } else {
    mITSROFrameLengthMUS = alpParams.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3; // ITS ROFrame duration in \mus
  }

  // this is a hack to provide Mat.LUT from the local file, in general will be provided by the framework from CCDB
  std::string matLUTPath = ic.options().get<std::string>("material-lut-path");
  std::string matLUTFile = o2::base::NameConf::getMatLUTFileName(matLUTPath);
  if (o2::utils::Str::pathExists(matLUTFile)) {
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
  recoData.collectData(pc, *mDataRequest.get());
  // select tracks of needed type, with minimal cuts, the real selected will be done in the vertexer

  std::vector<TrackWithTimeStamp> tracks;
  std::vector<o2::MCCompLabel> tracksMCInfo;
  std::vector<o2d::GlobalTrackID> gids;
  auto maxTrackTimeError = PVertexerParams::Instance().maxTimeErrorMUS;
  auto halfROFITS = 0.5 * mITSROFrameLengthMUS;
  auto hw2ErrITS = 2.f / std::sqrt(12.f) * mITSROFrameLengthMUS; // conversion from half-width to error for ITS

  auto creator = [maxTrackTimeError, hw2ErrITS, halfROFITS, &tracks, &gids](auto& _tr, GTrackID _origID, float t0, float terr) {
    if (!_origID.includesDet(DetID::ITS)) {
      return true; // just in case this selection was not done on RecoContainer filling level
    }
    if constexpr (isITSTrack<decltype(_tr)>()) {
      t0 += halfROFITS;  // ITS time is supplied in \mus as beginning of ROF
      terr *= hw2ErrITS; // error is supplied as a half-ROF duration, convert to \mus
    }
    // for all other tracks the time is in \mus with gaussian error
    if (terr < maxTrackTimeError) {
      tracks.emplace_back(TrackWithTimeStamp{_tr, {t0, terr}});
      gids.emplace_back(_origID);
    }
    return true;
  };

  recoData.createTracksVariadic(creator); // create track sample considered for vertexing

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
    auto ft0all = recoData.getFT0RecPoints();
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
  mVertexer.end();
  LOGF(INFO, "Primary vertexing total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getPrimaryVertexingSpec(GTrackID::mask_t src, bool validateWithFT0, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(src, useMC);
  if (validateWithFT0 && src[GTrackID::FT0]) {
    dataRequest->requestFT0RecPoints(false);
  }

  outputs.emplace_back("GLO", "PVTX", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVTX_CONTID", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVTX_CONTIDREFS", 0, Lifetime::Timeframe);

  if (useMC) {
    outputs.emplace_back("GLO", "PVTX_MCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "primary-vertexing",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<PrimaryVertexingSpec>(dataRequest, validateWithFT0, useMC)},
    Options{{"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}}}};
}

} // namespace vertexing
} // namespace o2
