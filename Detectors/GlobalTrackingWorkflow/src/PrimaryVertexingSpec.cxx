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
#include "CommonUtils/NameConf.h"
#include "DataFormatsFT0/RecPoints.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "FT0Reconstruction/InteractionTag.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsVertexing/PVertexer.h"
#include "DetectorsBase/GRPGeomHelper.h"

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
  PrimaryVertexingSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool skip, bool validateWithIR, bool useMC)
    : mDataRequest(dr), mGGCCDBRequest(gr), mSkip(skip), mUseMC(useMC), mValidateWithIR(validateWithIR) {}
  ~PrimaryVertexingSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  o2::vertexing::PVertexer mVertexer;
  bool mSkip{false};           ///< skip vertexing
  bool mUseMC{false};          ///< MC flag
  bool mValidateWithIR{false}; ///< require vertex validation with IR (e.g. from FT0)
  float mITSROFrameLengthMUS = 0.;
  TStopwatch mTimer;
};

void PrimaryVertexingSpec::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mVertexer.setValidateWithIR(mValidateWithIR);
}

void PrimaryVertexingSpec::run(ProcessingContext& pc)
{
  double timeCPU0 = mTimer.CpuTime(), timeReal0 = mTimer.RealTime();
  mTimer.Start(false);
  std::vector<PVertex> vertices;
  std::vector<GIndex> vertexTrackIDs;
  std::vector<V2TRef> v2tRefs;
  std::vector<o2::MCEventLabel> lblVtx;

  if (!mSkip) {
    o2::globaltracking::RecoContainer recoData;
    recoData.collectData(pc, *mDataRequest.get()); // select tracks of needed type, with minimal cuts, the real selected will be done in the vertexer
    updateTimeDependentParams(pc);                 // Make sure this is called after recoData.collectData, which may load some conditions

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
      if constexpr (std::is_base_of_v<o2::track::TrackParCov, std::decay_t<decltype(_tr)>>) {
        if (terr < maxTrackTimeError) {
          tracks.emplace_back(TrackWithTimeStamp{_tr, {t0, terr}});
          gids.emplace_back(_origID);
        }
      }
      return true;
    };

    recoData.createTracksVariadic(creator); // create track sample considered for vertexing

    if (mUseMC) {
      recoData.fillTrackMCLabels(gids, tracksMCInfo);
    }
    mVertexer.setStartIR(recoData.startIR);
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
  }

  pc.outputs().snapshot(Output{"GLO", "PVTX", 0, Lifetime::Timeframe}, vertices);
  pc.outputs().snapshot(Output{"GLO", "PVTX_CONTIDREFS", 0, Lifetime::Timeframe}, v2tRefs);
  pc.outputs().snapshot(Output{"GLO", "PVTX_CONTID", 0, Lifetime::Timeframe}, vertexTrackIDs);

  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "PVTX_MCTR", 0, Lifetime::Timeframe}, lblVtx);
  }

  mTimer.Stop();
  LOG(info) << "Found " << vertices.size() << " primary vertices, timing: CPU: "
            << mTimer.CpuTime() - timeCPU0 << " Real: " << mTimer.RealTime() - timeReal0 << " s";
}

void PrimaryVertexingSpec::endOfStream(EndOfStreamContext& ec)
{
  mVertexer.end();
  LOGF(info, "Primary vertexing total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

void PrimaryVertexingSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  // Note: strictly speaking, for Configurable params we don't need finaliseCCDB check, the singletons are updated at the CCDB fetcher level
  if (matcher == ConcreteDataMatcher("ITS", "ALPIDEPARAM", 0)) {
    LOG(info) << "ITS Alpide param updated";
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    par.printKeyValues();
    return;
  }
}

void PrimaryVertexingSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // Note: reading of the ITS AlpideParam needed for ITS timing is done by the RecoContainer
    auto grp = o2::base::GRPGeomHelper::instance().getGRPECS();
    const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    if (!grp->isDetContinuousReadOut(DetID::ITS)) {
      mITSROFrameLengthMUS = alpParams.roFrameLengthTrig / 1.e3; // ITS ROFrame duration in \mus
    } else {
      mITSROFrameLengthMUS = alpParams.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3; // ITS ROFrame duration in \mus
    }
    if (o2::base::GRPGeomHelper::instance().getGRPECS()->getRunType() != o2::parameters::GRPECSObject::RunType::COSMICS) {
      mVertexer.setBunchFilling(o2::base::GRPGeomHelper::instance().getGRPLHCIF()->getBunchFilling());
    }
    mVertexer.setITSROFrameLength(mITSROFrameLengthMUS);
    mVertexer.init();
  }
  // we may have other params which need to be queried regularly
}

DataProcessorSpec getPrimaryVertexingSpec(GTrackID::mask_t src, bool skip, bool validateWithFT0, bool useMC)
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

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              true,                              // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);

  return DataProcessorSpec{
    "primary-vertexing",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<PrimaryVertexingSpec>(dataRequest, ggRequest, skip, validateWithFT0, useMC)},
    Options{{"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}}}};
}

} // namespace vertexing
} // namespace o2
