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

/// @file  VertexTrackMatcherSpec.cxx
/// @brief Specs for vertex track association device
/// @author ruben.shahoyan@cern.ch

#include "Framework/ConfigParamRegistry.h"
#include "Framework/DeviceSpec.h"
#include "GlobalTrackingWorkflow/VertexTrackMatcherSpec.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsVertexing/VertexTrackMatcher.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCCalibration/VDriftHelper.h"
#include "ITSMFTBase/DPLAlpideParam.h"
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
  VertexTrackMatcherSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr) : mDataRequest(dr), mGGCCDBRequest(gr){};
  ~VertexTrackMatcherSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  o2::vertexing::VertexTrackMatcher mMatcher;
  TStopwatch mTimer;
};

void VertexTrackMatcherSpec::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  mTimer.Stop();
  mTimer.Reset();
  mMatcher.setPrescaleLogs(ic.options().get<int>("prescale-logs"));
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
}

void VertexTrackMatcherSpec::run(ProcessingContext& pc)
{
  double timeCPU0 = mTimer.CpuTime(), timeReal0 = mTimer.RealTime();
  mTimer.Start(false);

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc); // make sure called after recoData.collectData as some objects might be fetched there

  std::vector<o2::dataformats::VtxTrackIndex> trackIndex;
  std::vector<o2::dataformats::VtxTrackRef> vtxRefs;

  mMatcher.process(recoData, trackIndex, vtxRefs);

  pc.outputs().snapshot(Output{"GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe}, trackIndex);
  pc.outputs().snapshot(Output{"GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe}, vtxRefs);

  mTimer.Stop();
  LOG(info) << "Made " << trackIndex.size() << " track associations for " << recoData.getPrimaryVertices().size()
            << " vertices, timing: CPU: " << mTimer.CpuTime() - timeCPU0 << " Real: " << mTimer.RealTime() - timeReal0 << " s";
}

void VertexTrackMatcherSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  mTPCVDriftHelper.extractCCDBInputs(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // put here init-once stuff
    const auto& alpParamsITS = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    mMatcher.setITSROFrameLengthMUS(o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::ITS) ? alpParamsITS.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingMUS : alpParamsITS.roFrameLengthTrig * 1.e-3);
    const auto& alpParamsMFT = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
    mMatcher.setMFTROFrameLengthMUS(o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::MFT) ? alpParamsMFT.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingMUS : alpParamsMFT.roFrameLengthTrig * 1.e-3);
    LOGP(info, "VertexTrackMatcher ITSROFrameLengthMUS:{} MFTROFrameLengthMUS:{}", mMatcher.getITSROFrameLengthMUS(), mMatcher.getMFTROFrameLengthMUS());
  }
  // we may have other params which need to be queried regularly
  // VDrift may change from time to time
  if (mTPCVDriftHelper.isUpdated()) {
    auto& elParam = o2::tpc::ParameterElectronics::Instance();
    auto& detParam = o2::tpc::ParameterDetector::Instance();
    mMatcher.setTPCBin2MUS(elParam.ZbinWidth);
    auto& vd = mTPCVDriftHelper.getVDriftObject();
    mMatcher.setMaxTPCDriftTimeMUS(detParam.TPClength / (vd.refVDrift * vd.corrFact));
    mMatcher.setTPCTDriftOffset(vd.getTimeOffset());
    LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
         vd.corrFact, vd.refVDrift, vd.timeOffsetCorr, vd.refTimeOffset, mTPCVDriftHelper.getSourceName());
    mTPCVDriftHelper.acknowledgeUpdate();
  }
}

void VertexTrackMatcherSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "ALPIDEPARAM", 0)) {
    LOG(info) << "ITS Alpide param updated";
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    par.printKeyValues();
    return;
  }
  if (matcher == ConcreteDataMatcher("MFT", "ALPIDEPARAM", 0)) {
    LOG(info) << "MFT Alpide param updated";
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
    par.printKeyValues();
    return;
  }
}

void VertexTrackMatcherSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "Primary vertex - track matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getVertexTrackMatcherSpec(GTrackID::mask_t src)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(src, false);
  dataRequest->requestClusters(src & GTrackID::getSourcesMask("EMC,PHS,CPV"), false);
  dataRequest->requestPrimaryVerterticesTMP(false);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                              true,                           // GRPECS=true
                                                              true,                           // GRPLHCIF
                                                              false,                          // GRPMagField
                                                              false,                          // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry
                                                              dataRequest->inputs,
                                                              true);
  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);

  outputs.emplace_back("GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "pvertex-track-matching",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<VertexTrackMatcherSpec>(dataRequest, ggRequest)},
    Options{{"prescale-logs", VariantType::Int, 50, {"print vertex logs for each n-th TF"}}}};
}

} // namespace vertexing
} // namespace o2
