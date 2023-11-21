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

/// @file  SecondaryVertexingSpec.cxx

#include <vector>
#include "DataFormatsCalibration/MeanVertexObject.h"
#include "Framework/CCDBParamSpec.h"
#include "ReconstructionDataFormats/Decay3Body.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsITS/TrackITS.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "GlobalTrackingWorkflow/SecondaryVertexingSpec.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsVertexing/SVertexer.h"
#include "StrangenessTracking/StrangenessTracker.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TStopwatch.h"
#include "TPCCalibration/VDriftHelper.h"
#include "TPCCalibration/CorrectionMapsLoader.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DeviceSpec.h"
#include "TPCCalibration/CorrectionMapsLoader.h"

using namespace o2::framework;

using GTrackID = o2::dataformats::GlobalTrackID;
using GIndex = o2::dataformats::VtxTrackIndex;
using VRef = o2::dataformats::VtxTrackRef;
using PVertex = const o2::dataformats::PrimaryVertex;
using V0 = o2::dataformats::V0;
using Cascade = o2::dataformats::Cascade;
using Decay3Body = o2::dataformats::Decay3Body;
using RRef = o2::dataformats::RangeReference<int, int>;
using DataRequest = o2::globaltracking::DataRequest;

namespace o2
{
namespace vertexing
{

namespace o2d = o2::dataformats;

class SecondaryVertexingSpec : public Task
{
 public:
  SecondaryVertexingSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, GTrackID::mask_t src, bool enabCasc, bool enable3body, bool enableStrangenessTracking, bool useMC) : mDataRequest(dr), mGGCCDBRequest(gr), mSrc(src), mEnableCascades(enabCasc), mEnable3BodyVertices(enable3body), mEnableStrangenessTracking(enableStrangenessTracking), mUseMC(useMC) {}
  ~SecondaryVertexingSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  o2::tpc::CorrectionMapsLoader mTPCCorrMapsLoader{};
  GTrackID::mask_t mSrc{};
  bool mEnableCascades = false;
  bool mEnable3BodyVertices = false;
  bool mEnableStrangenessTracking = false;
  bool mUseMC = false;
  o2::vertexing::SVertexer mVertexer;
  o2::strangeness_tracking::StrangenessTracker mStrTracker;
  TStopwatch mTimer;
};

void SecondaryVertexingSpec::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  //-------- init geometry and field --------//
  mVertexer.setEnableCascades(mEnableCascades);
  mVertexer.setEnable3BodyDecays(mEnable3BodyVertices);
  mVertexer.setNThreads(ic.options().get<int>("threads"));
  mVertexer.setUseMC(mUseMC);
  if (mEnableStrangenessTracking) {
    mStrTracker.setCorrType(o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT);
    mStrTracker.setConfigParams(&o2::strangeness_tracking::StrangenessTrackingParamConfig::Instance());
    mStrTracker.setupThreads(ic.options().get<int>("threads"));
    mStrTracker.setupFitters();
    mStrTracker.setMCTruthOn(mUseMC);
    mVertexer.setStrangenessTracker(&mStrTracker);
  }
  if (mSrc[GTrackID::TPC]) {
    mTPCCorrMapsLoader.init(ic);
  }
}

void SecondaryVertexingSpec::run(ProcessingContext& pc)
{
  double timeCPU0 = mTimer.CpuTime(), timeReal0 = mTimer.RealTime();
  mTimer.Start(false);
  static std::array<size_t, 3> fitCalls{};

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc);

  mVertexer.process(recoData, pc);

  mTimer.Stop();
  auto calls = mVertexer.getNFitterCalls();
  LOGP(info, "Found {} V0s ({} fits), {} cascades ({} fits), {} 3-body decays ({} fits), {} strange tracks. Timing: CPU: {:.2f} Real: {:.2f} s",
       mVertexer.getNV0s(), calls[0] - fitCalls[0], mVertexer.getNCascades(), calls[1] - fitCalls[1], mVertexer.getN3Bodies(), calls[2] - fitCalls[2], mVertexer.getNStrangeTracks(),
       mTimer.CpuTime() - timeCPU0, mTimer.RealTime() - timeReal0);
  fitCalls = calls;
}

void SecondaryVertexingSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "Secondary vertexing total timing: Cpu: %.3e Real: %.3e s in %d slots, nThreads = %d",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1, mVertexer.getNThreads());
}

void SecondaryVertexingSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
    return;
  }
  if (mTPCCorrMapsLoader.accountCCDBInputs(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mStrTracker.setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
  if (matcher == ConcreteDataMatcher("GLO", "MEANVERTEX", 0)) {
    LOG(info) << "Imposing new MeanVertex: " << ((const o2::dataformats::MeanVertexObject*)obj)->asString();
    mVertexer.setMeanVertex((const o2::dataformats::MeanVertexObject*)obj);
    return;
  }
}

void SecondaryVertexingSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (mSrc[GTrackID::TPC]) {
    mTPCVDriftHelper.extractCCDBInputs(pc);
    mTPCCorrMapsLoader.extractCCDBInputs(pc);
  }
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    mVertexer.init();
    if (pc.services().get<const o2::framework::DeviceSpec>().inputTimesliceId == 0) {
      SVertexerParams::Instance().printKeyValues();
    }
    if (mEnableStrangenessTracking) {
      o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
      geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
    }
  }
  // we may have other params which need to be queried regularly
  if (mSrc[GTrackID::TPC]) {
    bool updateMaps = false;
    if (mTPCCorrMapsLoader.isUpdated()) {
      mVertexer.setTPCCorrMaps(&mTPCCorrMapsLoader);
      mTPCCorrMapsLoader.acknowledgeUpdate();
      updateMaps = true;
    }
    if (mTPCVDriftHelper.isUpdated()) {
      LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
           mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift,
           mTPCVDriftHelper.getVDriftObject().timeOffsetCorr, mTPCVDriftHelper.getVDriftObject().refTimeOffset,
           mTPCVDriftHelper.getSourceName());
      mVertexer.setTPCVDrift(mTPCVDriftHelper.getVDriftObject());
      mTPCVDriftHelper.acknowledgeUpdate();
      updateMaps = true;
    }
    if (updateMaps) {
      mTPCCorrMapsLoader.updateVDrift(mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift, mTPCVDriftHelper.getVDriftObject().getTimeOffset());
    }
  }
  if (mEnableStrangenessTracking) {
    if (o2::base::Propagator::Instance()->getNominalBz() != mStrTracker.getBz()) {
      mStrTracker.setBz(o2::base::Propagator::Instance()->getNominalBz());
      mStrTracker.setupFitters();
    }
  }

  pc.inputs().get<o2::dataformats::MeanVertexObject*>("meanvtx");
}

DataProcessorSpec getSecondaryVertexingSpec(GTrackID::mask_t src, bool enableCasc, bool enable3body, bool enableStrangenesTracking, bool useMC, const o2::tpc::CorrectionMapsLoaderGloOpts& sclOpts)
{
  std::vector<OutputSpec> outputs;
  Options opts{
    {"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}},
    {"threads", VariantType::Int, 1, {"Number of threads"}}};
  auto dataRequest = std::make_shared<DataRequest>();
  GTrackID::mask_t srcClus{};
  if (enableStrangenesTracking) {
    src |= (srcClus = GTrackID::getSourceMask(GTrackID::ITS));
  }
  if (src[GTrackID::TPC]) {
    srcClus |= GTrackID::getSourceMask(GTrackID::TPC);
  }
  if (srcClus.any()) {
    dataRequest->requestClusters(srcClus, useMC);
  }
  dataRequest->requestTracks(src, useMC);
  dataRequest->requestPrimaryVertertices(useMC);
  dataRequest->inputs.emplace_back("meanvtx", "GLO", "MEANVERTEX", 0, Lifetime::Condition, ccdbParamSpec("GLO/Calib/MeanVertex", {}, 1));
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                                                                                         // orbitResetTime
                                                              true,                                                                                          // GRPECS=true
                                                              false,                                                                                         // GRPLHCIF
                                                              true,                                                                                          // GRPMagField
                                                              true,                                                                                          // askMatLUT
                                                              enableStrangenesTracking ? o2::base::GRPGeomRequest::Aligned : o2::base::GRPGeomRequest::None, // geometry
                                                              dataRequest->inputs,
                                                              true);
  if (src[GTrackID::TPC]) {
    o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);
    o2::tpc::CorrectionMapsLoader::requestCCDBInputs(dataRequest->inputs, opts, sclOpts);
  }
  outputs.emplace_back("GLO", "V0S_IDX", 0, Lifetime::Timeframe);        // found V0s indices
  outputs.emplace_back("GLO", "V0S", 0, Lifetime::Timeframe);            // found V0s
  outputs.emplace_back("GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe);    // prim.vertex -> V0s refs

  outputs.emplace_back("GLO", "CASCS_IDX", 0, Lifetime::Timeframe);      // found Cascades indices
  outputs.emplace_back("GLO", "CASCS", 0, Lifetime::Timeframe);          // found Cascades
  outputs.emplace_back("GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe);  // prim.vertex -> Cascades refs

  outputs.emplace_back("GLO", "DECAYS3BODY_IDX", 0, Lifetime::Timeframe); // found 3 body vertices indices
  outputs.emplace_back("GLO", "DECAYS3BODY", 0, Lifetime::Timeframe);    // found 3 body vertices
  outputs.emplace_back("GLO", "PVTX_3BODYREFS", 0, Lifetime::Timeframe); // prim.vertex -> 3 body vertices refs

  if (enableStrangenesTracking) {
    outputs.emplace_back("GLO", "STRANGETRACKS", 0, Lifetime::Timeframe); // found strange track
    outputs.emplace_back("GLO", "CLUSUPDATES", 0, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back("GLO", "STRANGETRACKS_MC", 0, Lifetime::Timeframe);
      LOG(info) << "Strangeness tracker will use MC";
    }
  }

  return DataProcessorSpec{
    "secondary-vertexing",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<SecondaryVertexingSpec>(dataRequest, ggRequest, src, enableCasc, enable3body, enableStrangenesTracking, useMC)},
    opts};
}

} // namespace vertexing
} // namespace o2
