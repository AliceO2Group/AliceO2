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
#include "ReconstructionDataFormats/DecayNbody.h"
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
#include "DetectorsBase/GRPGeomHelper.h"
#include "TStopwatch.h"
#include "TPCCalibration/VDriftHelper.h"
#include "TPCCalibration/CorrectionMapsLoader.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DeviceSpec.h"

using namespace o2::framework;

using GTrackID = o2::dataformats::GlobalTrackID;
using GIndex = o2::dataformats::VtxTrackIndex;
using VRef = o2::dataformats::VtxTrackRef;
using PVertex = const o2::dataformats::PrimaryVertex;
using V0 = o2::dataformats::V0;
using Cascade = o2::dataformats::Cascade;
using DecayNbody = o2::dataformats::DecayNbody;
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
  SecondaryVertexingSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool enabCasc, bool enable3body) : mDataRequest(dr), mGGCCDBRequest(gr), mEnableCascades(enabCasc), mEnable3BodyVertices(enable3body) {}
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
  bool mEnableCascades = false;
  bool mEnable3BodyVertices = false;
  o2::vertexing::SVertexer mVertexer;
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
}

void SecondaryVertexingSpec::run(ProcessingContext& pc)
{
  double timeCPU0 = mTimer.CpuTime(), timeReal0 = mTimer.RealTime();
  mTimer.Start(false);

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc);

  auto& v0s = pc.outputs().make<std::vector<V0>>(Output{"GLO", "V0S", 0, Lifetime::Timeframe});
  auto& v0Refs = pc.outputs().make<std::vector<RRef>>(Output{"GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe});
  auto& cascs = pc.outputs().make<std::vector<Cascade>>(Output{"GLO", "CASCS", 0, Lifetime::Timeframe});
  auto& cascRefs = pc.outputs().make<std::vector<RRef>>(Output{"GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe});
  auto& vtx3body = pc.outputs().make<std::vector<DecayNbody>>(Output{"GLO", "DECAYS3BODY", 0, Lifetime::Timeframe});
  auto& vtx3bodyRefs = pc.outputs().make<std::vector<RRef>>(Output{"GLO", "PVTX_3BODYREFS", 0, Lifetime::Timeframe});

  mVertexer.process(recoData);
  mVertexer.extractSecondaryVertices(v0s, v0Refs, cascs, cascRefs, vtx3body, vtx3bodyRefs);

  mTimer.Stop();
  LOG(info) << "Found " << v0s.size() << " V0s, " << cascs.size() << " cascades, and " << vtx3body.size() << " 3-body decays; timing: CPU: "
            << mTimer.CpuTime() - timeCPU0 << " Real: " << mTimer.RealTime() - timeReal0 << " s";
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
}

void SecondaryVertexingSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  o2::tpc::VDriftHelper::extractCCDBInputs(pc);
  o2::tpc::CorrectionMapsLoader::extractCCDBInputs(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    mVertexer.init();
    if (pc.services().get<const o2::framework::DeviceSpec>().inputTimesliceId == 0) {
      SVertexerParams::Instance().printKeyValues();
    }
  }
  // we may have other params which need to be queried regularly
  bool updateMaps = false;
  if (mTPCCorrMapsLoader.isUpdated()) {
    mVertexer.setTPCCorrMaps(&mTPCCorrMapsLoader);
    mTPCCorrMapsLoader.acknowledgeUpdate();
    updateMaps = true;
  }
  if (mTPCVDriftHelper.isUpdated()) {
    LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} from source {}",
         mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift, mTPCVDriftHelper.getSourceName());
    mVertexer.setTPCVDrift(mTPCVDriftHelper.getVDriftObject());
    mTPCVDriftHelper.acknowledgeUpdate();
    updateMaps = true;
  }
  if (updateMaps) {
    mTPCCorrMapsLoader.updateVDrift(mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift);
  }
}

DataProcessorSpec getSecondaryVertexingSpec(GTrackID::mask_t src, bool enableCasc, bool enable3body)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  bool useMC = false;
  dataRequest->requestTracks(src, useMC);
  dataRequest->requestPrimaryVertertices(useMC);
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                              true,                           // GRPECS=true
                                                              false,                          // GRPLHCIF
                                                              true,                           // GRPMagField
                                                              true,                           // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry
                                                              dataRequest->inputs,
                                                              true);
  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);
  o2::tpc::CorrectionMapsLoader::requestCCDBInputs(dataRequest->inputs);

  outputs.emplace_back("GLO", "V0S", 0, Lifetime::Timeframe);            // found V0s
  outputs.emplace_back("GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe);    // prim.vertex -> V0s refs
  outputs.emplace_back("GLO", "CASCS", 0, Lifetime::Timeframe);          // found Cascades
  outputs.emplace_back("GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe);  // prim.vertex -> Cascades refs
  outputs.emplace_back("GLO", "DECAYS3BODY", 0, Lifetime::Timeframe);    // found 3 body vertices
  outputs.emplace_back("GLO", "PVTX_3BODYREFS", 0, Lifetime::Timeframe); // prim.vertex -> 3 body vertices refs

  return DataProcessorSpec{
    "secondary-vertexing",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<SecondaryVertexingSpec>(dataRequest, ggRequest, enableCasc, enable3body)},
    Options{{"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}},
            {"threads", VariantType::Int, 1, {"Number of threads"}}}};
}

} // namespace vertexing
} // namespace o2
