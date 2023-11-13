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

/// @file   CosmicsMatchingSpec.cxx

#include <vector>
#include <string>
#include "TStopwatch.h"
#include "GlobalTracking/MatchCosmics.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsTPC/Constants.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/ConfigParamRegistry.h"
#include "GlobalTrackingWorkflow/CosmicsMatchingSpec.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Headers/DataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "Framework/Task.h"
#include "Framework/CCDBParamSpec.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TPCCalibration/VDriftHelper.h"
#include "TPCCalibration/CorrectionMapsLoader.h"

using namespace o2::framework;
using MCLabelsTr = gsl::span<const o2::MCCompLabel>;
using GTrackID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

namespace o2
{
namespace globaltracking
{

class CosmicsMatchingSpec : public Task
{
 public:
  CosmicsMatchingSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC) : mDataRequest(dr), mGGCCDBRequest(gr), mUseMC(useMC) {}
  ~CosmicsMatchingSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  o2::tpc::CorrectionMapsLoader mTPCCorrMapsLoader{};
  o2::globaltracking::MatchCosmics mMatching; // matching engine
  bool mUseMC = true;
  TStopwatch mTimer;
};

void CosmicsMatchingSpec::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mMatching.setDebugFlag(ic.options().get<int>("debug-tree-flags"));
  mMatching.setUseMC(mUseMC);
  mTPCCorrMapsLoader.init(ic);
  //
}

void CosmicsMatchingSpec::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions

  mMatching.process(recoData);
  pc.outputs().snapshot(Output{"GLO", "COSMICTRC", 0, Lifetime::Timeframe}, mMatching.getCosmicTracks());
  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "COSMICTRC_MC", 0, Lifetime::Timeframe}, mMatching.getCosmicTracksLbl());
  }
  mTimer.Stop();
}

void CosmicsMatchingSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  mTPCVDriftHelper.extractCCDBInputs(pc);
  mTPCCorrMapsLoader.extractCCDBInputs(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot) | o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L));

    // pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict"); // called by the RecoContainer
    // also alpParams is called by the RecoContainer
    auto grp = o2::base::GRPGeomHelper::instance().getGRPECS();
    const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    if (!grp->isDetContinuousReadOut(DetID::ITS)) {
      mMatching.setITSROFrameLengthMUS(alpParams.roFrameLengthTrig / 1.e3); // ITS ROFrame duration in \mus
    } else {
      mMatching.setITSROFrameLengthMUS(alpParams.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3); // ITS ROFrame duration in \mus
    }
    mMatching.init();
  }
  bool updateMaps = false;
  if (mTPCCorrMapsLoader.isUpdated()) {
    mMatching.setTPCCorrMaps(&mTPCCorrMapsLoader);
    mTPCCorrMapsLoader.acknowledgeUpdate();
    updateMaps = true;
  }
  if (mTPCVDriftHelper.isUpdated()) {
    LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} and DriftTimeOffset correction {} wrt {} from source {}",
         mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift,
         mTPCVDriftHelper.getVDriftObject().timeOffsetCorr, mTPCVDriftHelper.getVDriftObject().refTimeOffset,
         mTPCVDriftHelper.getSourceName());
    mMatching.setTPCVDrift(mTPCVDriftHelper.getVDriftObject());
    mTPCVDriftHelper.acknowledgeUpdate();
    updateMaps = true;
  }
  if (updateMaps) {
    mTPCCorrMapsLoader.updateVDrift(mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift, mTPCVDriftHelper.getVDriftObject().getTimeOffset());
  }
}

void CosmicsMatchingSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
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
    mMatching.setITSDict((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
}

void CosmicsMatchingSpec::endOfStream(EndOfStreamContext& ec)
{
  mMatching.end();
  LOGF(info, "Cosmics matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getCosmicsMatchingSpec(GTrackID::mask_t src, bool useMC, int lumiType)
{
  std::vector<OutputSpec> outputs;
  Options opts{
    {"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}},
    {"debug-tree-flags", VariantType::Int, 0, {"DebugFlagTypes bit-pattern for debug tree"}}};

  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(src, useMC);
  dataRequest->requestClusters(src, false); // no MC labels for clusters needed for refit only

  outputs.emplace_back("GLO", "COSMICTRC", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("GLO", "COSMICTRC_MC", 0, Lifetime::Timeframe);
  }

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);
  o2::tpc::CorrectionMapsLoader::requestCCDBInputs(dataRequest->inputs, opts, lumiType);

  return DataProcessorSpec{
    "cosmics-matcher",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CosmicsMatchingSpec>(dataRequest, ggRequest, useMC)},
    opts};
}

} // namespace globaltracking
} // namespace o2
