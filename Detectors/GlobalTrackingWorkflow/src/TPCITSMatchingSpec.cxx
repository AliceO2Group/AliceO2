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

/// @file   TPCITSMatchingSpec.cxx

#include <vector>

#include "GlobalTracking/MatchTPCITS.h"
#include "GlobalTracking/MatchTPCITSParams.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsTPC/Constants.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/DataRefUtils.h"
#include "Framework/CCDBParamSpec.h"
#include <string>
#include "TStopwatch.h"
#include "Framework/ConfigParamRegistry.h"
#include "GlobalTrackingWorkflow/TPCITSMatchingSpec.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "GlobalTracking/MatchTPCITSParams.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "Headers/DataHeader.h"
#include "CommonDataFormat/BunchFilling.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TPCCalibration/VDriftHelper.h"
#include "TPCCalibration/CorrectionMapsLoader.h"

using namespace o2::framework;
using MCLabelsCl = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using MCLabelsTr = gsl::span<const o2::MCCompLabel>;
using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace globaltracking
{

class TPCITSMatchingDPL : public Task
{
 public:
  TPCITSMatchingDPL(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, const o2::tpc::CorrectionMapsLoaderGloOpts& sclOpts,
                    bool useFT0, bool calib, bool skipTPCOnly, bool useMC)
    : mDataRequest(dr), mGGCCDBRequest(gr), mUseFT0(useFT0), mCalibMode(calib), mSkipTPCOnly(skipTPCOnly), mUseMC(useMC)
  {
    mTPCCorrMapsLoader.setLumiScaleType(sclOpts.lumiType);
    mTPCCorrMapsLoader.setLumiScaleMode(sclOpts.lumiMode);
  }
  ~TPCITSMatchingDPL() override = default;
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
  o2::globaltracking::MatchTPCITS mMatching; // matching engine
  bool mUseFT0 = false;
  bool mCalibMode = false;
  bool mSkipTPCOnly = false; // to use only externally constrained tracks (for test only)
  bool mUseMC = true;
  TStopwatch mTimer;
};

void TPCITSMatchingDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mMatching.setNThreads(std::max(1, ic.options().get<int>("nthreads")));
  mMatching.setUseBCFilling(!ic.options().get<bool>("ignore-bc-check"));
  mMatching.setDebugFlag(ic.options().get<int>("debug-tree-flags"));
  mTPCCorrMapsLoader.init(ic);
}

void TPCITSMatchingDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions

  static pmr::vector<o2::MCCompLabel> dummyMCLab, dummyMCLabAB;
  static pmr::vector<o2::dataformats::Triplet<float, float, float>> dummyCalib;

  auto& matchedTracks = pc.outputs().make<std::vector<o2::dataformats::TrackTPCITS>>(Output{"GLO", "TPCITS", 0});
  auto& ABTrackletRefs = pc.outputs().make<std::vector<o2::itsmft::TrkClusRef>>(Output{"GLO", "TPCITSAB_REFS", 0});
  auto& ABTrackletClusterIDs = pc.outputs().make<std::vector<int>>(Output{"GLO", "TPCITSAB_CLID", 0});
  auto& matchLabels = mUseMC ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{"GLO", "TPCITS_MC", 0}) : dummyMCLab;
  auto& ABTrackletLabels = mUseMC ? pc.outputs().make<std::vector<o2::MCCompLabel>>(Output{"GLO", "TPCITSAB_MC", 0}) : dummyMCLabAB;
  auto& calib = mCalibMode ? pc.outputs().make<std::vector<o2::dataformats::Triplet<float, float, float>>>(Output{"GLO", "TPCITS_VDTGL", 0}) : dummyCalib;

  mMatching.run(recoData, matchedTracks, ABTrackletRefs, ABTrackletClusterIDs, matchLabels, ABTrackletLabels, calib);

  mTimer.Stop();
}

void TPCITSMatchingDPL::endOfStream(EndOfStreamContext& ec)
{
  mMatching.end();
  LOGF(info, "TPC-ITS matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

void TPCITSMatchingDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
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
  if (matcher == ConcreteDataMatcher("GLO", "ITSTPCPARAM", 0)) {
    LOG(info) << "ITS-TPC Matching params updated from ccdb";
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mMatching.setITSDictionary((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "ALPIDEPARAM", 0)) {
    LOG(info) << "ITS Alpide param updated";
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    par.printKeyValues();
    return;
  }
}

void TPCITSMatchingDPL::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  mTPCVDriftHelper.extractCCDBInputs(pc);
  mTPCCorrMapsLoader.extractCCDBInputs(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<o2::globaltracking::MatchTPCITSParams*>("MatchParam");

    //  Note: ITS/CLUSDICT and ITS/ALPIDEPARAM are requested/loaded by the recocontainer
    const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    if (mMatching.isITSTriggered()) {
      mMatching.setITSROFrameLengthMUS(alpParams.roFrameLengthTrig / 1.e3); // ITS ROFrame duration in \mus
    } else {
      mMatching.setITSROFrameLengthInBC(alpParams.roFrameLengthInBC); // ITS ROFrame duration in \mus
    }
    mMatching.setITSTimeBiasInBC(alpParams.roFrameBiasInBC);
    mMatching.setSkipTPCOnly(mSkipTPCOnly);
    mMatching.setITSTriggered(!o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::ITS));
    mMatching.setMCTruthOn(mUseMC);
    mMatching.setUseFT0(mUseFT0);
    mMatching.setVDriftCalib(mCalibMode);
    if (o2::base::GRPGeomHelper::instance().getGRPECS()->getRunType() != o2::parameters::GRPECSObject::RunType::COSMICS) {
      mMatching.setBunchFilling(o2::base::GRPGeomHelper::instance().getGRPLHCIF()->getBunchFilling());
    } else {
      mMatching.setCosmics(true);
    }
    mMatching.init();
  }
  // we may have other params which need to be queried regularly
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

DataProcessorSpec getTPCITSMatchingSpec(GTrackID::mask_t src, bool useFT0, bool calib, bool skipTPCOnly, bool useMC, const o2::tpc::CorrectionMapsLoaderGloOpts& sclOpts)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  if ((src & GTrackID::getSourcesMask("TPC-TRD,TPC-TOF,TPC-TRD-TOF")).any()) { // preliminary stage of extended workflow ?
    dataRequest->setMatchingInputStrict();
  }
  dataRequest->inputs.emplace_back("MatchParam", "GLO", "ITSTPCPARAM", 0, Lifetime::Condition, ccdbParamSpec("GLO/Config/ITSTPCParam"));

  dataRequest->requestTracks(src, useMC);
  dataRequest->requestTPCClusters(false);
  dataRequest->requestITSClusters(useMC); // Only ITS clusters labels are needed for the afterburner
  if (useFT0) {
    dataRequest->requestFT0RecPoints(false);
  }
  outputs.emplace_back("GLO", "TPCITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "TPCITSAB_REFS", 0, Lifetime::Timeframe); // AftetBurner ITS tracklet references (referred by GlobalTrackID::ITSAB) on cluster indices
  outputs.emplace_back("GLO", "TPCITSAB_CLID", 0, Lifetime::Timeframe); // cluster indices of ITS tracklets attached by the AfterBurner

  if (calib) {
    outputs.emplace_back("GLO", "TPCITS_VDTGL", 0, Lifetime::Timeframe);
  }

  if (useMC) {
    outputs.emplace_back("GLO", "TPCITS_MC", 0, Lifetime::Timeframe);
    outputs.emplace_back("GLO", "TPCITSAB_MC", 0, Lifetime::Timeframe); // AfterBurner ITS tracklet MC
  }
  // Note: ITS/CLUSDICT and ITS/ALPIDEPARAM are requested/loaded by the recocontainer

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                              // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              true,                              // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true); // query only once all objects except mag.field

  Options opts{
    {"nthreads", VariantType::Int, 1, {"Number of afterburner threads"}},
    {"ignore-bc-check", VariantType::Bool, false, {"Do not check match candidate against BC filling"}},
    {"debug-tree-flags", VariantType::Int, 0, {"DebugFlagTypes bit-pattern for debug tree"}}};

  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);
  o2::tpc::CorrectionMapsLoader::requestCCDBInputs(dataRequest->inputs, opts, sclOpts);

  return DataProcessorSpec{
    "itstpc-track-matcher",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCITSMatchingDPL>(dataRequest, ggRequest, sclOpts, useFT0, calib, skipTPCOnly, useMC)},
    opts};
}

} // namespace globaltracking
} // namespace o2
