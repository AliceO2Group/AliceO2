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
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsTPC/Constants.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/DataRefUtils.h"
#include <string>
#include "TStopwatch.h"
#include "Framework/ConfigParamRegistry.h"
#include "GlobalTrackingWorkflow/TPCITSMatchingSpec.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/DigitizationContext.h"
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
#include "DataFormatsParameters/GRPObject.h"
#include "Headers/DataHeader.h"
#include "CommonDataFormat/BunchFilling.h"
#include "CommonDataFormat/Pair.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ITSMFTReconstruction/ClustererParam.h"

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
  TPCITSMatchingDPL(std::shared_ptr<DataRequest> dr, bool useFT0, bool calib, bool skipTPCOnly, bool useMC)
    : mDataRequest(dr), mUseFT0(useFT0), mCalibMode(calib), mSkipTPCOnly(skipTPCOnly), mUseMC(useMC) {}
  ~TPCITSMatchingDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<DataRequest> mDataRequest;
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
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP();
  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};
  mMatching.setSkipTPCOnly(mSkipTPCOnly);
  mMatching.setITSTriggered(!grp->isDetContinuousReadOut(o2::detectors::DetID::ITS));

  mMatching.setMCTruthOn(mUseMC);
  mMatching.setUseFT0(mUseFT0);
  mMatching.setVDriftCalib(mCalibMode);
  mMatching.setNThreads(std::max(1, ic.options().get<int>("nthreads")));
  mMatching.setUseBCFilling(!ic.options().get<bool>("ignore-bc-check"));
  //
  // this is a hack to provide Mat.LUT from the local file, in general will be provided by the framework from CCDB
  std::string matLUTPath = ic.options().get<std::string>("material-lut-path");
  std::string matLUTFile = o2::base::NameConf::getMatLUTFileName(matLUTPath);
  if (o2::utils::Str::pathExists(matLUTFile)) {
    auto* lut = o2::base::MatLayerCylSet::loadFromFile(matLUTFile);
    o2::base::Propagator::Instance()->setMatLUT(lut);
    LOG(info) << "Loaded material LUT from " << matLUTFile;
  } else {
    LOG(info) << "Material LUT " << matLUTFile << " file is absent, only TGeo can be used";
  }

  int dbgFlags = ic.options().get<int>("debug-tree-flags");
  mMatching.setDebugFlag(dbgFlags);

  // set bunch filling. Eventually, this should come from CCDB
  if (mMatching.getUseBCFilling()) {
    const auto* digctx = o2::steer::DigitizationContext::loadFromFile();
    const auto& bcfill = digctx->getBunchFilling();
    mMatching.setBunchFilling(bcfill);
  }
  //
}

void TPCITSMatchingDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions

  mMatching.run(recoData);

  pc.outputs().snapshot(Output{"GLO", "TPCITS", 0, Lifetime::Timeframe}, mMatching.getMatchedTracks());
  pc.outputs().snapshot(Output{"GLO", "TPCITSAB_REFS", 0, Lifetime::Timeframe}, mMatching.getABTrackletRefs());
  pc.outputs().snapshot(Output{"GLO", "TPCITSAB_CLID", 0, Lifetime::Timeframe}, mMatching.getABTrackletClusterIDs());
  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "TPCITS_MC", 0, Lifetime::Timeframe}, mMatching.getMatchLabels());
    pc.outputs().snapshot(Output{"GLO", "TPCITSAB_MC", 0, Lifetime::Timeframe}, mMatching.getABTrackletLabels());
  }

  if (mCalibMode) {
    pc.outputs().snapshot(Output{"GLO", "TPCITS_VDTGL", 0, Lifetime::Timeframe}, mMatching.getTglITSTPC());
  }
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
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;

    //  Note: ITS/CLUSDICT and ITS/ALPIDEPARAM are requested/loaded by the recocontainer
    const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    if (mMatching.isITSTriggered()) {
      mMatching.setITSROFrameLengthMUS(alpParams.roFrameLengthTrig / 1.e3); // ITS ROFrame duration in \mus
    } else {
      mMatching.setITSROFrameLengthInBC(alpParams.roFrameLengthInBC); // ITS ROFrame duration in \mus
    }
    mMatching.init();
  }
  // we may have other params which need to be queried regularly
}

DataProcessorSpec getTPCITSMatchingSpec(GTrackID::mask_t src, bool useFT0, bool calib, bool skipTPCOnly, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  if ((src & GTrackID::getSourcesMask("TPC-TRD,TPC-TOF,TPC-TRD-TOF")).any()) { // preliminary stage of extended workflow ?
    dataRequest->setMatchingInputStrict();
  }

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

  return DataProcessorSpec{
    "itstpc-track-matcher",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCITSMatchingDPL>(dataRequest, useFT0, calib, skipTPCOnly, useMC)},
    Options{
      {"nthreads", VariantType::Int, 1, {"Number of afterburner threads"}},
      {"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}},
      {"ignore-bc-check", VariantType::Bool, false, {"Do not check match candidate against BC filling"}},
      {"debug-tree-flags", VariantType::Int, 0, {"DebugFlagTypes bit-pattern for debug tree"}}}};
}

} // namespace globaltracking
} // namespace o2
