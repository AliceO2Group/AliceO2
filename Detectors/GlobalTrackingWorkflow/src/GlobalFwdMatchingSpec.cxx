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

/// @file   GlobalFwdMatchingSpec.cxx

#include <vector>
#include <string>
#include "TStopwatch.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "CommonUtils/StringUtils.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ReconstructionDataFormats/GlobalFwdTrack.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsParameters/GRPObject.h"
#include "GlobalTracking/MatchGlobalFwd.h"
#include "GlobalTrackingWorkflow/GlobalFwdMatchingSpec.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "DetectorsBase/Propagator.h"
#include "TGeoGlobalMagField.h"
#include "Field/MagneticField.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "MCHTracking/TrackExtrap.h"

using namespace o2::framework;
using MCLabelsTr = gsl::span<const o2::MCCompLabel>;
using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace globaltracking
{

class GlobalFwdMatchingDPL : public Task
{
 public:
  GlobalFwdMatchingDPL(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC, bool MatchRootOutput)
    : mDataRequest(dr), mGGCCDBRequest(gr), mUseMC(useMC), mMatchRootOutput(MatchRootOutput) {}
  ~GlobalFwdMatchingDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mMatchRootOutput = false;
  o2::globaltracking::MatchGlobalFwd mMatching;             // Forward matching engine
  const o2::itsmft::TopologyDictionary* mMFTDict = nullptr; // cluster patterns dictionary

  bool mUseMC = true;
  TStopwatch mTimer;
};

void GlobalFwdMatchingDPL::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mMatching.setMCTruthOn(mUseMC);

  const auto& matchingParam = GlobalFwdMatchingParam::Instance();
  if (matchingParam.isMatchUpstream() && mMatchRootOutput) {
    LOG(fatal) << "Invalid MFTMCH matching configuration: matchUpstream and enable-match-output";
  }
}

void GlobalFwdMatchingDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions

  mMatching.run(recoData);

  const auto& matchingParam = GlobalFwdMatchingParam::Instance();

  if (matchingParam.saveMode == kSaveTrainingData) {
    pc.outputs().snapshot(Output{"GLO", "GLFWDMFT", 0, Lifetime::Timeframe}, mMatching.getMFTMatchingPlaneParams());
    pc.outputs().snapshot(Output{"GLO", "GLFWDMCH", 0, Lifetime::Timeframe}, mMatching.getMCHMatchingPlaneParams());
    pc.outputs().snapshot(Output{"GLO", "GLFWDINF", 0, Lifetime::Timeframe}, mMatching.getMFTMCHMatchInfo());
  } else {
    pc.outputs().snapshot(Output{"GLO", "GLFWD", 0, Lifetime::Timeframe}, mMatching.getMatchedFwdTracks());
  }

  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "GLFWD_MC", 0, Lifetime::Timeframe}, mMatching.getMatchLabels());
  }
  if (mMatchRootOutput) {
    pc.outputs().snapshot(Output{"GLO", "MTC_MFTMCH", 0, Lifetime::Timeframe}, mMatching.getMFTMCHMatchInfo());
  }
  mTimer.Stop();
}

void GlobalFwdMatchingDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "Forward matcher total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

void GlobalFwdMatchingDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    if (matcher == ConcreteDataMatcher("GLO", "GRPMAGFIELD", 0)) {
      o2::mch::TrackExtrap::setField();
    }
    return;
  }
  if (matcher == ConcreteDataMatcher("MFT", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mMatching.setMFTDictionary((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
  if (matcher == ConcreteDataMatcher("MFT", "ALPIDEPARAM", 0)) {
    LOG(info) << "MFT Alpide param updated";
    return;
  }
}

void GlobalFwdMatchingDPL::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;

    auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
    double centerMFT[3] = {0, 0, -61.4}; // Field at center of MFT
    auto Bz = field->getBz(centerMFT);
    LOG(info) << "Setting Global forward matching Bz = " << Bz;
    mMatching.setBz(Bz);
    mMatching.setMFTTriggered(!o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::MFT));
    if (o2::base::GRPGeomHelper::instance().getGRPECS()->getRunType() != o2::parameters::GRPECSObject::RunType::COSMICS) {
      mMatching.setBunchFilling(o2::base::GRPGeomHelper::instance().getGRPLHCIF()->getBunchFilling());
    }

    // apply needed settings
    const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
    if (mMatching.isMFTTriggered()) {
      mMatching.setMFTROFrameLengthMUS(alpParams.roFrameLengthTrig / 1.e3); // MFT ROFrame duration in \mus
    } else {
      mMatching.setMFTROFrameLengthInBC(alpParams.roFrameLengthInBC); // MFT ROFrame duration in \mus
    }

    mMatching.init();
  }
  // we may have other params which need to be queried regularly
}

DataProcessorSpec getGlobalFwdMatchingSpec(bool useMC, bool matchRootOutput)
{

  const auto& matchingParam = GlobalFwdMatchingParam::Instance();

  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  o2::dataformats::GlobalTrackID::mask_t src = o2::dataformats::GlobalTrackID::getSourcesMask("MFT,MCH");

  dataRequest->requestMFTClusters(false); // MFT clusters labels are not used
  dataRequest->requestTracks(src, useMC);

  if (matchingParam.isMatchUpstream()) {
    dataRequest->requestMFTMCHMatches(useMC); // Request MFTMCH Matches
  }

  if (matchingParam.useMIDMatch) {
    dataRequest->requestMCHMIDMatches(false); // Request MCHMID Matches. Labels are not used
  }
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              true,                              // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              false,                             // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true); // query only once all objects except mag.field

  if (matchingParam.saveMode == kSaveTrainingData) {
    outputs.emplace_back("GLO", "GLFWDMFT", 0, Lifetime::Timeframe);
    outputs.emplace_back("GLO", "GLFWDMCH", 0, Lifetime::Timeframe);
    outputs.emplace_back("GLO", "GLFWDINF", 0, Lifetime::Timeframe);
  } else {
    outputs.emplace_back("GLO", "GLFWD", 0, Lifetime::Timeframe);
  }

  if (useMC) {
    outputs.emplace_back("GLO", "GLFWD_MC", 0, Lifetime::Timeframe);
  }

  if (matchRootOutput) {
    outputs.emplace_back("GLO", "MTC_MFTMCH", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "globalfwd-track-matcher",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<GlobalFwdMatchingDPL>(dataRequest, ggRequest, useMC, matchRootOutput)},
    Options{}};
}

} // namespace globaltracking
} // namespace o2
