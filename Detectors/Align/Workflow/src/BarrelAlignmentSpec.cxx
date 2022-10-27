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

/// @file   BarrelAlignmentSpec.cxx

#include <vector>
#include <string>
#include <filesystem>
#include <TMethodCall.h>
#include <TStopwatch.h>
#include <TROOT.h>
#include "TMethodCall.h"
#include "AlignmentWorkflow/BarrelAlignmentSpec.h"
#include "Align/AlignableDetectorITS.h"
#include "Align/Controller.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsParameters/GRPObject.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "TRDBase/TrackletTransformer.h"

#include "Headers/DataHeader.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"
#include "Framework/CCDBParamSpec.h"

/*
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsTPC/Constants.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/DPLAlpideParam.h"
*/

using namespace o2::framework;
using namespace o2::globaltracking;
using namespace o2::align;

using GTrackID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

namespace o2
{
namespace align
{

class BarrelAlignmentSpec : public Task
{
 public:
  BarrelAlignmentSpec(GTrackID::mask_t srcMP, std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> ggrec, DetID::mask_t detmask)
    : mDataRequest(dr), mGRPGeomRequest(ggrec), mMPsrc{srcMP}, mDetMask{detmask} {}
  ~BarrelAlignmentSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  GTrackID::mask_t mMPsrc{};
  DetID::mask_t mDetMask{};
  std::unique_ptr<Controller> mController;
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGRPGeomRequest;
  std::string mConfMacro{};
  std::unique_ptr<TMethodCall> mUsrConfMethod;
  std::unique_ptr<o2::trd::TrackletTransformer> mTRDTransformer;
  TStopwatch mTimer;
};

void BarrelAlignmentSpec::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGRPGeomRequest);
  mController = std::make_unique<Controller>(mDetMask, mMPsrc);
  mConfMacro = ic.options().get<std::string>("config-macro");
  if (!mConfMacro.empty()) {
    if (!std::filesystem::exists(mConfMacro)) {
      LOG(fatal) << "Requested user macro " << mConfMacro << " does not exist";
    }
    std::string tmpmacro = mConfMacro + "+";
    if (gROOT->LoadMacro(tmpmacro.c_str())) {
      LOG(fatal) << "Failed to load user macro " << tmpmacro;
    }
    std::filesystem::path mpth(mConfMacro);
    mConfMacro = mpth.stem();
    mUsrConfMethod = std::make_unique<TMethodCall>();
    mUsrConfMethod->InitWithPrototype(mConfMacro.c_str(), "o2::align::Controller*, int");
  }
  if (GTrackID::includesDet(DetID::TRD, mMPsrc)) {
    mTRDTransformer.reset(new o2::trd::TrackletTransformer);
    if (ic.options().get<bool>("apply-xor")) {
      mTRDTransformer->setApplyXOR();
    }
    mController->setTRDTransformer(mTRDTransformer.get());
  }
}

void BarrelAlignmentSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  bool initOnceDone = false;
  if (!initOnceDone) {
    initOnceDone = true;
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    if (!mController->getInitGeomDone()) {
      mController->initDetectors();
    }
    if (mTRDTransformer) { // need geometry loaded
      mTRDTransformer->init();
    }

    // call this in the very end
    if (mUsrConfMethod) {
      int dummyPar = 0, ret = 0;
      Controller* tmpPtr = mController.get();
      const void* args[2] = {&tmpPtr, &dummyPar};
      mUsrConfMethod->Execute(nullptr, args, 2, &ret);
      if (ret != 0) {
        LOG(fatal) << "Execution of user method config method " << mConfMacro << " failed with " << ret;
      }
    }
  }
  if (GTrackID::includesDet(DetID::TRD, mMPsrc)) {
    pc.inputs().get<o2::trd::CalVdriftExB*>("calvdexb"); // just to trigger the finaliseCCDB
  }
}

void BarrelAlignmentSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    auto* its = mController->getDetector(o2::detectors::DetID::ITS);
    if (its) {
      LOG(info) << "cluster dictionary updated";
      ((AlignableDetectorITS*)its)->setITSDictionary((const o2::itsmft::TopologyDictionary*)obj);
      return;
    }
  }
  if (matcher == ConcreteDataMatcher("TRD", "CALVDRIFTEXB", 0)) {
    LOG(info) << "CalVdriftExB object has been updated";
    mTRDTransformer->setCalVdriftExB((const o2::trd::CalVdriftExB*)obj);
    return;
  }
}

void BarrelAlignmentSpec::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  updateTimeDependentParams(pc);
  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  mController->setRecoContainer(&recoData);
  mController->setTimingInfo(pc.services().get<o2::framework::TimingInfo>());
  mController->process();

  mTimer.Stop();
}

void BarrelAlignmentSpec::endOfStream(EndOfStreamContext& ec)
{
  mController->closeMPRecOutput();
  mController->closeMilleOutput();
  mController->closeResidOutput();

  mController->addAutoConstraints();
  mController->genPedeSteerFile();

  LOGF(info, "Barrel alignment data pereparation total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getBarrelAlignmentSpec(GTrackID::mask_t srcMP, GTrackID::mask_t src, DetID::mask_t dets)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(src, false);
  dataRequest->requestClusters(src, false);
  dataRequest->requestPrimaryVertertices(false);

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                                 // orbitResetTime
                                                                true,                                 // GRPECS=true
                                                                false,                                // GRPLHCIF
                                                                true,                                 // GRPMagField
                                                                false,                                // askMatLUT
                                                                o2::base::GRPGeomRequest::Alignments, // geometry
                                                                dataRequest->inputs,
                                                                false, // ask update once (except field)
                                                                true); // init PropagatorD
  if (GTrackID::includesDet(DetID::TRD, srcMP)) {
    dataRequest->inputs.emplace_back("calvdexb", "TRD", "CALVDRIFTEXB", 0, Lifetime::Condition, ccdbParamSpec("TRD/Calib/CalVdriftExB"));
  }

  return DataProcessorSpec{
    "barrel-alignment",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<BarrelAlignmentSpec>(srcMP, dataRequest, ccdbRequest, dets)},
    Options{
      ConfigParamSpec{"apply-xor", o2::framework::VariantType::Bool, false, {"flip the 8-th bit of slope and position (for processing TRD CTFs from 2021 pilot beam)"}},
      ConfigParamSpec{"config-macro", VariantType::String, "", {"configuration macro with signature (o2::align::Controller*, int) to execute from init"}}}};
}

} // namespace align
} // namespace o2
