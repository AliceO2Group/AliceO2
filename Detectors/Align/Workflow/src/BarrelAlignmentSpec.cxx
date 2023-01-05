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
#include <TSystem.h>
#include "TMethodCall.h"
#include "AlignmentWorkflow/BarrelAlignmentSpec.h"
#include "Align/AlignableDetectorITS.h"
#include "Align/Controller.h"
#include "Align/AlignConfig.h"
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
#include "CommonUtils/TreeStreamRedirector.h"

#include "Headers/DataHeader.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DeviceSpec.h"

#include <boost/interprocess/sync/named_semaphore.hpp>

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
  enum PostProc { WriteResults = 0x1 << 0,
                  CheckConstaints = 0x1 << 1,
                  GenPedeFiles = 0x1 << 2,
                  LabelPedeResults = 0x1 << 3 };
  BarrelAlignmentSpec(GTrackID::mask_t srcMP, std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> ggrec, DetID::mask_t detmask, int postprocess, bool useMC)
    : mDataRequest(dr), mGRPGeomRequest(ggrec), mMPsrc{srcMP}, mDetMask{detmask}, mPostProcessing(postprocess), mUseMC(useMC) {}
  ~BarrelAlignmentSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::string mIniParFile{};
  bool mUseIniParErrors = true;
  bool mUseMC = false;
  bool mIgnoreCCDBAlignment = false;
  int mPostProcessing = 0; // special mode of extracting alignment or constraints check
  GTrackID::mask_t mMPsrc{};
  DetID::mask_t mDetMask{};
  std::unique_ptr<Controller> mController;
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGRPGeomRequest;
  std::string mConfMacro{};
  std::unique_ptr<TMethodCall> mUsrConfMethod;
  std::unique_ptr<o2::trd::TrackletTransformer> mTRDTransformer;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  //
  TStopwatch mTimer;
};

void BarrelAlignmentSpec::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGRPGeomRequest);
  int dbg = ic.options().get<int>("debug-output"), inst = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
  mController = std::make_unique<Controller>(mDetMask, mMPsrc, mUseMC, inst);
  if (dbg) {
    mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(fmt::format("mpDebug_{}.root", inst).c_str(), "recreate");
    mController->setDebugOutputLevel(dbg);
    mController->setDebugStream(mDBGOut.get());
  }

  mConfMacro = ic.options().get<std::string>("config-macro");
  if (!mConfMacro.empty()) {
    if (!std::filesystem::exists(mConfMacro)) {
      LOG(fatal) << "Requested user macro " << mConfMacro << " does not exist";
    }
    std::string tmpmacro = mConfMacro + "+";
    TString cmd = gSystem->GetMakeSharedLib();
    cmd.ReplaceAll("$Opt", "-O0 -g -ggdb");

    // protect macro compilation by semaphore (to avoid problems in the pipelined code)
    {
      boost::interprocess::named_semaphore* sem = nullptr;
      std::string semhashedstring{};
      std::hash<std::string> hasher;
      semhashedstring = "align_macro_" + std::to_string(hasher(mConfMacro)).substr(0, 16);
      try {
        sem = new boost::interprocess::named_semaphore(boost::interprocess::open_or_create_t{}, semhashedstring.c_str(), 1);
      } catch (std::exception e) {
        LOGP(error, "Exception occurred during {} compilation semaphore setup", tmpmacro);
        sem = nullptr;
      }
      if (sem) {
        sem->wait(); // wait until we can enter (no one else there)
      }
      gSystem->SetMakeSharedLib(cmd.Data());
      auto res = gROOT->LoadMacro(tmpmacro.c_str());
      if (sem) {
        sem->post();
        if (sem->try_wait()) { // if nobody else is waiting remove the semaphore resource
          sem->post();
          boost::interprocess::named_semaphore::remove(semhashedstring.c_str());
        }
      }
      if (res) {
        LOG(fatal) << "Failed to load user macro " << tmpmacro;
      }
    }
    std::filesystem::path mpth(mConfMacro);
    mConfMacro = mpth.stem();
    mUsrConfMethod = std::make_unique<TMethodCall>();
    mUsrConfMethod->InitWithPrototype(mConfMacro.c_str(), "o2::align::Controller*, int");
  }
  mIgnoreCCDBAlignment = ic.options().get<bool>("ignore-ccdb-alignment");
  if (!mPostProcessing) {
    if (GTrackID::includesDet(DetID::TRD, mMPsrc)) {
      mTRDTransformer.reset(new o2::trd::TrackletTransformer);
      if (ic.options().get<bool>("apply-xor")) {
        mTRDTransformer->setApplyXOR();
      }
      mController->setTRDTransformer(mTRDTransformer.get());
    }
    mController->setAllowAfterburnerTracks(ic.options().get<bool>("allow-afterburner-tracks"));
  }
  mIniParFile = ic.options().get<std::string>("initial-params-file");
  mUseIniParErrors = !ic.options().get<bool>("ignore-initial-params-errors");
  if (mPostProcessing && !(mPostProcessing != GenPedeFiles) && (mIniParFile.empty() || mIniParFile == "none")) {
    LOGP(warn, "Postprocessing {} is requested but the initial-params-file is not provided", mPostProcessing);
  }
}

void BarrelAlignmentSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  if (!initOnceDone) {
    initOnceDone = true;
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    if (!mIgnoreCCDBAlignment) {
      for (auto id = DetID::First; id <= DetID::Last; id++) {
        const auto* alg = o2::base::GRPGeomHelper::instance().getAlignment(id);
        if (alg && !alg->empty()) {
          o2::base::GeometryManager::applyAlignment(*alg);
        }
        gGeoManager->RefreshPhysicalNodes(false);
      }
    } else {
      LOG(warn) << "CCDB alignment is NOT applied to ideal geometry";
    }
    if (!mController->getInitGeomDone()) {
      mController->initDetectors();
    }
    if (mTRDTransformer) { // need geometry loaded
      mTRDTransformer->init();
    }

    if (!(mIniParFile.empty() || mIniParFile == "none")) {
      mController->readParameters(mIniParFile, mUseIniParErrors);
      mController->applyAlignmentFromMPSol();
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
    AlignConfig::Instance().printKeyValues(true);
    o2::base::PropagatorD::Instance()->setTGeoFallBackAllowed(false);
  }
  if (GTrackID::includesDet(DetID::TRD, mMPsrc) && mTRDTransformer) {
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
  if (mPostProcessing) { // special mode, no data processing
    if (mController->getInstanceID() == 0) {
      if (mPostProcessing & PostProc::CheckConstaints) {
        mController->addAutoConstraints();
        mController->checkConstraints();
      }
      if (mPostProcessing & PostProc::WriteResults) {
        mController->writeCalibrationResults();
      }
    }
    pc.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
  } else {
    RecoContainer recoData;
    recoData.collectData(pc, *mDataRequest.get());
    mController->setRecoContainer(&recoData);
    mController->setTimingInfo(pc.services().get<o2::framework::TimingInfo>());
    mController->process();
  }
  mTimer.Stop();
}

void BarrelAlignmentSpec::endOfStream(EndOfStreamContext& ec)
{
  auto inst = ec.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
  if (!mPostProcessing) {
    LOGP(info, "Barrel alignment data pereparation total timing: Cpu: {:.3e} Real: {:.3e}  s in {} slots, instance {}", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1, inst);
    mController->closeMPRecOutput();
    mController->closeMilleOutput();
    mController->closeResidOutput();
  }
  if (inst == 0) {
    if (!mPostProcessing || (mPostProcessing & PostProc::GenPedeFiles)) {
      LOG(info) << "Writing millepede control files";
      if (!mPostProcessing) {
        mController->terminate(); // finalize data stat
      }
      mController->addAutoConstraints();
      mController->genPedeSteerFile();
      mController->getStat().print();
    } else if (mPostProcessing & PostProc::LabelPedeResults) {
      mController->writeLabeledPedeResults();
    }
  }
  mDBGOut.reset();
}

DataProcessorSpec getBarrelAlignmentSpec(GTrackID::mask_t srcMP, GTrackID::mask_t src, DetID::mask_t dets, DetID::mask_t skipDetClusters, int postprocess, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  if (!postprocess) {
    dataRequest->requestTracks(src, useMC);
    dataRequest->requestClusters(src, false, skipDetClusters);
    dataRequest->requestPrimaryVertertices(useMC);
    if (GTrackID::includesDet(DetID::TRD, srcMP)) {
      dataRequest->inputs.emplace_back("calvdexb", "TRD", "CALVDRIFTEXB", 0, Lifetime::Condition, ccdbParamSpec("TRD/Calib/CalVdriftExB"));
    }
  }
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                                 // orbitResetTime
                                                                true,                                 // GRPECS=true
                                                                false,                                // GRPLHCIF
                                                                true,                                 // GRPMagField
                                                                true,                                 // askMatLUT
                                                                o2::base::GRPGeomRequest::Alignments, // geometry
                                                                dataRequest->inputs,
                                                                false, // ask update once (except field)
                                                                true); // init PropagatorD
  return DataProcessorSpec{
    "barrel-alignment",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<BarrelAlignmentSpec>(srcMP, dataRequest, ccdbRequest, dets, postprocess, useMC)},
    Options{
      ConfigParamSpec{"apply-xor", o2::framework::VariantType::Bool, false, {"flip the 8-th bit of slope and position (for processing TRD CTFs from 2021 pilot beam)"}},
      ConfigParamSpec{"allow-afterburner-tracks", VariantType::Bool, false, {"allow using ITS-TPC afterburner tracks"}},
      ConfigParamSpec{"ignore-ccdb-alignment", VariantType::Bool, false, {"do not aplly CCDB alignment to ideal geometry"}},
      ConfigParamSpec{"initial-params-file", VariantType::String, "", {"initial parameters file"}},
      ConfigParamSpec{"config-macro", VariantType::String, "", {"configuration macro with signature (o2::align::Controller*, int) to execute from init"}},
      ConfigParamSpec{"ignore-initial-params-errors", VariantType::Bool, false, {"ignore initial params (if any) errors for precondition"}},
      ConfigParamSpec{"debug-output", VariantType::Int, 0, {"produce debugging output root files"}}}};
}

} // namespace align
} // namespace o2
