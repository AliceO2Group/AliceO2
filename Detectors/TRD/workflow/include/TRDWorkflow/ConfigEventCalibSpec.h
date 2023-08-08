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

#ifndef O2_TRD_CONFIGEVENTCALIBSPEC_H
#define O2_TRD_CONFIGEVENTCALIBSPEC_H

/// \file   ConfigEventCalibSpec.h
/// \brief DPL device for dealing with the configuration events

#include "DetectorsCalibration/Utils.h"
#include "CommonUtils/MemFileHelper.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CCDBParamSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CCDB/BasicCCDBManager.h"
#include "DetectorsBase/GRPGeomHelper.h"

#include "TRDCalibration/CalibratorConfigEvents.h"
#include "DataFormatsTRD/TrapConfigEvent.h"
#include "TRDQC/StatusHelper.h"

#include <chrono>
#include <unistd.h>
#include <memory>

using namespace o2::framework;

namespace o2::trd
{

class ConfigEventCalibDevice : public o2::framework::Task
{
 public:
  // ConfigEventCalibDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}
  ConfigEventCalibDevice(bool dummy) : mIsDummy(dummy) {}

  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    auto delay = ic.options().get<uint32_t>("max-delay");
    mCalibrator = std::make_unique<o2::trd::CalibratorConfigEvents>();
    if (ic.options().get<bool>("enable-root-output")) {
      mCalibrator->createFile();
    }
    // we dont need to get currently valid config as we will *always* write a new one into ccdb on start of run.
    // assuming run is long enough.
    // TODO should the first write be quicker?
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto trapConfigEvent = pc.inputs().get<const std::vector<MCMEvent>>("input");
    const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    // Obtain rough time from the data header (done only once)
    if (mStartTime == 0) {
      o2::dataformats::TFIDInfo ti;
      o2::base::TFIDInfoHelper::fillTFIDInfo(pc, ti);
      if (!ti.isDummy()) {
        mStartTime = ti.creation;
      }
    }
    // consume the incoming partial configuration events.
    mCalibrator->process(trapConfigEvent);
    if (mCalibrator->timeLimitReached()) {
      LOGP(info, "Sending TrapConfigEvent after {} TFs seen, HCID seen {} [{:.2f}%] finalising trapconfig event comparison",
           mNumberConfigTFsProcessed, mCalibrator->countHCIDPresent(), ((float)mCalibrator->countHCIDPresent()) / constants::NCHAMBER * 50.0);
      mCalibrator->collapseRegisterValues();
      if (mCalibrator->isDifferent()) {
        sendOutput(pc.outputs());
        mCalibrator->clearEventStructures();
      }
      // figure out missing mcm and hcid that we should have seen but have not yet seen.
      std::stringstream missingMCM;
      std::stringstream missingHCID;
      mCalibrator->stillMissingHCID(missingHCID);
      mCalibrator->stillMissingMCM(missingMCM);
    }
    ++mNumberConfigTFsProcessed;

    if (pc.transitionState() == TransitionHandlingState::Requested) {
      LOGP(info, "Run stop requested, finalizing");
      mRunStopRequested = true;
      mCalibrator->closeFile();
    }
    // sendOutput(pc.outputs());
    // mCalibrator->clearEventStructures();
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    if (mRunStopRequested) {
      return;
    }
    // TODO should I not be saving at end of stream, *if* sufficient data, i.e. like 5 minutes from last clear?
    mCalibrator->closeFile();
    sendOutput(ec.outputs());
  }

  void stop() final
  {
    mCalibrator->closeFile();
  }

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    const auto& payload = mCalibrator->getCcdbObject();

    auto clName = o2::utils::MemFileHelper::getClassName(payload);
    auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
    std::map<std::string, std::string> metadata;
    long startValidity = mStartTime;
    o2::ccdb::CcdbObjectInfo info("TRD/Calib/ConfigEvent", clName, flName, metadata, startValidity, startValidity + 3 * o2::ccdb::CcdbObjectInfo::MONTH);

    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TRAPEVT", 0}, *image.get()); // o2::trd::TrapConfigEvent
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TRAPEVT", 0}, info);         // root-serialized
  }

 private:
  std::unique_ptr<o2::trd::CalibratorConfigEvents> mCalibrator;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  std::shared_ptr<o2::trd::TrapConfigEvent> mCCDBCurrentTrapConfigEvent;
  bool mRunStopRequested = false; // flag that run was stopped (and the last output is sent)
  uint64_t mStartTime = 0;
  uint64_t mNumberConfigTFsProcessed = 0;
  bool mIsDummy;
};

DataProcessorSpec getTRDConfigEventCalibSpec()
{

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TRAPEVT"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TRAPEVT"}, Lifetime::Sporadic);
  std::vector<InputSpec> inputs;
  inputs.emplace_back("input", "TRD", "TRDCFG");

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return DataProcessorSpec{
    "calib-confevt-calibration",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ConfigEventCalibDevice>(false)}, // TRDConfigEventCalibSpec)},

    Options{
      {"min-to-accumulate", VariantType::Int, 900, {"time to accumulate config events before we check ccdb for a difference, in seconds."}},
      {"enable-store-all", VariantType::Bool, false, {"store all the intermediate values"}},
      {"max-delay", VariantType::UInt32, 2u, {"number of slots in past to consider"}},
      {"enable-root-output", VariantType::Bool, false, {"output configs to a root file"}},
    }};
}

} // namespace o2::trd

#endif // O2_TRD_CONFIGEVENTCALIBSPEC_H
