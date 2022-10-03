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

#ifndef O2_HMPID_DATAPROCESSOR_H
#define O2_HMPID_DATAPROCESSOR_H

#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/RunStatusChecker.h"
#include <unistd.h>

#include "HMPIDCalibration/HMPIDDCSProcessor.h"

#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace hmpid
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;
using namespace o2::ccdb;
using CcdbManager = o2::ccdb::BasicCCDBManager;
using clbUtils = o2::calibration::Utils;
using HighResClock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;
using namespace o2::framework;
using RunStatus = o2::dcs::RunStatusChecker::RunStatus;

class HMPIDDCSDataProcessor : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {

    // will follow HMP-run by default:
    //--local-test is passed only when using it on local installation
    // to verify fits etc

    mLocalTest = ic.options().get<bool>("local-test");

    if (mLocalTest) {
      mCheckRunStartStop = false;
    }
    LOGP(info, "mCheckRunStartStop {} ", mCheckRunStartStop);
    std::vector<DPID> vect;

    bool useCCDBtoConfigure = ic.options().get<bool>("use-ccdb-to-configure");
    LOGP(info, "useCCDBtoConfigure set {} ", useCCDBtoConfigure);
    if (useCCDBtoConfigure) {
      LOG(info) << "Configuring via CCDB";
      std::string ccdbpath = ic.options().get<std::string>("ccdb-path");
      auto& mgr = CcdbManager::instance();
      mgr.setURL(ccdbpath);
      CcdbApi api;
      api.init(mgr.getURL());
      long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
      std::unordered_map<DPID, std::string>* dpid2DataDesc =
        mgr.getForTimeStamp<std::unordered_map<DPID, std::string>>(
          "HMP/Config/DCSDPconfig", ts);
      for (auto& i : *dpid2DataDesc) {
        vect.push_back(i.first);
      }
    } else {
      LOG(info) << "Configuring via hardcoded strings";
      std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);

      for (const auto& i : expaliases) {
        vect.emplace_back(i, o2::dcs::DPVAL_DOUBLE);
        // LOG(info) << i;
      }

    } // end else

    // LOG(info) << "Listing Data Points for HMPID:";
    // for (auto& i : vect) {
    // LOG(info) << i;
    // }

    mProcessor = std::make_unique<o2::hmpid::HMPIDDCSProcessor>();
    bool useVerboseMode = ic.options().get<bool>("use-verbose-mode");
    LOG(info) << " ************************* Verbose?" << useVerboseMode;

    if (useVerboseMode) {
      mProcessor->useVerboseMode();
    }
    mProcessor->init(vect);
    mTimer = HighResClock::now();
  }

  //==========================================================================

  void run(o2::framework::ProcessingContext& pc) final
  {

    auto timeNow = HighResClock::now();
    long dataTime = (long)(pc.services().get<o2::framework::TimingInfo>().creation);
    if (dataTime == 0xffffffffffffffff) {
      dataTime = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms
    }

    /* ef : only for local simulation to verify fits:
    // set startValidity if not set already, and mCheckRunStartStop (--follow-hmpid-run) is not used
    if (mProcessor->getStartValidity() == o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP && mCheckRunStartStop == false)
    {
      mProcessor->setStartValidity(dataTime);
    }
    */

    if (mCheckRunStartStop) {
      const auto* grp = mRunChecker.check(); // check if there is a run with HMP
      // this is an example of what it will return
      if (mRunChecker.getRunStatus() == RunStatus::NONE) {
        LOGP(info, "No run with is ongoing or finished");
      } else if (mRunChecker.getRunStatus() ==
                 RunStatus::START) { // saw new run with wanted detectors
        LOGP(info, "Run {} has started", mRunChecker.getFollowedRun());
        grp->print();
        mProcessor->setRunNumberFromGRP(
          mRunChecker.getFollowedRun()); // ef: just the same as for emcal?
                                         // ef: set startValidity here if run-specific object
        if (mProcessor->getStartValidity() == o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP) {
          mProcessor->setStartValidity(dataTime);
        }
      } else if (mRunChecker.getRunStatus() ==
                 RunStatus::ONGOING) { // run which was already seen is still
                                       // ongoing
        LOGP(info, "Run {} is still ongoing", mRunChecker.getFollowedRun());
      } else if (mRunChecker.getRunStatus() ==
                 RunStatus::STOP) { // run which was already seen was stopped
                                    // (EOR seen)
        LOGP(info, "Run {} was stopped", mRunChecker.getFollowedRun());
      }
    } else {
      mProcessor->setRunNumberFromGRP(-2); // ef: just the same as for emcal?
    }

    // process datapoints:
    auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");
    mProcessor->process(dps);

    // ef: runspecific object : send CCDB
    if (mCheckRunStartStop && (mRunChecker.getRunStatus() == RunStatus::STOP)) {
      mProcessor->finalize();

      mProcessor->setEndValidityRunSpecific(dataTime);

      sendChargeThresOutput(pc.outputs());
      sendRefIndexOutput(pc.outputs());
      mProcessor->clearCCDBObjects(); // clears the vectors
      mProcessor->resetStartValidity();
      mProcessor->resetEndValidity();
    }
  }

  //==========================================================================

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    // ef : only for local testing of Fits etc.:
    if (mLocalTest) {
      auto timeNow = HighResClock::now();
      long dataTime = (long)(ec.services().get<o2::framework::TimingInfo>().creation);
      if (dataTime == 0xffffffffffffffff) {
        dataTime = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms
      }

      mProcessor->setEndValidityRunIndependent(dataTime);
      mProcessor->finalize();

      sendChargeThresOutput(ec.outputs());
      sendRefIndexOutput(ec.outputs());

      mProcessor->clearCCDBObjects(); // clears the vectors
      mProcessor->resetStartValidity();
      mProcessor->resetEndValidity();
    } // <end if mLocalTest>
  }

  //==========================================================================

 private:
  // fill CCDB with ChargeThresh (arQthre)
  void sendChargeThresOutput(o2::framework::DataAllocator& output)
  {
    const auto& payload = mProcessor->getChargeCutObj();
    auto& info = mProcessor->getHmpidChargeInfo();

    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/"
              << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp()
              << " : " << info.getEndValidityTimestamp();
    output.snapshot(
      Output{o2::calibration::Utils::gDataOriginCDBPayload, "ChargeCut", 0},
      *image.get());
    output.snapshot(
      Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ChargeCut", 0},
      info);
  }

  // fill CCDB with RefIndex (arrMean)
  void sendRefIndexOutput(o2::framework::DataAllocator& output)
  {
    // fill CCDB with RefIndex (std::vector<TF1> arrMean)
    const auto& payload = mProcessor->getRefIndexObj();
    auto& info = mProcessor->getccdbRefInfo();

    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/"
              << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp()
              << " : " << info.getEndValidityTimestamp();

    output.snapshot(
      Output{o2::calibration::Utils::gDataOriginCDBPayload, "RefIndex", 0},
      *image.get());
    output.snapshot(
      Output{o2::calibration::Utils::gDataOriginCDBWrapper, "RefIndex", 0},
      info);
  }

  std::vector<std::string> aliases = {
    "HMP_ENV_PENV",
    "HMP_MP[0..6]_GAS_PMWPC",
    "HMP_MP[0..6]_LIQ_LOOP_RAD_[0..2]_IN_TEMP",
    "HMP_MP[0..6]_LIQ_LOOP_RAD_[0..2]_OUT_TEMP",
    "HMP_MP_[0..6]_SEC_[0..5]_HV_VMON",
    "HMP_TRANPLANT_MEASURE_[0..29]_WAVELENGHT",
    "HMP_TRANPLANT_MEASURE_[0..29]_ARGONREFERENCE",
    "HMP_TRANPLANT_MEASURE_[0..29]_ARGONCELL",
    "HMP_TRANPLANT_MEASURE_[0..29]_C6F14REFERENCE",
    "HMP_TRANPLANT_MEASURE_[0..29]_C6F14CELL"};

  bool isRunStarted = false;
  bool mLocalTest = false;
  bool mCheckRunStartStop = true;
  o2::dcs::RunStatusChecker mRunChecker{o2::detectors::DetID::getMask("HMP")};

  std::unique_ptr<HMPIDDCSProcessor> mProcessor;
  HighResClock::time_point mTimer;

}; // end class HMPIDDCSDataProcessor
} // namespace hmpid

namespace framework
{

o2::framework::DataProcessorSpec getHMPIDDCSDataProcessorSpec()
{

  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(
    ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload,
                            "ChargeCut"},
    Lifetime::Sporadic);
  outputs.emplace_back(
    ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper,
                            "ChargeCut"},
    Lifetime::Sporadic);

  outputs.emplace_back(
    ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload,
                            "RefIndex"},
    Lifetime::Sporadic);
  outputs.emplace_back(
    ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper,
                            "RefIndex"},
    Lifetime::Sporadic);

  return o2::framework::DataProcessorSpec{
    "hmp-dcs-data-processor", Inputs{{"input", "DCS", "HMPDATAPOINTS"}},
    outputs, AlgorithmSpec{adaptFromTask<o2::hmpid::HMPIDDCSDataProcessor>()},
    Options{{"ccdb-path",
             VariantType::String,
             o2::base::NameConf::getCCDBServer(),
             {"Path to CCDB"}},
            {"use-ccdb-to-configure",
             VariantType::Bool,
             false,
             {"Use CCDB to configure"}},
            {"local-test",
             VariantType::Bool,
             false,
             {"Local installation test"}}, // Check HMPID runs SOR/EOR by default
            {"use-verbose-mode",
             VariantType::Bool,
             false,
             {"Use verbose mode"}}}};
}

} // namespace framework
} // namespace o2

#endif
