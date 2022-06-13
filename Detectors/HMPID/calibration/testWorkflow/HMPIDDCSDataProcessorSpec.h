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

/// @file   DCSHMPIDDataProcessorSpec.h
/// @brief  HMPID Processor for DCS Data Points

#include <unistd.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/RunStatusChecker.h"

#include "HMPIDCalibration/HMPIDDCSProcessor.h"

#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPECSObject.h"

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
    mCheckRunStartStop = ic.options().get<bool>("follow-hmpid-run");

    std::vector<DPID> vect;

    bool useCCDBtoConfigure = ic.options().get<bool>("use-ccdb-to-configure");
    if (useCCDBtoConfigure) {
      LOG(info) << "Configuring via CCDB";
      std::string ccdbpath = ic.options().get<std::string>("ccdb-path");
      auto& mgr = CcdbManager::instance();
      mgr.setURL(ccdbpath);
      CcdbApi api;
      api.init(mgr.getURL());
      long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      std::unordered_map<DPID, std::string>* dpid2DataDesc = mgr.getForTimeStamp<std::unordered_map<DPID, std::string>>("HMP/Config/DCSDPconfig", ts);
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
    //   LOG(info) << i;
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
    if (mCheckRunStartStop) {
      const auto* grp = mRunChecker.check(); // check if there is a run with HMP
      // this is an example of what it will return
      if (mRunChecker.getRunStatus() == RunStatus::NONE) {
        LOGP(info, "No run with is ongoing or finished");
      } else if (mRunChecker.getRunStatus() == RunStatus::START) { // saw new run with wanted detectors
        LOGP(info, "Run {} has started", mRunChecker.getFollowedRun());
        grp->print();
      } else if (mRunChecker.getRunStatus() == RunStatus::ONGOING) { // run which was already seen is still ongoing
        LOGP(info, "Run {} is still ongoing", mRunChecker.getFollowedRun());
      } else if (mRunChecker.getRunStatus() == RunStatus::STOP) { // run which was already seen was stopped (EOR seen)
        LOGP(info, "Run {} was stopped", mRunChecker.getFollowedRun());
      }
    }

    auto startValidity = DataRefUtils::getHeader<DataProcessingHeader*>(pc.inputs().getFirstValid(true))->creation;

    auto timeNow = HighResClock::now();

    // process datapoints:
    auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");
    mProcessor->process(dps);
    if (startValidity == 0xffffffffffffffff) {                                                                   // it means it is not set
      startValidity = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms
    }

    mProcessor->setStartValidity(startValidity);
  }

  //==========================================================================

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    mProcessor->finalize();

    sendChargeThresOutput(ec.outputs());
    sendRefIndexOutput(ec.outputs());
  }

  //==========================================================================

 private:
  // fill CCDB with ChargeThresh (arQthre)
  void sendChargeThresOutput(o2::framework::DataAllocator& output)
  {
    const auto& payload = mProcessor->getChargeCutObj();
    auto& info = mProcessor->getHmpidChargeInfo();

    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ChargeCut", 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ChargeCut", 0}, info);
  }

  // fill CCDB with RefIndex (arrMean)
  void sendRefIndexOutput(o2::framework::DataAllocator& output)
  {
    // fill CCDB with RefIndex (std::vector<TF1> arrMean)
    const auto& payload = mProcessor->getRefIndexObj();
    auto& info = mProcessor->getccdbRefInfo();

    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "RefIndex", 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "RefIndex", 0}, info);
  }

  std::vector<std::string> aliases =
    {"HMP_ENV_PENV",
     "HMP_MP[0..6]_GAS_PMWPC",
     "HMP_MP[0..6]_LIQ_LOOP_RAD_[0..2]_IN_TEMP",
     "HMP_MP[0..6]_LIQ_LOOP_RAD_[0..2]_OUT_TEMP",
     "HMP_MP_[0..6]_SEC_[0..5]_HV_VMON",
     "HMP_TRANPLANT_MEASURE_[0..29]_WAVELENGHT",
     "HMP_TRANPLANT_MEASURE_[0..29]_ARGONREFERENCE",
     "HMP_TRANPLANT_MEASURE_[0..29]_ARGONCELL",
     "HMP_TRANPLANT_MEASURE_[0..29]_C6F14REFERENCE",
     "HMP_TRANPLANT_MEASURE_[0..29]_C6F14CELL"};

  bool mCheckRunStartStop = false;
  o2::dcs::RunStatusChecker mRunChecker{o2::detectors::DetID::getMask("HMP")};

  std::unique_ptr<HMPIDDCSProcessor> mProcessor;
  HighResClock::time_point mTimer;
  int64_t mDPsUpdateInterval;

}; // end class HMPIDDCSDataProcessor
} // namespace hmpid

namespace framework
{

o2::framework::DataProcessorSpec getHMPIDDCSDataProcessorSpec()
{

  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ChargeCut"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ChargeCut"}, Lifetime::Sporadic);

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "RefIndex"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "RefIndex"}, Lifetime::Sporadic);

  return o2::framework::DataProcessorSpec{
    "hmp-dcs-data-processor",
    Inputs{{"input", "DCS", "HMPDATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::hmpid::HMPIDDCSDataProcessor>()},
    Options{{"ccdb-path", VariantType::String, "localhost:8080", {"Path to CCDB"}},
            {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
            {"follow-hmpid-run", VariantType::Bool, false, {"Check HMPID runs SOR/EOR"}},
            {"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}}}};
}

} // namespace framework
} // namespace o2

#endif
