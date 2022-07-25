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

#ifndef O2_EMCAL_DATAPROCESSORSPEC_H
#define O2_EMCAL_DATAPROCESSORSPEC_H

/// @file   EMCALDCSDataProcessorSpec.h
/// @brief  EMCAL Processor for DCS Data Points

#include <unistd.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/RunStatusChecker.h"
#include "EMCALCalibration/EMCDCSProcessor.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsParameters/GRPECSObject.h"

using namespace o2::framework;

namespace o2
{
namespace emcal
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;
using namespace o2::ccdb;
using CcdbManager = o2::ccdb::BasicCCDBManager;
using clbUtils = o2::calibration::Utils;
using HighResClock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;
using RunStatus = o2::dcs::RunStatusChecker::RunStatus;

class EMCALDCSDataProcessor : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
    std::string ccdbpath = ic.options().get<std::string>("ccdb-path");
    auto& mgr = CcdbManager::instance();
    mgr.setURL(ccdbpath);

    mCheckRunStartStop = ic.options().get<bool>("follow-emcal-run");

    std::vector<DPID> vect;
    mDPsUpdateInterval = ic.options().get<int64_t>("DPs-update-interval");
    if (mDPsUpdateInterval == 0) {
      LOG(error) << "EMC DPs update interval set to zero seconds --> changed to 60";
      mDPsUpdateInterval = 60;
    }
    bool useCCDBtoConfigure = ic.options().get<bool>("use-ccdb-to-configure");
    if (useCCDBtoConfigure) {
      LOG(info) << "Configuring via CCDB";
      long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      std::unordered_map<DPID, std::string>* dpid2DataDesc = mgr.getForTimeStamp<std::unordered_map<DPID, std::string>>("EMC/Config/DCSDPconfig", ts);
      for (auto& i : *dpid2DataDesc) {
        vect.push_back(i.first);
      }
    } else {
      LOG(info) << "Configuring via hardcoded strings";

      std::vector<std::string> aliasesTEMP = {"EMC_PT_[00..83].Temperature", "EMC_PT_[88..91].Temperature", "EMC_PT_[96..159].Temperature"};
      std::vector<std::string> aliasesUINT = {"EMC_DDL_LIST[0..1]", "EMC_SRU[00..19]_CFG", "EMC_SRU[00..19]_FMVER",
                                              "EMC_TRU[00..45]_PEAKFINDER", "EMC_TRU[00..45]_L0ALGSEL", "EMC_TRU[00..45]_COSMTHRESH",
                                              "EMC_TRU[00..45]_GLOBALTHRESH", "EMC_TRU[00..45]_MASK[0..5]",
                                              "EMC_STU_ERROR_COUNT_TRU[0..67]", "DMC_STU_ERROR_COUNT_TRU[0..55]"};
      std::vector<std::string> aliasesINT = {"EMC_STU_FWVERS", "EMC_STU_GA[0..1]", "EMC_STU_GB[0..1]", "EMC_STU_GC[0..1]",
                                             "EMC_STU_JA[0..1]", "EMC_STU_JB[0..1]", "EMC_STU_JC[0..1]", "EMC_STU_PATCHSIZE", "EMC_STU_GETRAW",
                                             "EMC_STU_MEDIAN", "EMC_STU_REGION", "DMC_STU_FWVERS", "DMC_STU_PHOS_scale[0..3]", "DMC_STU_GA[0..1]",
                                             "DMC_STU_GB[0..1]", "DMC_STU_GC[0..1]", "DMC_STU_JA[0..1]", "DMC_STU_JB[0..1]", "DMC_STU_JC[0..1]",
                                             "DMC_STU_PATCHSIZE", "DMC_STU_GETRAW", "DMC_STU_MEDIAN", "DMC_STU_REGION", "EMC_RUNNUMBER"};

      std::vector<std::string> expaliasesTEMP = o2::dcs::expandAliases(aliasesTEMP);
      std::vector<std::string> expaliasesUINT = o2::dcs::expandAliases(aliasesUINT);
      std::vector<std::string> expaliasesINT = o2::dcs::expandAliases(aliasesINT);

      for (const auto& i : expaliasesTEMP) {
        vect.emplace_back(i, o2::dcs::DPVAL_DOUBLE);
      }
      for (const auto& i : expaliasesINT) {
        vect.emplace_back(i, o2::dcs::DPVAL_INT);
      }
      for (const auto& i : expaliasesUINT) {
        vect.emplace_back(i, o2::dcs::DPVAL_UINT);
      }
    }

    LOG(info) << "Listing Data Points for EMC:";
    for (auto& i : vect) {
      LOG(info) << i;
    }

    mProcessor = std::make_unique<o2::emcal::EMCDCSProcessor>();
    bool useVerboseMode = ic.options().get<bool>("use-verbose-mode");
    LOG(info) << " ************************* Verbose? " << useVerboseMode;
    if (useVerboseMode) {
      mProcessor->useVerboseMode();
    }
    mProcessor->init(vect);
    mTimer = HighResClock::now();
    mReportTiming = ic.options().get<bool>("report-timing") || useVerboseMode;
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    TStopwatch sw;
    if (mCheckRunStartStop) {
      const auto* grp = mRunChecker.check(); // check if there is a run with EMC
      // this is an example of what it will return
      if (mRunChecker.getRunStatus() == RunStatus::NONE) {
        LOGP(info, "No run with is ongoing or finished");
      } else if (mRunChecker.getRunStatus() == RunStatus::START) { // saw new run with wanted detectors
        LOGP(info, "Run {} has started", mRunChecker.getFollowedRun());
        grp->print();
        mProcessor->setRunNumberFromGRP(mRunChecker.getFollowedRun());
      } else if (mRunChecker.getRunStatus() == RunStatus::ONGOING) { // run which was already seen is still ongoing
        LOGP(info, "Run {} is still ongoing", mRunChecker.getFollowedRun());
      } else if (mRunChecker.getRunStatus() == RunStatus::STOP) { // run which was already seen was stopped (EOR seen)
        LOGP(info, "Run {} was stopped", mRunChecker.getFollowedRun());
      }
    } else {
      mProcessor->setRunNumberFromGRP(-2);
    }

    //    auto tfid = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
    auto tfid = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->creation;
    auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");
    mProcessor->setTF(tfid);
    mProcessor->process(dps);
    auto timeNow = HighResClock::now();
    Duration elapsedTime = timeNow - mTimer; // in seconds
    if (elapsedTime.count() >= mDPsUpdateInterval) {
      sendELMButput(pc.outputs());
      mTimer = timeNow;
    }
    if ((mCheckRunStartStop && (mRunChecker.getRunStatus() == RunStatus::START)) || (!mCheckRunStartStop && mProcessor->isUpdateFEEcfg())) {
      sendCFGoutput(pc.outputs());
    }
    sw.Stop();
    if (mReportTiming) {
      LOGP(info, "Timing CPU:{:.3e} Real:{:.3e} at slice {}", sw.CpuTime(), sw.RealTime(), pc.services().get<o2::framework::TimingInfo>().timeslice);
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    sendELMButput(ec.outputs());
    //    sendCFGoutput(ec.outputs());
  }

 private:
  bool mReportTiming{false};
  std::unique_ptr<EMCDCSProcessor> mProcessor;
  HighResClock::time_point mTimer;
  int64_t mDPsUpdateInterval;
  bool mCheckRunStartStop = false;
  o2::dcs::RunStatusChecker mRunChecker{o2::detectors::DetID::getMask("EMC")};

  //________________________________________________________________
  void sendELMButput(DataAllocator& output)
  {
    // extract CCDB infos and Temperature Sensor data

    mProcessor->processElmb();
    mProcessor->updateElmbCCDBinfo();

    const auto& payload = mProcessor->getELMBdata();
    auto& info = mProcessor->getccdbELMBinfo();
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "EMC_ELMB", 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "EMC_ELMB", 0}, info);
  }

  //________________________________________________________________
  void sendCFGoutput(DataAllocator& output)
  {
    // extract CCDB and FeeDCS info

    //    if (mProcessor->isUpdateFEEcfg()) {
    mProcessor->updateFeeCCDBinfo();

    const auto& payload = mProcessor->getFeeDCSdata();
    auto& info = mProcessor->getccdbFeeDCSinfo();
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "EMC_FeeDCS", 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "EMC_FeeDCS", 0}, info);
    //   }
  }

}; // end class
} // namespace emcal

namespace framework
{

DataProcessorSpec getEMCALDCSDataProcessorSpec()
{

  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "EMC_ELMB"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "EMC_ELMB"}, Lifetime::Sporadic);

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "EMC_FeeDCS"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "EMC_FeeDCS"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "emc-dcs-data-processor",
    Inputs{{"input", "DCS", "EMCDATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::emcal::EMCALDCSDataProcessor>()},
    Options{{"ccdb-path", VariantType::String, o2::base::NameConf::getCCDBServer(), {"Path to CCDB"}},
            {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
            {"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}},
            {"report-timing", VariantType::Bool, false, {"Report timing for every slice"}},
            {"follow-emcal-run", VariantType::Bool, false, {"Check EMCAL runs SOR/EOR"}},
            {"DPs-update-interval", VariantType::Int64, 600ll, {"Interval (in s) after which to update the DPs CCDB entry"}}}};
}

} // namespace framework
} // namespace o2

#endif
