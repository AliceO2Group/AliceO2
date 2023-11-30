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

#ifndef O2_TRD_DATAPROCESSOR_H
#define O2_TRD_DATAPROCESSOR_H

/// @file   DCSTRDDataProcessorSpec.h
/// @brief  TRD Processor for DCS Data Points

#include <unistd.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/AliasExpander.h"
#include "TRDCalibration/DCSProcessor.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;

class TRDDCSDataProcessor : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {

    std::vector<DPID> vect;
    mGasDPsUpdateInterval = ic.options().get<int64_t>("DPs-update-interval-gas");
    if (mGasDPsUpdateInterval == 0) {
      LOG(error) << "TRD DPs update interval set to zero seconds --> changed to 900s";
      mGasDPsUpdateInterval = 900;
    }
    mCurrentsDPsUpdateInterval = ic.options().get<int64_t>("DPs-update-interval-currents");
    if (mCurrentsDPsUpdateInterval == 0) {
      LOG(error) << "TRD DPs update interval set to zero seconds --> changed to 120s";
      mCurrentsDPsUpdateInterval = 120;
    }
    mVoltagesDPsUpdateInterval = ic.options().get<int64_t>("DPs-update-interval-voltages");
    if (mVoltagesDPsUpdateInterval == 0) {
      LOG(error) << "TRD DPs update interval set to zero seconds --> changed to 600s";
      mVoltagesDPsUpdateInterval = 600;
    }
    mMinUpdateIntervalU = ic.options().get<int64_t>("DPs-min-update-interval-voltages");
    // LB: Env DPs, only update every 30 min
    mEnvDPsUpdateInterval = ic.options().get<int64_t>("DPs-update-interval-env");
    if (mEnvDPsUpdateInterval == 0) {
      LOG(error) << "TRD DPs update interval set to zero seconds --> changed to 1800s";
      mEnvDPsUpdateInterval = 1800;
    }
    bool useCCDBtoConfigure = ic.options().get<bool>("use-ccdb-to-configure");
    if (useCCDBtoConfigure) {
      LOG(info) << "Configuring via CCDB";
      std::string ccdbpath = ic.options().get<std::string>("ccdb-path");
      auto& mgr = o2::ccdb::BasicCCDBManager::instance();
      mgr.setURL(ccdbpath);
      long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      std::unordered_map<DPID, std::string>* dpid2DataDesc = mgr.getForTimeStamp<std::unordered_map<DPID, std::string>>("TRD/Config/DCSDPconfig", ts);
      for (auto& i : *dpid2DataDesc) {
        vect.push_back(i.first);
      }
    } else {
      LOG(info) << "Configuring via hardcoded strings";
      std::vector<std::string> aliasesFloat;
      std::vector<std::string> aliasesInt;
      std::vector<std::string> aliasesString;
      aliasesFloat.insert(aliasesFloat.end(), {"trd_gasCO2", "trd_gasH2O", "trd_gasO2"});
      aliasesFloat.insert(aliasesFloat.end(), {"trd_gaschromatographCO2", "trd_gaschromatographN2", "trd_gaschromatographXe"});
      aliasesFloat.insert(aliasesFloat.end(), {"trd_hvAnodeImon[00..539]", "trd_hvAnodeUmon[00..539]", "trd_hvDriftImon[00..539]", "trd_hvDriftUmon[00..539]"});
      aliasesFloat.insert(aliasesFloat.end(), {"CavernTemperature", "temperature_P2_external"});
      aliasesFloat.insert(aliasesFloat.end(), {"CavernAtmosPressure", "SurfaceAtmosPressure", "CavernAtmosPressure2"});
      aliasesFloat.insert(aliasesFloat.end(), {"UXC2Humidity"});
      aliasesInt.insert(aliasesInt.end(), {"trd_fed_runNo"});
      aliasesInt.insert(aliasesInt.end(), {"trd_chamberStatus[00..539]"});
      aliasesString.insert(aliasesString.end(), {"trd_CFGtag[00..539]"});

      for (const auto& i : o2::dcs::expandAliases(aliasesFloat)) {
        vect.emplace_back(i, o2::dcs::DPVAL_DOUBLE);
      }
      for (const auto& i : o2::dcs::expandAliases(aliasesInt)) {
        vect.emplace_back(i, o2::dcs::DPVAL_INT);
      }
      for (const auto& i : o2::dcs::expandAliases(aliasesString)) {
        vect.emplace_back(i, o2::dcs::DPVAL_STRING);
      }
    }

    LOG(info) << "Listing Data Points for TRD:";
    for (auto& i : vect) {
      LOG(info) << i;
    }

    mProcessor = std::make_unique<o2::trd::DCSProcessor>();
    int verbosity = ic.options().get<int>("processor-verbosity");
    if (verbosity > 0) {
      LOG(info) << "Using verbose mode for TRD DCS processor";
      mProcessor->setVerbosity(verbosity);
    }

    // LB: set maximum number of alarms in change in FedChamberStatus and FedCFGtag
    int alarmfed = ic.options().get<int>("DPs-max-counter-alarm-fed");
    if (alarmfed >= 0) {
      LOG(info) << "Setting max number of alarms in FED objects changes to " << alarmfed;
      mProcessor->setMaxCounterAlarmFed(alarmfed);
    } else {
      LOG(info) << "Invalid max number of alarms in FED objects changes " << alarmfed << ", using default value of 1";
    }

    // LB: set minimum number of DPs in DCS Processor to update ChamberStatus/CFGtag
    int minupdatefed = ic.options().get<int>("DPs-min-counter-update-fed");
    if (minupdatefed >= 0 && minupdatefed <= 540) {
      LOG(info) << "Setting min number of DPs to update ChamberStatus/CFGtag to " << minupdatefed;
      mProcessor->setFedMinimunDPsForUpdate(minupdatefed);
    } else {
      LOG(info) << "Invalid min number of DPs to update ChamberStatus/CFGtag " << alarmfed << ", using default value of 522";
    }

    // LB: set minimum voltage variation to update Anode/DriftUmon
    int utrigger = ic.options().get<int>("DPs-voltage-variation-trigger");
    if (utrigger > 0) {
      LOG(info) << "Setting voltage variation trigger of DPs to update Anode/DriftUMon to " << utrigger;
      mProcessor->setUVariationTriggerForUpdate(utrigger);
    } else {
      LOG(info) << "Invalid voltage variation trigger of DPs to update Anode/DriftUMon to " << utrigger << ", using default value of 1 V";
    }

    mProcessor->init(vect);
    mTimerGas = std::chrono::high_resolution_clock::now();
    mTimerVoltages = mTimerGas;
    mTimerCurrents = mTimerGas;
    mTimerEnv = mTimerGas;
    // LB: new DPs for Fed
    mTimerFedChamberStatus = mTimerGas;
    mTimerFedCFGtag = mTimerGas;

    mReportTiming = ic.options().get<bool>("report-timing") || verbosity > 0;
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    TStopwatch sw;
    auto currentTimeStamp = pc.services().get<o2::framework::TimingInfo>().creation;
    auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");
    auto timeNow = std::chrono::high_resolution_clock::now();
    if (currentTimeStamp < 1577833200000UL || currentTimeStamp > 2208985200000UL) {
      LOG(warning) << "The creation time of this TF is set to " << currentTimeStamp << ". Overwriting this with current time";
      // in case it is not set
      currentTimeStamp = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms
    }
    mProcessor->setCurrentTS(currentTimeStamp);
    mProcessor->process(dps);

    auto elapsedTimeGas = timeNow - mTimerGas; // in ns
    if (elapsedTimeGas.count() * 1e-9 >= mGasDPsUpdateInterval) {
      sendDPsoutputGas(pc.outputs());
      mTimerGas = timeNow;
    }

    auto elapsedTimeVoltages = timeNow - mTimerVoltages; // in ns
    if ((elapsedTimeVoltages.count() * 1e-9 >= mVoltagesDPsUpdateInterval) ||
        (mProcessor->shouldUpdateVoltages() && (elapsedTimeVoltages.count() * 1e-9 >= mMinUpdateIntervalU))) {
      sendDPsoutputVoltages(pc.outputs());
      mTimerVoltages = timeNow;
    }

    auto elapsedTimeCurrents = timeNow - mTimerCurrents; // in ns
    if (elapsedTimeCurrents.count() * 1e-9 >= mCurrentsDPsUpdateInterval) {
      sendDPsoutputCurrents(pc.outputs());
      mTimerCurrents = timeNow;
    }

    auto elapsedTimeEnv = timeNow - mTimerEnv; // in ns
    if (elapsedTimeEnv.count() * 1e-9 >= mEnvDPsUpdateInterval) {
      sendDPsoutputEnv(pc.outputs());
      mTimerEnv = timeNow;
    }

    // LB: processing logic for FedChamberStatus and FedCFGtag
    if (mProcessor->shouldUpdateFedChamberStatus()) {
      sendDPsoutputFedChamberStatus(pc.outputs());
    }

    if (mProcessor->shouldUpdateFedCFGtag()) {
      sendDPsoutputFedCFGtag(pc.outputs());
    }

    if (mProcessor->shouldUpdateRun()) {
      sendDPsoutputRun(pc.outputs());
    }
    sw.Stop();
    if (mReportTiming) {
      LOGP(info, "Timing CPU:{:.3e} Real:{:.3e} at slice {}", sw.CpuTime(), sw.RealTime(), pc.services().get<o2::framework::TimingInfo>().timeslice);
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    sendDPsoutputGas(ec.outputs());
    sendDPsoutputVoltages(ec.outputs());
    sendDPsoutputCurrents(ec.outputs());
    sendDPsoutputEnv(ec.outputs());
    sendDPsoutputRun(ec.outputs());
    // LB: new DPs for Fed
    sendDPsoutputFedChamberStatus(ec.outputs());
    sendDPsoutputFedCFGtag(ec.outputs());
  }

 private:
  bool mReportTiming = false;
  std::unique_ptr<DCSProcessor> mProcessor;
  std::chrono::high_resolution_clock::time_point mTimerGas;
  std::chrono::high_resolution_clock::time_point mTimerVoltages;
  std::chrono::high_resolution_clock::time_point mTimerCurrents;
  std::chrono::high_resolution_clock::time_point mTimerEnv;
  // LB: new DPs for Fed
  std::chrono::high_resolution_clock::time_point mTimerFedChamberStatus;
  std::chrono::high_resolution_clock::time_point mTimerFedCFGtag;

  int64_t mGasDPsUpdateInterval;
  int64_t mVoltagesDPsUpdateInterval;
  int64_t mCurrentsDPsUpdateInterval;
  int64_t mMinUpdateIntervalU;
  int64_t mEnvDPsUpdateInterval;
  // LB: new DPs for Fed
  int64_t mFedChamberStatusDPsUpdateInterval;
  int64_t mFedCFGtagDPsUpdateInterval;

  void sendDPsoutputVoltages(DataAllocator& output)
  {
    // extract CCDB infos and calibration object for DPs
    if (mProcessor->updateVoltagesDPsCCDB()) {
      const auto& payload = mProcessor->getTRDVoltagesDPsInfo();
      auto& info = mProcessor->getccdbVoltagesDPsInfo();
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
      LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
                << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_DCSUDPs", 0}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_DCSUDPs", 0}, info);
      mProcessor->clearVoltagesDPsInfo();
    } else {
      auto& info = mProcessor->getccdbVoltagesDPsInfo();
      LOG(info) << "Not sending object " << info.getPath() << "/" << info.getFileName() << " since no DPs were processed for it";
    }
  }

  void sendDPsoutputCurrents(DataAllocator& output)
  {
    // extract CCDB infos and calibration object for DPs
    if (mProcessor->updateCurrentsDPsCCDB()) {
      const auto& payload = mProcessor->getTRDCurrentsDPsInfo();
      auto& info = mProcessor->getccdbCurrentsDPsInfo();
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
      LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
                << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_DCSIDPs", 0}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_DCSIDPs", 0}, info);
      mProcessor->clearCurrentsDPsInfo();
    } else {
      auto& info = mProcessor->getccdbCurrentsDPsInfo();
      LOG(info) << "Not sending object " << info.getPath() << "/" << info.getFileName() << " since no DPs were processed for it";
    }
  }

  //________________________________________________________________
  void sendDPsoutputGas(DataAllocator& output)
  {
    // extract CCDB infos and calibration object for DPs
    if (mProcessor->updateGasDPsCCDB()) {
      const auto& payload = mProcessor->getTRDGasDPsInfo();
      auto& info = mProcessor->getccdbGasDPsInfo();
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
      LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
                << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_DCSGasDPs", 0}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_DCSGasDPs", 0}, info);
      mProcessor->clearGasDPsInfo();
    } else {
      auto& info = mProcessor->getccdbGasDPsInfo();
      LOG(info) << "Not sending object " << info.getPath() << "/" << info.getFileName() << " since no DPs were processed for it";
    }
  }

  //________________________________________________________________
  void sendDPsoutputEnv(DataAllocator& output)
  {
    // extract CCDB infos and calibration object for DPs
    if (mProcessor->updateEnvDPsCCDB()) {
      const auto& payload = mProcessor->getTRDEnvDPsInfo();
      auto& info = mProcessor->getccdbEnvDPsInfo();
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
      LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
                << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_DCSEnvDPs", 0}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_DCSEnvDPs", 0}, info);
      mProcessor->clearEnvDPsInfo();
    } else {
      auto& info = mProcessor->getccdbEnvDPsInfo();
      LOG(info) << "Not sending object " << info.getPath() << "/" << info.getFileName() << " since no DPs were processed for it";
    }
  }

  //________________________________________________________________
  void sendDPsoutputRun(DataAllocator& output)
  {
    // extract CCDB infos and calibration object for DPs
    if (mProcessor->updateRunDPsCCDB()) {
      const auto& payload = mProcessor->getTRDRunDPsInfo();
      auto& info = mProcessor->getccdbRunDPsInfo();
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
      LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
                << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_DCSRunDPs", 0}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_DCSRunDPs", 0}, info);
      mProcessor->clearRunDPsInfo();
    } else {
      auto& info = mProcessor->getccdbRunDPsInfo();
      // LOG(info) << "Not sending object " << info.getPath() << "/" << info.getFileName() << " since no DPs were processed for it";
      LOG(info) << "Not sending object " << info.getPath() << "/" << info.getFileName() << " as upload of Run DPs was deactivated";
    }
  }

  // LB: new DP for FedChamberStatus
  //________________________________________________________________
  void sendDPsoutputFedChamberStatus(DataAllocator& output)
  {
    // extract CCDB infos and calibration object for DPs
    if (mProcessor->updateFedChamberStatusDPsCCDB()) {
      const auto& payload = mProcessor->getTRDFedChamberStatusDPsInfo();
      auto& info = mProcessor->getccdbFedChamberStatusDPsInfo();
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
      LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
                << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_ChamberStat", 0}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_ChamberStat", 0}, info);
      mProcessor->clearFedChamberStatusDPsInfo();
    } else {
      auto& info = mProcessor->getccdbFedChamberStatusDPsInfo();
      LOG(info) << "Not sending object " << info.getPath() << "/" << info.getFileName() << " since no DPs were processed for it";
    }
  }

  // LB: new DP for FedCFGtag
  //________________________________________________________________
  void sendDPsoutputFedCFGtag(DataAllocator& output)
  {
    // extract CCDB infos and calibration object for DPs
    if (mProcessor->updateFedCFGtagDPsCCDB()) {
      const auto& payload = mProcessor->getTRDFedCFGtagDPsInfo();
      auto& info = mProcessor->getccdbFedCFGtagDPsInfo();
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
      LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
                << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_CFGtag", 0}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_CFGtag", 0}, info);
      mProcessor->clearFedCFGtagDPsInfo();
    } else {
      auto& info = mProcessor->getccdbFedCFGtagDPsInfo();
      LOG(info) << "Not sending object " << info.getPath() << "/" << info.getFileName() << " since no DPs were processed for it";
    }
  }

}; // end class
} // namespace trd

namespace framework
{

DataProcessorSpec getTRDDCSDataProcessorSpec()
{

  std::vector<OutputSpec> outputs;

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_DCSGasDPs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_DCSGasDPs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_DCSUDPs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_DCSUDPs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_DCSIDPs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_DCSIDPs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_DCSRunDPs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_DCSRunDPs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_DCSEnvDPs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_DCSEnvDPs"});
  // LB: new DPs for Fed
  // Must use reduced names due to initializer string cannot exceed descriptor size in Data Format
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_ChamberStat"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_ChamberStat"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_CFGtag"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_CFGtag"});

  return DataProcessorSpec{
    "trd-dcs-data-processor",
    Inputs{{"input", "DCS", "TRDDATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::trd::TRDDCSDataProcessor>()},
    Options{{"ccdb-path", VariantType::String, "http://localhost:8080", {"Path to CCDB"}},
            {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
            {"report-timing", VariantType::Bool, false, {"Report timing for every slice"}},
            {"processor-verbosity", VariantType::Int, 0, {"Increase for more verbose output (max 3)"}},
            {"DPs-update-interval-currents", VariantType::Int64, 120ll, {"Interval (in s) after which to update the DPs CCDB entry for current parameters"}},
            {"DPs-update-interval-voltages", VariantType::Int64, 600ll, {"Interval (in s) after which to update the DPs CCDB entry for voltage parameters"}},
            {"DPs-update-interval-env", VariantType::Int64, 1800ll, {"Interval (in s) after which to update the DPs CCDB entry for environment parameters"}},
            {"DPs-min-update-interval-voltages", VariantType::Int64, 120ll, {"Minimum range to be covered by voltage CCDB object"}},
            {"DPs-voltage-variation-trigger", VariantType::Int64, 1ll, {"Voltage variation trigger for upload of CCDB object"}},
            {"DPs-update-interval-gas", VariantType::Int64, 900ll, {"Interval (in s) after which to update the DPs CCDB entry for gas parameters"}},
            {"DPs-max-counter-alarm-fed", VariantType::Int, 1, {"Maximum number of alarms after FedChamberStatus and FedCFGtag changes, following changes are logged as warnings"}},
            {"DPs-min-counter-update-fed", VariantType::Int, 522, {"Minimum number of DPs to update FedChamberStatus and FedCFGtag objects"}}}};
}

} // namespace framework
} // namespace o2

#endif
