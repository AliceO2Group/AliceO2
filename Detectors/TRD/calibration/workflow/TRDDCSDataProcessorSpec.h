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
      LOG(error) << "TRD DPs update interval set to zero seconds --> changed to 60s";
      mCurrentsDPsUpdateInterval = 60;
    }
    mVoltagesDPsUpdateInterval = ic.options().get<int64_t>("DPs-update-interval-voltages");
    if (mVoltagesDPsUpdateInterval == 0) {
      LOG(error) << "TRD DPs update interval set to zero seconds --> changed to 600s";
      mVoltagesDPsUpdateInterval = 600;
    }
    mMinUpdateIntervalU = ic.options().get<int64_t>("DPs-min-update-interval-voltages");
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
      aliasesFloat.insert(aliasesFloat.end(), {"trd_aliEnvTempCavern", "trd_aliEnvTempP2"});
      aliasesFloat.insert(aliasesFloat.end(), {"trd_aliEnvPressure00", "trd_aliEnvPressure01", "trd_aliEnvPressure02"});
      // aliasesFloat.insert(aliasesFloat.end(), {"trd_cavernHumidity", "trd_fedEnvTemp[00..539]"});
      aliasesInt.insert(aliasesInt.end(), {"trd_runNo", "trd_runType"});
      // aliasesInt.insert(aliasesInt.end(), {"trd_fedChamberStatus[00..539]"});
      // aliasesString.insert(aliasesString.end(), {"trd_fedCFGtag[00..539]"});

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
    int verbosity = ic.options().get<bool>("processor-verbosity");
    if (verbosity > 0) {
      LOG(info) << "Using verbose mode for TRD DCS processor";
      mProcessor->setVerbosity(verbosity);
    }
    mProcessor->init(vect);
    mTimerGas = std::chrono::high_resolution_clock::now();
    mTimerVoltages = mTimerGas;
    mTimerCurrents = mTimerGas;
    mTimerEnv = mTimerGas;
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto currentTimeStamp = DataRefUtils::getHeader<DataProcessingHeader*>(pc.inputs().getFirstValid(true))->creation;
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
    if (elapsedTimeCurrents.count() * 1e-9 >= mVoltagesDPsUpdateInterval) {
      sendDPsoutputCurrents(pc.outputs());
      mTimerCurrents = timeNow;
    }

    auto elapsedTimeEnv = timeNow - mTimerEnv; // in ns
    if (elapsedTimeEnv.count() * 1e-9 >= mEnvDPsUpdateInterval) {
      sendDPsoutputEnv(pc.outputs());
      mTimerEnv = timeNow;
    }

    if (mProcessor->shouldUpdateRun()) {
      sendDPsoutputRun(pc.outputs());
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    sendDPsoutputGas(ec.outputs());
    sendDPsoutputVoltages(ec.outputs());
    sendDPsoutputCurrents(ec.outputs());
    sendDPsoutputEnv(ec.outputs());
    sendDPsoutputRun(ec.outputs());
  }

 private:
  std::unique_ptr<DCSProcessor> mProcessor;
  std::chrono::high_resolution_clock::time_point mTimerGas;
  std::chrono::high_resolution_clock::time_point mTimerVoltages;
  std::chrono::high_resolution_clock::time_point mTimerCurrents;
  std::chrono::high_resolution_clock::time_point mTimerEnv;
  int64_t mGasDPsUpdateInterval;
  int64_t mVoltagesDPsUpdateInterval;
  int64_t mCurrentsDPsUpdateInterval;
  int64_t mMinUpdateIntervalU;
  int64_t mEnvDPsUpdateInterval;

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

  return DataProcessorSpec{
    "trd-dcs-data-processor",
    Inputs{{"input", "DCS", "TRDDATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::trd::TRDDCSDataProcessor>()},
    Options{{"ccdb-path", VariantType::String, "http://localhost:8080", {"Path to CCDB"}},
            {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
            {"processor-verbosity", VariantType::Int, 0, {"Increase for more verbose output (max 3)"}},
            {"DPs-update-interval-currents", VariantType::Int64, 60ll, {"Interval (in s) after which to update the DPs CCDB entry for current parameters"}},
            {"DPs-update-interval-voltages", VariantType::Int64, 600ll, {"Interval (in s) after which to update the DPs CCDB entry for voltage parameters"}},
            {"DPs-update-interval-env", VariantType::Int64, 1800ll, {"Interval (in s) after which to update the DPs CCDB entry for environment parameters"}},
            {"DPs-min-update-interval-voltages", VariantType::Int64, 120ll, {"Minimum range to be covered by voltage CCDB object"}},
            {"DPs-update-interval-gas", VariantType::Int64, 900ll, {"Interval (in s) after which to update the DPs CCDB entry for gas parameters"}}}};
}

} // namespace framework
} // namespace o2

#endif
