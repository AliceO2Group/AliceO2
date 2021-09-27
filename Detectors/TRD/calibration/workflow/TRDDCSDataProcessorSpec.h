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
    bool useCCDBtoConfigure = ic.options().get<bool>("use-ccdb-to-configure");
    if (useCCDBtoConfigure) {
      LOG(info) << "Configuring via CCDB";
      std::string ccdbpath = ic.options().get<std::string>("ccdb-path");
      auto& mgr = o2::ccdb::BasicCCDBManager::instance();
      mgr.setURL(ccdbpath);
      o2::ccdb::CcdbApi api;
      api.init(mgr.getURL());
      long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      std::unordered_map<DPID, std::string>* dpid2DataDesc = mgr.getForTimeStamp<std::unordered_map<DPID, std::string>>("TRD/Config/DCSDPconfig", ts);
      for (auto& i : *dpid2DataDesc) {
        vect.push_back(i.first);
      }
    } else {
      LOG(info) << "Configuring via hardcoded strings";
      std::vector<std::string> aliasesFloat;
      aliasesFloat.insert(aliasesFloat.end(), {"trd_gasCO2", "trd_gasH2O", "trd_gasO2"});
      aliasesFloat.insert(aliasesFloat.end(), {"trd_gaschromatographCO2", "trd_gaschromatographN2", "trd_gaschromatographXe"});
      // aliasesFloat.insert(aliasesFloat.end(), {"trd_hvAnodeImon[00..539]", "trd_hvAnodeUmon[00..539]", "trd_hvDriftImon[00..539]", "trd_hvDriftImon[00..539]"});
      // std::vector<std::string> aliasesInt = {"trd_fedChamberStatus[00..539]", "trd_runNo", "trd_runType"};
      std::vector<std::string> aliasesInt = {"trd_runNo", "trd_runType"};
      std::vector<std::string> expAliasesFloat = o2::dcs::expandAliases(aliasesFloat);
      std::vector<std::string> expAliasesInt = o2::dcs::expandAliases(aliasesInt);
      for (const auto& i : expAliasesFloat) {
        vect.emplace_back(i, o2::dcs::RAW_DOUBLE);
      }
      for (const auto& i : expAliasesInt) {
        vect.emplace_back(i, o2::dcs::RAW_INT);
      }
    }

    LOG(info) << "Listing Data Points for TRD:";
    for (auto& i : vect) {
      LOG(info) << i;
    }

    mProcessor = std::make_unique<o2::trd::DCSProcessor>();
    bool useVerboseMode = ic.options().get<bool>("use-verbose-mode");
    if (useVerboseMode) {
      LOG(info) << "Using verbose mode for TRD DCS processor";
      mProcessor->useVerboseMode();
    }
    mProcessor->init(vect);
    mTimer = std::chrono::high_resolution_clock::now();
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto startValidity = DataRefUtils::getHeader<DataProcessingHeader*>(pc.inputs().getFirstValid(true))->creation;
    auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");
    auto timeNow = std::chrono::high_resolution_clock::now();
    if (startValidity == 0xffffffffffffffff) {
      // in case it is not set
      startValidity = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms
    }
    mProcessor->setStartValidity(startValidity);
    mProcessor->process(dps);

    auto elapsedTime = timeNow - mTimer; // in ns
    if (elapsedTime.count() * 1e-9 >= mGasDPsUpdateInterval) {
      sendDPsoutputGas(pc.outputs());
      mTimer = timeNow;
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    sendDPsoutputGas(ec.outputs());
  }

 private:
  std::unique_ptr<DCSProcessor> mProcessor;
  std::chrono::high_resolution_clock::time_point mTimer;
  int64_t mGasDPsUpdateInterval;

  //________________________________________________________________
  void sendDPsoutputGas(DataAllocator& output)
  {
    // extract CCDB infos and calibration object for DPs
    mProcessor->updateDPsCCDB();
    const auto& payload = mProcessor->getTRDDPsInfo();
    auto& info = mProcessor->getccdbDPsInfo();
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TRD_DCSGasDPs", 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TRD_DCSGasDPs", 0}, info);
    mProcessor->clearDPsinfo();
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

  return DataProcessorSpec{
    "trd-dcs-data-processor",
    Inputs{{"input", "DCS", "TRDDATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::trd::TRDDCSDataProcessor>()},
    Options{{"ccdb-path", VariantType::String, "http://localhost:8080", {"Path to CCDB"}},
            {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
            {"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}},
            {"DPs-update-interval-gas", VariantType::Int64, 900ll, {"Interval (in s) after which to update the DPs CCDB entry for gas parameters"}}}};
}

} // namespace framework
} // namespace o2

#endif
