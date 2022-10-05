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

#ifndef O2_ZDC_DATAPROCESSOR_H
#define O2_ZDC_DATAPROCESSOR_H

/// @file   DCSZDCDataProcessorSpec.h
/// @brief  ZDC Processor for DCS Data Points

#include <unistd.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/AliasExpander.h"
#include "ZDCCalib/ZDCDCSProcessor.h"
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
namespace zdc
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;
using namespace o2::ccdb;
using CcdbManager = o2::ccdb::BasicCCDBManager;
using clbUtils = o2::calibration::Utils;
using HighResClock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;

class ZDCDCSDataProcessor : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {

    std::vector<DPID> vect;
    mDPsUpdateInterval = ic.options().get<int64_t>("DPs-update-interval");
    if (mDPsUpdateInterval == 0) {
      LOG(error) << "ZDC DPs update interval was set to 0 seconds --> changed to 240";
      mDPsUpdateInterval = 240;
    }
    bool useCCDBtoConfigure = ic.options().get<bool>("use-ccdb-to-configure");
    if (useCCDBtoConfigure) {
      LOG(info) << "Configuring via CCDB";
      std::string ccdbpath = ic.options().get<std::string>("ccdb-path");
      auto& mgr = CcdbManager::instance();
      mgr.setURL(ccdbpath);
      CcdbApi api;
      api.init(mgr.getURL());
      long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      std::unordered_map<DPID, std::string>* dpid2Data = mgr.getForTimeStamp<std::unordered_map<DPID, std::string>>("ZDC/Calib/DCSconfig", ts);
      for (auto& i : *dpid2Data) {
        vect.push_back(i.first);
      }
    } else {
      LOG(info) << "Configuring via hardcoded strings";
      std::vector<std::string> aliases = {"ZDC_ZNA_POS.actual.position",
                                          "ZDC_ZPA_POS.actual.position",
                                          "ZDC_ZNC_POS.actual.position",
                                          "ZDC_ZPC_POS.actual.position",
                                          "ZDC_ZNA_HV0.actual.vMon",
                                          "ZDC_ZNA_HV1.actual.vMon",
                                          "ZDC_ZNA_HV2.actual.vMon",
                                          "ZDC_ZNA_HV3.actual.vMon",
                                          "ZDC_ZNA_HV4.actual.vMon",
                                          "ZDC_ZPA_HV0.actual.vMon",
                                          "ZDC_ZPA_HV1.actual.vMon",
                                          "ZDC_ZPA_HV2.actual.vMon",
                                          "ZDC_ZPA_HV3.actual.vMon",
                                          "ZDC_ZPA_HV4.actual.vMon",
                                          "ZDC_ZNC_HV0.actual.vMon",
                                          "ZDC_ZNC_HV1.actual.vMon",
                                          "ZDC_ZNC_HV2.actual.vMon",
                                          "ZDC_ZNC_HV3.actual.vMon",
                                          "ZDC_ZNC_HV4.actual.vMon",
                                          "ZDC_ZPC_HV0.actual.vMon",
                                          "ZDC_ZPC_HV1.actual.vMon",
                                          "ZDC_ZPC_HV2.actual.vMon",
                                          "ZDC_ZPC_HV3.actual.vMon",
                                          "ZDC_ZPC_HV4.actual.vMon",
                                          "ZDC_ZEM_HV0.actual.vMon",
                                          "ZDC_ZEM_HV1.actual.vMon",
                                          "ZDC_ZNA_HV0_D[1..2]",
                                          "ZDC_ZNC_HV0_D[1..2]",
                                          "ZDC_ZPA_HV0_D[1..2]",
                                          "ZDC_ZPC_HV0_D[1..2]"};
      std::vector<std::string> aliasesInt = {"ZDC_CONFIG_[00..32]"};
      std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);
      std::vector<std::string> expaliasesInt = o2::dcs::expandAliases(aliasesInt);
      for (const auto& i : expaliases) {
        vect.emplace_back(i, o2::dcs::RAW_DOUBLE);
      }
    }

    LOG(info) << "Listing Data Points for ZDC:";
    for (auto& i : vect) {
      LOG(info) << i;
    }

    mProcessor = std::make_unique<o2::zdc::ZDCDCSProcessor>();
    bool useVerboseMode = ic.options().get<bool>("use-verbose-mode");
    LOG(info) << " ************************* Verbose?" << useVerboseMode;
    if (useVerboseMode) {
      mProcessor->useVerboseMode();
    }
    mProcessor->init(vect);
    mTimer = HighResClock::now();
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto startValidity = DataRefUtils::getHeader<DataProcessingHeader*>(pc.inputs().getFirstValid(true))->creation;
    auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");
    auto timeNow = HighResClock::now();
    if (startValidity == 0xffffffffffffffff) {                                                                   // it means it is not set
      startValidity = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms
    }
    mProcessor->setStartValidity(startValidity);
    mProcessor->process(dps);
    Duration elapsedTime = timeNow - mTimer; // in seconds
    // LOG(info) << "mDPsUpdateInterval " << mDPsUpdateInterval << "[sec.]";

    if (elapsedTime.count() >= mDPsUpdateInterval) {
      sendDPsoutput(pc.outputs());
      mTimer = timeNow;
    }
    // sendLVandHVoutput(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    sendDPsoutput(ec.outputs());
    // sendLVandHVoutput(ec.outputs());
  }

 private:
  std::unique_ptr<ZDCDCSProcessor> mProcessor;
  HighResClock::time_point mTimer;
  int64_t mDPsUpdateInterval;

  //________________________________________________________________
  void sendDPsoutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration object for DPs
    mProcessor->updateDPsCCDB();
    const auto& payload = mProcessor->getZDCDPsInfo();
    auto& info = mProcessor->getccdbDPsInfo();
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ZDC_DCSDPs", 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDC_DCSDPs", 0}, info);
    mProcessor->clearDPsinfo();
  }

  //________________________________________________________________
  void sendLVandHVoutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects for LV and HV and send them to the output

    if (mProcessor->isHVUpdated()) {
      const auto& payload = mProcessor->getHVStatus();
      auto& info = mProcessor->getccdbHVInfo();
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
      LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
                << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ZDC_DCSDPs", 0}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDC_DCSDPs", 0}, info);
    }
  }

}; // end class
} // namespace zdc

namespace framework
{

DataProcessorSpec getZDCDCSDataProcessorSpec()
{

  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ZDC_DCSDPs"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDC_DCSDPs"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "zdc-dcs-data-processor",
    Inputs{{"input", "DCS", "ZDCDATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::zdc::ZDCDCSDataProcessor>()},
    Options{{"ccdb-path", VariantType::String, "http://localhost:8080", {"Path to CCDB"}},
            {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
            {"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}},
            {"DPs-update-interval", VariantType::Int64, 600ll, {"Interval (in s) after which to update the DPs CCDB entry"}}}};
}

} // namespace framework
} // namespace o2

#endif
