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

#ifndef O2_FT0_DATAPROCESSOR_H
#define O2_FT0_DATAPROCESSOR_H

/// @file   DCSFT0DataProcessorSpec.h
/// @brief  FT0 Processor for DCS Data Points

#include <unistd.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/AliasExpander.h"
#include "FT0Calibration/FT0DCSProcessor.h"
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
namespace ft0
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;
using namespace o2::ccdb;
using CcdbManager = o2::ccdb::BasicCCDBManager;
using clbUtils = o2::calibration::Utils;
using HighResClock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;

class FT0DCSDataProcessor : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
    std::vector<DPID> vect;
    mDPsUpdateInterval = ic.options().get<int64_t>("DPs-update-interval");
    if (mDPsUpdateInterval == 0) {
      LOG(error) << "FT0 DPs update interval set to zero seconds --> changed to 10 min";
      mDPsUpdateInterval = 600;
    }
    bool useCCDBtoConfigure = ic.options().get<bool>("use-ccdb-to-configure");
    if (useCCDBtoConfigure) {
      LOG(info) << "Configuring via CCDB";
      std::string ccdbpath = ic.options().get<std::string>("ccdb-path");
      auto& mgr = CcdbManager::instance();
      mgr.setURL(ccdbpath);
      long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      std::unordered_map<DPID, std::string>* dpid2DataDesc = mgr.getForTimeStamp<std::unordered_map<DPID, std::string>>("FT0/Config/DCSDPconfig", ts);
      for (auto& i : *dpid2DataDesc) {
        vect.push_back(i.first);
      }
    } else {
      LOG(info) << "Configuring via hardcoded strings";
      std::vector<std::string> aliasesHV = {"FT0/HV/FT0_A/MCP_A[1..5]/actual/iMon",
                                            "FT0/HV/FT0_A/MCP_B[1..5]/actual/iMon",
                                            "FT0/HV/FT0_A/MCP_C[1..2]/actual/iMon",
                                            "FT0/HV/FT0_A/MCP_C[4..5]/actual/iMon",
                                            "FT0/HV/FT0_A/MCP_D[1..5]/actual/iMon",
                                            "FT0/HV/FT0_A/MCP_E[1..5]/actual/iMon",
                                            "FT0/HV/FT0_C/MCP_A[2..5]/actual/iMon",
                                            "FT0/HV/FT0_C/MCP_B[1..6]/actual/iMon",
                                            "FT0/HV/FT0_C/MCP_C[1..2]/actual/iMon",
                                            "FT0/HV/FT0_C/MCP_C[5..6]/actual/iMon",
                                            "FT0/HV/FT0_C/MCP_D[1..2]/actual/iMon",
                                            "FT0/HV/FT0_C/MCP_D[5..6]/actual/iMon",
                                            "FT0/HV/FT0_C/MCP_E[1..6]/actual/iMon",
                                            "FT0/HV/FT0_C/MCP_F[2..5]/actual/iMon",
                                            "FT0/HV/MCP_LC/actual/iMon"};
      std::string aliasesADCZERO = "FT0/PM/channel[000..211]/actual/ADC[0..1]_BASELINE";
      std::vector<std::string> expAliasesHV = o2::dcs::expandAliases(aliasesHV);
      std::vector<std::string> expAliasesADCZERO = o2::dcs::expandAlias(aliasesADCZERO);
      for (const auto& i : expAliasesHV) {
        vect.emplace_back(i, o2::dcs::DPVAL_DOUBLE);
      }
      for (const auto& i : expAliasesADCZERO) {
        vect.emplace_back(i, o2::dcs::DPVAL_UINT);
      }
    }

    const bool useVerboseMode = ic.options().get<bool>("use-verbose-mode");
    LOG(info) << "Verbose mode: " << useVerboseMode;

    if (useVerboseMode) {
      LOG(info) << "Listing Data Points for FT0:";
      for (auto& i : vect) {
        LOG(info) << i;
      }
    }

    mProcessor = std::make_unique<o2::ft0::FT0DCSProcessor>();

    if (useVerboseMode) {
      mProcessor->useVerboseMode();
    }
    mProcessor->init(vect);
    mTimer = HighResClock::now();
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto timeNow = HighResClock::now();
    long dataTime = (long)(pc.services().get<o2::framework::TimingInfo>().creation);
    if (dataTime == 0xffffffffffffffff) {                                                                   // it means it is not set
      dataTime = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms
    }
    if (mProcessor->getStartValidity() == o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP) {
      if (mProcessor->getVerboseMode()) {
        LOG(info) << "startValidity for DPs changed to = " << dataTime;
      }
      mProcessor->setStartValidity(dataTime);
    }
    auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");
    mProcessor->process(dps);
    Duration elapsedTime = timeNow - mTimer; // in seconds
    if (elapsedTime.count() >= mDPsUpdateInterval) {
      sendDPsoutput(pc.outputs());
      mTimer = timeNow;
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    sendDPsoutput(ec.outputs());
  }

 private:
  std::unique_ptr<FT0DCSProcessor> mProcessor;
  HighResClock::time_point mTimer;
  int64_t mDPsUpdateInterval;

  void sendDPsoutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration object for DPs
    mProcessor->updateDPsCCDB();
    const auto& payload = mProcessor->getFT0DPsInfo();
    auto& info = mProcessor->getccdbDPsInfo();
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "FT0_DCSDPs", 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "FT0_DCSDPs", 0}, info);
    mProcessor->clearDPsinfo();
    mProcessor->resetStartValidity();
  }

}; // end class
} // namespace ft0

namespace framework
{

DataProcessorSpec getFT0DCSDataProcessorSpec()
{

  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "FT0_DCSDPs"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "FT0_DCSDPs"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "ft0-dcs-data-processor",
    Inputs{{"input", "DCS", "FT0DATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::ft0::FT0DCSDataProcessor>()},
    Options{{"ccdb-path", VariantType::String, "http://localhost:8080", {"Path to CCDB"}},
            {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
            {"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}},
            {"DPs-update-interval", VariantType::Int64, 600ll, {"Interval (in s) after which to update the DPs CCDB entry"}}}};
}

} // namespace framework
} // namespace o2

#endif
