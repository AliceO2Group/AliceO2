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

#ifndef O2_TOF_DATAPROCESSOR_H
#define O2_TOF_DATAPROCESSOR_H

/// @file   DCSTOFDataProcessorSpec.h
/// @brief  TOF Processor for DCS Data Points

#include <unistd.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/AliasExpander.h"
#include "TOFCalibration/TOFDCSProcessor.h"
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
namespace tof
{

using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;
using namespace o2::ccdb;
using CcdbManager = o2::ccdb::BasicCCDBManager;
using clbUtils = o2::calibration::Utils;
using HighResClock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;

class TOFDCSDataProcessor : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {

    std::vector<DPID> vect;
    mDPsUpdateInterval = ic.options().get<int64_t>("DPs-update-interval");
    if (mDPsUpdateInterval == 0) {
      LOG(error) << "TOF DPs update interval set to zero seconds --> changed to 60";
      mDPsUpdateInterval = 60;
    }
    bool useCCDBtoConfigure = ic.options().get<bool>("use-ccdb-to-configure");
    if (useCCDBtoConfigure) {
      LOG(info) << "Configuring via CCDB";
      std::string ccdbpath = ic.options().get<std::string>("ccdb-path");
      auto& mgr = CcdbManager::instance();
      mgr.setURL(ccdbpath);
      long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      std::unordered_map<DPID, std::string>* dpid2DataDesc = mgr.getForTimeStamp<std::unordered_map<DPID, std::string>>("TOF/Config/DCSDPconfig", ts);
      for (auto& i : *dpid2DataDesc) {
        vect.push_back(i.first);
      }
    } else {
      LOG(info) << "Configuring via hardcoded strings";
      std::vector<std::string> aliases = {"tof_hv_vp_[00..89]", "tof_hv_vn_[00..89]", "tof_hv_ip_[00..89]", "tof_hv_in_[00..89]"};
      std::vector<std::string> aliasesInt = {"TOF_FEACSTATUS_[00..71]"};
      std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);
      std::vector<std::string> expaliasesInt = o2::dcs::expandAliases(aliasesInt);
      for (const auto& i : expaliases) {
        vect.emplace_back(i, o2::dcs::DPVAL_DOUBLE);
      }
      for (const auto& i : expaliasesInt) {
        vect.emplace_back(i, o2::dcs::DPVAL_INT);
      }
    }

    LOG(info) << "Listing Data Points for TOF:";
    for (auto& i : vect) {
      LOG(info) << i;
    }

    mProcessor = std::make_unique<o2::tof::TOFDCSProcessor>();
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
    LOG(debug) << "startValidity = " << startValidity;
    auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");
    auto timeNow = HighResClock::now();
    if (startValidity == 0xffffffffffffffff) {                                                                   // it means it is not set
      startValidity = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms
    }
    mProcessor->setStartValidity(startValidity);
    mProcessor->process(dps);
    Duration elapsedTime = timeNow - mTimer; // in seconds
    if (elapsedTime.count() >= mDPsUpdateInterval) {
      if (mProcessor->areAllDPsFilled()) {
        sendDPsoutput(pc.outputs());
        mTimer = timeNow;
      } else {
        LOG(debug) << "Not sending yet, not all DPs were filled";
      }
    }
    sendLVandHVoutput(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    if (!mProcessor->areAllDPsFilled()) {
      LOG(debug) << "Not all DPs are filled, sending to CCDB what we have anyway";
    }
    sendDPsoutput(ec.outputs());
    sendLVandHVoutput(ec.outputs());
  }

 private:
  std::unique_ptr<TOFDCSProcessor> mProcessor;
  HighResClock::time_point mTimer;
  int64_t mDPsUpdateInterval;

  //________________________________________________________________
  void sendDPsoutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration object for DPs
    mProcessor->updateDPsCCDB();
    const auto& payload = mProcessor->getTOFDPsInfo();
    auto& info = mProcessor->getccdbDPsInfo();
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_DCSDPs", 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_DCSDPs", 0}, info);
    mProcessor->clearDPsinfo();
  }

  //________________________________________________________________
  void sendLVandHVoutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects for LV and HV, convert it to TMemFile and send them to the output

    if (mProcessor->isLVUpdated()) {
      const auto& payload = mProcessor->getLVStatus();
      auto& info = mProcessor->getccdbLVInfo();
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
      LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
                << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_LVStatus", 0}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_LVStatus", 0}, info);
    }
    if (mProcessor->isHVUpdated()) {
      const auto& payload = mProcessor->getHVStatus();
      auto& info = mProcessor->getccdbHVInfo();
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
      LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
                << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_HVStatus", 0}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_HVStatus", 0}, info);
    }
  }

}; // end class
} // namespace tof

namespace framework
{

DataProcessorSpec getTOFDCSDataProcessorSpec()
{

  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_LVStatus"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_LVStatus"}, Lifetime::Sporadic);

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_HVStatus"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_HVStatus"}, Lifetime::Sporadic);

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TOF_DCSDPs"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TOF_DCSDPs"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tof-dcs-data-processor",
    Inputs{{"input", "DCS", "TOFDATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::tof::TOFDCSDataProcessor>()},
    Options{{"ccdb-path", VariantType::String, "http://localhost:8080", {"Path to CCDB"}},
            {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
            {"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}},
            {"DPs-update-interval", VariantType::Int64, 600ll, {"Interval (in s) after which to update the DPs CCDB entry"}}}};
}

} // namespace framework
} // namespace o2

#endif
