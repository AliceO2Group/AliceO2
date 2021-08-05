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

#ifndef O2_MFT_DATAPROCESSOR_H
#define O2_MFT_DATAPROCESSOR_H

/// @file   DCSMFTDataProcessorSpec.h
/// @brief  MFT Processor for DCS Data Points

#include <unistd.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/AliasExpander.h"
#include "MFTCondition/MFTDCSProcessor.h"
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
namespace mft
{

  using DPID = o2::dcs::DataPointIdentifier;
  using DPVAL = o2::dcs::DataPointValue;
  using DPCOM = o2::dcs::DataPointCompositeObject;
  using namespace o2::ccdb;
  using CcdbManager = o2::ccdb::BasicCCDBManager;
  using clbUtils = o2::calibration::Utils;
  using HighResClock = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double, std::ratio<1, 1>>;

  class MFTDCSDataProcessor : public o2::framework::Task
  {
  public:

    //________________________________________________________________
    void init(o2::framework::InitContext& ic) final
    {
      std::vector<DPID> vect;
      mDPsUpdateInterval = ic.options().get<int64_t>("DPs-update-interval");
      if (mDPsUpdateInterval == 0) {
        LOG(ERROR) << "MFT DPs update interval set to zero seconds --> changed to 60";
        mDPsUpdateInterval = 60;
      }
      bool useCCDBtoConfigure = ic.options().get<bool>("use-ccdb-to-configure");

      mStart = ic.options().get<int64_t>("tstart");
      mEnd = ic.options().get<int64_t>("tend");

      if (useCCDBtoConfigure) {
  LOG(INFO) << "Configuring via CCDB";
  std::string ccdbpath = ic.options().get<std::string>("ccdb-path");
  auto& mgr = CcdbManager::instance();
  mgr.setURL(ccdbpath);
  CcdbApi api;
  api.init(mgr.getURL());
  long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  std::unordered_map<DPID, std::string>* dpid2DataDesc = mgr.getForTimeStamp<std::unordered_map<DPID, std::string>>("TOF/DCSconfig", ts);
  for (auto& i : *dpid2DataDesc) {
    vect.push_back(i.first);
  }
      }

      else {
        LOG(INFO) << "Configuring via hardcoded strings";
  std::vector<std::string> aliases = {"mft_main:MFT_PSU_Zone/H[0..1]D[0..4]F[0..1]Z[0..3].Monitoirng.Current.Analog",
                                            "mft_main:MFT_PSU_Zone/H[0..1]D[0..4]F[0..1]Z[0..3].Monitoirng.Current.Digital",
                                            "mft_main:MFT_PSU_Zone/H[0..1]D[0..4]F[0..1]Z[0..3].Monitoirng.Current.BackBias"};
  std::vector<std::string> expaliases = o2::dcs::expandAliases(aliases);
  for (const auto& i : expaliases) {
    vect.emplace_back(i, o2::dcs::RAW_DOUBLE);
  }
      }

      LOG(INFO) << "Listing Data Points for MFT:";
      for (auto& i : vect) {
  LOG(INFO) << i;
      }

      mProcessor = std::make_unique<o2::mft::MFTDCSProcessor>();
      bool useVerboseMode = ic.options().get<bool>("use-verbose-mode");
      LOG(INFO) << " ************************* Verbose?" << useVerboseMode;

      if (useVerboseMode) {
  mProcessor->useVerboseMode();
      }
      mProcessor->init(vect);

      mTimer = HighResClock::now();

    }

    //________________________________________________________________
    void run(o2::framework::ProcessingContext& pc) final
    {
      auto tfid = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
      auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");

      mProcessor->setTF(tfid);
      mProcessor->process(dps);

      auto timeNow = HighResClock::now();
      Duration elapsedTime = timeNow - mTimer; // in seconds

      LOG(INFO) << "mDPsUpdateInterval " <<mDPsUpdateInterval << "[sec.]";

      if (elapsedTime.count() >= mDPsUpdateInterval) {
  sendDPsoutput(pc.outputs());
  mTimer = timeNow;
      }
    }

    //________________________________________________________________
    void endOfStream(o2::framework::EndOfStreamContext& ec) final
    {
      sendDPsoutput(ec.outputs());
    }

  private:
    std::unique_ptr<MFTDCSProcessor> mProcessor;
    HighResClock::time_point mTimer;
    int64_t mDPsUpdateInterval;

    long mStart;
    long mEnd;

    //________________________________________________________________
    void sendDPsoutput(DataAllocator& output)
    {
      // extract CCDB infos and calibration object for DPs
      mProcessor->updateDPsCCDB();
      const auto& payload = mProcessor->getMFTDPsInfo();
      auto& info = mProcessor->getccdbDPsInfo();

      long tstart = mStart;
      long tend = mEnd;

      if (tstart == -1) {
  tstart = o2::ccdb::getCurrentTimestamp();
      }

      if (tend == -1) {
  constexpr long SECONDSPERYEAR = 365 * 24 * 60 * 60;
  tend = o2::ccdb::getFutureTimestamp(SECONDSPERYEAR);
      }

      info.setStartValidityTimestamp(tstart);
      info.setEndValidityTimestamp(tend);

      auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
      LOG(INFO) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
    << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "MFT_DCSDPs", 0}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "MFT_DCSDPs", 0}, info);
      mProcessor->clearDPsinfo();
    }
    //________________________________________________________________
  }; // end class
} // namespace mft

namespace framework
{

  DataProcessorSpec getMFTDCSDataProcessorSpec()
  {

    using clbUtils = o2::calibration::Utils;

    std::vector<OutputSpec> outputs;

    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "MFT_DCSDPs"});
    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "MFT_DCSDPs"});

    return DataProcessorSpec{
      "mft-dcs-data-processor",
  Inputs{{"input", "DCS", "MFTDATAPOINTS"}},
  outputs,
    AlgorithmSpec{adaptFromTask<o2::mft::MFTDCSDataProcessor>()},
    Options{
      {"ccdb-path", VariantType::String, "http://localhost:8080", {"Path to CCDB"}},
      {"tstart", VariantType::Int64, -1ll, {"Start of validity timestamp"}},
      {"tend", VariantType::Int64, -1ll, {"End of validity timestamp"}},
      {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
      {"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}},
      {"DPs-update-interval", VariantType::Int64, 600ll, {"Interval (in s) after which to update the DPs CCDB entry"}}}};
  }

} // namespace framework
} // namespace o2

#endif
