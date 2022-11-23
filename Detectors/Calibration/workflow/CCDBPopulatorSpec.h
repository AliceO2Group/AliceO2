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

#ifndef O2_CALIBRATION_CCDBPOPULATOR_H
#define O2_CALIBRATION_CCDBPOPULATOR_H

/// @file   CCDBPopulator.h
/// @brief  device to populate CCDB

#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DataRefUtils.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Headers/DataHeader.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CommonUtils/NameConf.h"
#include <unordered_map>
#include <chrono>

namespace o2
{
namespace calibration
{

class CCDBPopulator : public o2::framework::Task
{
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using CcdbApi = o2::ccdb::CcdbApi;

 public:
  void init(o2::framework::InitContext& ic) final
  {
    mCCDBpath = ic.options().get<std::string>("ccdb-path");
    mSSpecMin = ic.options().get<std::int64_t>("sspec-min");
    mSSpecMax = ic.options().get<std::int64_t>("sspec-max");
    mFatalOnFailure = ic.options().get<bool>("fatal-on-failure");
    mThrottlingDelayMS = ic.options().get<std::int64_t>("throttling-delay");
    mAPI.init(mCCDBpath);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    int nSlots = pc.inputs().getNofParts(0);
    if (nSlots != pc.inputs().getNofParts(1)) {
      LOGP(alarm, "Number of slots={} in part0 is different from that ({}) in part1", nSlots, pc.inputs().getNofParts(1));
      return;
    } else if (nSlots == 0) {
      LOG(alarm) << "0 slots received";
      return;
    }
    auto runNoFromDH = pc.services().get<o2::framework::TimingInfo>().runNumber;
    std::string runNoStr;
    if (runNoFromDH > 0) {
      runNoStr = std::to_string(runNoFromDH);
    }
    auto nowMS = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::map<std::string, std::string> metadata;
    for (int isl = 0; isl < nSlots; isl++) {
      auto refWrp = pc.inputs().get("clbWrapper", isl);
      auto refPld = pc.inputs().get("clbPayload", isl);
      if (!o2::framework::DataRefUtils::isValid(refWrp)) {
        LOGP(alarm, "Wrapper is not valid for slot {}", isl);
        continue;
      }
      if (!o2::framework::DataRefUtils::isValid(refPld)) {
        LOGP(alarm, "Payload is not valid for slot {}", isl);
        continue;
      }
      if (mSSpecMin >= 0 && mSSpecMin <= mSSpecMax) { // there is a selection
        auto ss = std::int64_t(o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(refWrp)->subSpecification);
        if (ss < mSSpecMin || ss > mSSpecMax) {
          continue;
        }
      }
      const auto wrp = pc.inputs().get<CcdbObjectInfo*>(refWrp);
      const auto pld = pc.inputs().get<gsl::span<char>>(refPld); // this is actually an image of TMemFile
      if (!wrp) {
        LOGP(alarm, "No CcdbObjectInfo info for {} at slot {}",
             o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(refWrp)->dataDescription.as<std::string>(), isl);
        continue;
      }
      const auto* md = &wrp->getMetaData();
      if (runNoFromDH > 0 && md->find(o2::base::NameConf::CCDBRunTag.data()) == md->end()) { // if valid run number is provided and it is not filled in the metadata, add it to the clone
        metadata = *md;                                                                      // clone since the md from the message is const
        metadata[o2::base::NameConf::CCDBRunTag.data()] = runNoStr;
        md = &metadata;
      }
      std::string msg = fmt::format("Storing in ccdb {}/{} of size {} valid for {} : {}", wrp->getPath(), wrp->getFileName(), pld.size(), wrp->getStartValidityTimestamp(), wrp->getEndValidityTimestamp());
      auto& lastLog = mThrottling[wrp->getPath()];
      if (lastLog.first + mThrottlingDelayMS < nowMS) {
        if (lastLog.second) {
          msg += fmt::format(" ({} uploads were logged as INFO)", lastLog.second);
          lastLog.second = 0;
        }
        lastLog.first = nowMS;
        LOG(important) << msg;
      } else {
        lastLog.second++;
        LOG(info) << msg;
      }
      int res = mAPI.storeAsBinaryFile(&pld[0], pld.size(), wrp->getFileName(), wrp->getObjectType(), wrp->getPath(),
                                       *md, wrp->getStartValidityTimestamp(), wrp->getEndValidityTimestamp());
      if (res && mFatalOnFailure) {
        LOGP(fatal, "failed on uploading to {} / {}", mAPI.getURL(), wrp->getPath());
      }

      // do we need to override previous object?
      if (wrp->isAdjustableEOV() && !mAPI.isSnapshotMode()) {
        o2::ccdb::adjustOverriddenEOV(mAPI, *wrp.get());
      }
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(info) << "EndOfStream received";
  }

 private:
  CcdbApi mAPI;
  long mThrottlingDelayMS = 0;                             // LOG(important) at most once per this period for given path
  bool mFatalOnFailure = true;                             // produce fatal on failed upload
  std::unordered_map<std::string, std::pair<long, int>> mThrottling;
  std::int64_t mSSpecMin = -1;                             // min subspec to accept
  std::int64_t mSSpecMax = -1;                             // max subspec to accept
  std::string mCCDBpath = "http://ccdb-test.cern.ch:8080"; // CCDB path
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getCCDBPopulatorDeviceSpec(const std::string& defCCDB, const std::string& nameExt)
{
  using clbUtils = o2::calibration::Utils;
  std::vector<InputSpec> inputs = {{"clbPayload", "CLP"}, {"clbWrapper", "CLW"}};
  std::string devName = "ccdb-populator";
  devName += nameExt;
  return DataProcessorSpec{
    devName,
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<o2::calibration::CCDBPopulator>()},
    Options{
      {"ccdb-path", VariantType::String, defCCDB, {"Path to CCDB"}},
      {"sspec-min", VariantType::Int64, -1L, {"min subspec to accept"}},
      {"sspec-max", VariantType::Int64, -1L, {"max subspec to accept"}},
      {"throttling-delay", VariantType::Int64, 300000L, {"produce important type log at most once per this period in ms for each CCDB path"}},
      {"fatal-on-failure", VariantType::Bool, false, {"do not produce fatal on failed upload"}}}};
}

} // namespace framework
} // namespace o2

#endif
