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
    mFatalOnFailure = ic.options().get<bool>("no-fatal-on-failure");
    mAPI.init(mCCDBpath);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    int nSlots = pc.inputs().getNofParts(0);
    assert(pc.inputs().getNofParts(1) == nSlots);
    auto runNoFromDH = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getFirstValid(true))->runNumber;
    std::string runNoStr;
    if (runNoFromDH > 0) {
      runNoStr = std::to_string(runNoFromDH);
    }
    std::map<std::string, std::string> metadata;
    for (int isl = 0; isl < nSlots; isl++) {
      auto refWrp = pc.inputs().get("clbWrapper", isl);
      auto refPld = pc.inputs().get("clbPayload", isl);
      if (mSSpecMin >= 0 && mSSpecMin <= mSSpecMax) { // there is a selection
        auto ss = std::int64_t(o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(refWrp)->subSpecification);
        if (ss < mSSpecMin || ss > mSSpecMax) {
          continue;
        }
      }
      const auto wrp = pc.inputs().get<CcdbObjectInfo*>(refWrp);
      const auto pld = pc.inputs().get<gsl::span<char>>(refPld); // this is actually an image of TMemFile
      const auto* md = &wrp->getMetaData();
      if (runNoFromDH > 0 && md->find(o2::base::NameConf::CCDBRunTag.data()) == md->end()) { // if valid run number is provided and it is not filled in the metadata, add it to the clone
        metadata = *md;                                                                      // clone since the md from the message is const
        metadata[o2::base::NameConf::CCDBRunTag.data()] = runNoStr;
        md = &metadata;
      }

      LOG(info) << "Storing in ccdb " << wrp->getPath() << "/" << wrp->getFileName() << " of size " << pld.size()
                << " Valid for " << wrp->getStartValidityTimestamp() << " : " << wrp->getEndValidityTimestamp();
      int res = mAPI.storeAsBinaryFile(&pld[0], pld.size(), wrp->getFileName(), wrp->getObjectType(), wrp->getPath(),
                                       *md, wrp->getStartValidityTimestamp(), wrp->getEndValidityTimestamp());
      if (res && mFatalOnFailure) {
        LOGP(fatal, "failed on uploading to {} / {}", mAPI.getURL(), wrp->getPath());
      }

      // do we need to override previous object?
      if (wrp->isAdjustableEOV()) {
        o2::ccdb::adjustOverriddenEOV(mAPI, *wrp.get());
      }
    }
  }

 private:
  CcdbApi mAPI;
  bool mFatalOnFailure = true;                             // produce fatal on failed upload
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
      {"no-fatal-on-failure", VariantType::Bool, false, {"do not produce fatal on failed upload"}}}};
}

} // namespace framework
} // namespace o2

#endif
