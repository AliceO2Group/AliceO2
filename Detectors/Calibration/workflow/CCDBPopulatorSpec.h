// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"

using CcdbManager = o2::ccdb::BasicCCDBManager;

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
    auto& mgr = CcdbManager::instance();
    mgr.setURL(mCCDBpath);
    mAPI.init(mgr.getURL());
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    int nSlots = pc.inputs().getNofParts(0);
    assert(pc.inputs().getNofParts(1) == nSlots);

    for (int isl = 0; isl < nSlots; isl++) {
      const auto wrp = pc.inputs().get<CcdbObjectInfo*>("clbWrapper", isl);
      const auto pld = pc.inputs().get<gsl::span<char>>("clbPayload", isl); // this is actually an image of TMemFile

      LOG(INFO) << "Storing in ccdb " << wrp->getPath() << "/" << wrp->getFileName() << " of size " << pld.size()
                << " Valid for " << wrp->getStartValidityTimestamp() << " : " << wrp->getEndValidityTimestamp();
      mAPI.storeAsBinaryFile(&pld[0], pld.size(), wrp->getFileName(), wrp->getObjectType(), wrp->getPath(),
                             wrp->getMetaData(), wrp->getStartValidityTimestamp(), wrp->getEndValidityTimestamp());
    }
  }

 private:
  CcdbApi mAPI;
  std::string mCCDBpath = "http://ccdb-test.cern.ch:8080"; // CCDB path
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getCCDBPopulatorDeviceSpec()
{
  using clbUtils = o2::calibration::Utils;
  std::vector<InputSpec> inputs = {{"clbPayload", "CLP"}, {"clbWrapper", "CLW"}};

  return DataProcessorSpec{
    "ccdb-populator",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<o2::calibration::CCDBPopulator>()},
    Options{
      {"ccdb-path", VariantType::String, "http://ccdb-test.cern.ch:8080", {"Path to CCDB"}}}};
}

} // namespace framework
} // namespace o2

#endif
