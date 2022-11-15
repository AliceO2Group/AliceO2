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

#ifndef O2_TRD_VDANDEXBCALIBSPEC_H
#define O2_TRD_VDANDEXBCALIBSPEC_H

/// \file   VdAndExBCalibSpec.h
/// \brief DPL device for steering the TRD vD and ExB time slot based calibration
/// \author Ole Schmidt

#include "TRDCalibration/CalibratorVdExB.h"
#include "DetectorsCalibration/Utils.h"
#include "DataFormatsTRD/AngularResidHistos.h"
#include "CommonUtils/MemFileHelper.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include <chrono>

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class VdAndExBCalibDevice : public o2::framework::Task
{
 public:
  VdAndExBCalibDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}
  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    auto enableOutput = ic.options().get<bool>("enable-root-output");
    auto slotL = ic.options().get<uint32_t>("sec-per-slot");
    auto delay = ic.options().get<uint32_t>("max-delay");
    mCalibrator = std::make_unique<o2::trd::CalibratorVdExB>(enableOutput);
    mCalibrator->setSlotLengthInSeconds(slotL);
    mCalibrator->setMaxSlotsDelay(delay);
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto runStartTime = std::chrono::high_resolution_clock::now();
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    mCalibrator->retrievePrev(pc);
    auto dataAngRes = pc.inputs().get<o2::trd::AngularResidHistos>("input");
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());
    LOG(info) << "Processing TF " << mCalibrator->getCurrentTFInfo().tfCounter << " with " << dataAngRes.getNEntries() << " AngularResidHistos entries";
    mCalibrator->process(dataAngRes);
    sendOutput(pc.outputs());
    std::chrono::duration<double, std::milli> runDuration = std::chrono::high_resolution_clock::now() - runStartTime;
    LOGP(info, "Duration for run method: {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(runDuration).count());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(info) << "Finalizing calibration";
    mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
    sendOutput(ec.outputs());
  }

 private:
  std::unique_ptr<o2::trd::CalibratorVdExB> mCalibrator;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // TODO in principle, this routine is generic, can be moved to Utils.h

    using clbUtils = o2::calibration::Utils;
    const auto& payloadVec = mCalibrator->getCcdbObjectVector();
    auto& infoVec = mCalibrator->getCcdbObjectInfoVector(); // use non-const version as we update it
    assert(payloadVec.size() == infoVec.size());

    for (uint32_t i = 0; i < payloadVec.size(); i++) {
      auto& w = infoVec[i];
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
      LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();

      output.snapshot(Output{clbUtils::gDataOriginCDBPayload, "VDRIFTEXB", i}, *image.get()); // vector<char>
      output.snapshot(Output{clbUtils::gDataOriginCDBWrapper, "VDRIFTEXB", i}, w);            // root-serialized
    }
    if (payloadVec.size()) {
      mCalibrator->initOutput(); // reset the outputs once they are already sent
    }
  }
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getTRDVdAndExBCalibSpec()
{
  using device = o2::calibration::VdAndExBCalibDevice;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "VDRIFTEXB"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "VDRIFTEXB"}, Lifetime::Sporadic);
  std::vector<InputSpec> inputs;
  inputs.emplace_back("input", "TRD", "ANGRESHISTS");
  inputs.emplace_back("calvdexb", "TRD", "CALVDRIFTEXB", 0, Lifetime::Condition, ccdbParamSpec("TRD/Calib/CalVdriftExB"));
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return DataProcessorSpec{
    "calib-vdexb-calibration",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(ccdbRequest)},
    Options{
      {"sec-per-slot", VariantType::UInt32, 900u, {"number of seconds per calibration time slot"}},
      {"max-delay", VariantType::UInt32, 2u, {"number of slots in past to consider"}},
      {"enable-root-output", VariantType::Bool, false, {"output tprofiles and fits to root file"}},
    }};
}
} // namespace framework
} // namespace o2

#endif // O2_TRD_VDANDEXBCALIBSPEC_H
