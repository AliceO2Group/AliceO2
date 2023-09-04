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

#ifndef O2_TRD_T0FITSPEC_H
#define O2_TRD_T0FITSPEC_H

/// \file   T0FitSpec.h
/// \brief  DPL device for steering the TRD t0 fits
/// \author Luisa Bergmann

#include "TRDCalibration/T0Fit.h"
#include "DetectorsCalibration/Utils.h"
#include "DataFormatsTRD/PHData.h"
#include "DataFormatsTRD/CalT0.h"
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

class T0FitDevice : public o2::framework::Task
{
 public:
  T0FitDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}
  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    auto slotL = ic.options().get<uint32_t>("sec-per-slot");
    auto delay = ic.options().get<uint32_t>("max-delay");

    mFitInstance = std::make_unique<o2::trd::T0Fit>();
    mFitInstance->setSlotLengthInSeconds(slotL);
    mFitInstance->setMaxSlotsDelay(delay);
    if (ic.options().get<bool>("enable-root-output")) {
      mFitInstance->createOutputFile();
    }
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    if (mRunStopRequested) {
      return;
    }
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);

    auto dataT0Fit = pc.inputs().get<std::vector<o2::trd::PHData>>("input");
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mFitInstance->getCurrentTFInfo());
    LOG(detail) << "Processing TF " << mFitInstance->getCurrentTFInfo().tfCounter << " with " << dataT0Fit.size() << " PHData entries";
    mFitInstance->process(dataT0Fit);

    if (pc.transitionState() == TransitionHandlingState::Requested) {
      LOG(info) << "Run stop requested, finalizing";
      mRunStopRequested = true;
      mFitInstance->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
      mFitInstance->closeOutputFile();
    }
    sendOutput(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    if (mRunStopRequested) {
      return;
    }
    mFitInstance->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
    mFitInstance->closeOutputFile();
    sendOutput(ec.outputs());
  }

  void stop() final
  {
    mFitInstance->closeOutputFile();
  }

 private:
  std::unique_ptr<o2::trd::T0Fit> mFitInstance;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  bool mRunStopRequested = false; // flag that run was stopped (and the last output is sent)
  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output

    using clbUtils = o2::calibration::Utils;
    const auto& payloadVec = mFitInstance->getCcdbObjectVector();
    auto& infoVec = mFitInstance->getCcdbObjectInfoVector(); // use non-const version as we update it

    assert(payloadVec.size() == infoVec.size());

    for (uint32_t i = 0; i < payloadVec.size(); i++) {
      auto& w = infoVec[i];
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
      LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();

      output.snapshot(Output{clbUtils::gDataOriginCDBPayload, "CALT0", i}, *image.get()); // vector<char>
      output.snapshot(Output{clbUtils::gDataOriginCDBWrapper, "CALT0", i}, w);            // root-serialized
    }
    if (payloadVec.size()) {
      mFitInstance->initOutput(); // reset the outputs once they are already sent
    }
  }
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getTRDT0FitSpec()
{
  using device = o2::calibration::T0FitDevice;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "CALT0"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "CALT0"}, Lifetime::Sporadic);
  std::vector<InputSpec> inputs;
  inputs.emplace_back("input", "TRD", "PULSEHEIGHT");
  inputs.emplace_back("calt0", "TRD", "CALT0", 0, Lifetime::Condition, ccdbParamSpec("TRD/Calib/CalT0"));
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  return DataProcessorSpec{
    "calib-t0-fit",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(ccdbRequest)},
    Options{
      {"sec-per-slot", VariantType::UInt32, 900u, {"number of seconds per calibration time slot"}},
      {"max-delay", VariantType::UInt32, 2u, {"number of slots in past to consider"}},
      {"enable-root-output", VariantType::Bool, false, {"output t0 values to root file"}},
    }};
}

} // namespace framework
} // namespace o2

#endif // O2_TRD_T0FITSPEC_H
