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

#ifndef O2_CALIBRATION_FT0CALIB_SLEWING_COLLECTOR_H
#define O2_CALIBRATION_FT0CALIB_SLEWING_COLLECTOR_H

/// @file   FT0CalibSlewingCollectorSpec.h
/// @brief  Device to collect information for FT0 time slewing calibration.

//#include "FT0Calibration/FT0CollectCalibInfo.h"
#include "FT0Calibration/FT0CalibCollector.h"
#include "DetectorsCalibration/Utils.h"
#include "DataFormatsFT0/FT0CalibrationInfoObject.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include <fairlogger/Logger.h>
#include "DetectorsBase/GRPGeomHelper.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{

class FT0CalibCollectorDevice : public o2::framework::Task
{

 public:
  FT0CalibCollectorDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}
  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    bool isTFsendingPolicy = ic.options().get<bool>("tf-sending-policy");
    int maxEnt = ic.options().get<int>("max-number-hits-to-fill-tree");
    auto slotL = ic.options().get<uint32_t>("tf-per-slot");
    bool absMaxEnt = ic.options().get<bool>("is-max-number-hits-to-fill-tree-absolute");
    mCollector = std::make_unique<o2::ft0::FT0CalibCollector>(isTFsendingPolicy, maxEnt);
    mCollector->setSlotLength(slotL);
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCollector->getCurrentTFInfo());
    auto data = pc.inputs().get<gsl::span<o2::ft0::FT0CalibrationInfoObject>>("input");
    mCollector->process(data);
    sendOutput(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    mCollector->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
    // we force finalizing slot zero (unless everything was already finalized), no matter how many entries we had
    if (mCollector->getNSlots() != 0) {
      mCollector->finalizeSlot(mCollector->getSlot(0));
    }
    sendOutput(ec.outputs());
  }

 private:
  std::unique_ptr<o2::ft0::FT0CalibCollector> mCollector;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  int mMaxNumOfHits = 0;

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    // in output we send the calibration tree
    auto& collectedInfo = mCollector->getCollectedCalibInfo();
    LOG(debug) << "In CollectorSpec sendOutput: size = " << collectedInfo.size();
    if (collectedInfo.size()) {
      auto entries = collectedInfo.size();
      // this means that we are ready to send the output
      auto entriesPerChannel = mCollector->getEntriesPerChannel();
      output.snapshot(Output{o2::header::gDataOriginFT0, "COLLECTEDINFO", 0, Lifetime::Timeframe}, collectedInfo);
      output.snapshot(Output{o2::header::gDataOriginFT0, "ENTRIESCH", 0, Lifetime::Timeframe}, entriesPerChannel);
      mCollector->initOutput(); // reset the output for the next round
    }
  }
};

} // namespace ft0

namespace framework
{

DataProcessorSpec getFT0CalibCollectorDeviceSpec()
{
  using device = o2::ft0::FT0CalibCollectorDevice;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginFT0, "COLLECTEDINFO", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginFT0, "ENTRIESCH", 0, Lifetime::Timeframe);

  std::vector<InputSpec> inputs;
  inputs.emplace_back("input", "FT0", "CALIB_INFO");
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return DataProcessorSpec{
    "calib-ft0calib-collector",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(ccdbRequest)},
    Options{
      {"max-number-hits-to-fill-tree", VariantType::Int, 1000, {"maximum number of entries in one channel to trigger teh filling of the tree"}},
      {"is-max-number-hits-to-fill-tree-absolute", VariantType::Bool, false, {"to decide if we want to multiply the max-number-hits-to-fill-tree by the number of channels (when set to true), or not (when set to false) for fast checks"}},
      {"tf-sending-policy", VariantType::Bool, false, {"if we are sending output at every TF; otherwise, we use the max-number-hits-to-fill-tree"}},
      {"tf-per-slot", o2::framework::VariantType::UInt32, 200000u, {"TF per slot "}}}};
}

} // namespace framework

} // namespace o2

#endif
