// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "FT0Calibration/FT0CalibrationInfoObject.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "FairLogger.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{

class FT0CalibCollectorDevice : public o2::framework::Task
{

 public:
  void init(o2::framework::InitContext& ic) final
  {
    bool isTFsendingPolicy = ic.options().get<bool>("tf-sending-policy");
    int maxEnt = ic.options().get<int>("max-number-hits-to-fill-tree");
    bool isTest = ic.options().get<bool>("running-in-test-mode");
    bool absMaxEnt = ic.options().get<bool>("is-max-number-hits-to-fill-tree-absolute");
    mCollector = std::make_unique<o2::ft0::FT0CalibCollector>(isTFsendingPolicy, maxEnt);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime; // is this the timestamp of the current TF?
    auto data = pc.inputs().get<gsl::span<o2::ft0::FT0CalibrationInfoObject>>("input");
    mCollector->process(tfcounter, data);
    sendOutput(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
    mCollector->checkSlotsToFinalize(INFINITE_TF);
    // we force finalizing slot zero (unless everything was already finalized), no matter how many entries we had
    if (mCollector->getNSlots() != 0) {
      mCollector->finalizeSlot(mCollector->getSlot(0));
    }
    sendOutput(ec.outputs());
  }

 private:
  std::unique_ptr<o2::ft0::FT0CalibCollector> mCollector;
  int mMaxNumOfHits = 0;

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    // in output we send the calibration tree
    auto& collectedInfo = mCollector->getCollectedCalibInfo();
    LOG(DEBUG) << "In CollectorSpec sendOutput: size = " << collectedInfo.size();
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

  return DataProcessorSpec{
    "calib-ft0calib-collector",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"max-number-hits-to-fill-tree", VariantType::Int, 1000, {"maximum number of entries in one channel to trigger teh filling of the tree"}},
      {"is-max-number-hits-to-fill-tree-absolute", VariantType::Bool, false, {"to decide if we want to multiply the max-number-hits-to-fill-tree by the number of channels (when set to true), or not (when set to false) for fast checks"}},
      {"tf-sending-policy", VariantType::Bool, false, {"if we are sending output at every TF; otherwise, we use the max-number-hits-to-fill-tree"}},
      {"running-in-test-mode", VariantType::Bool, false, {"to run in test mode for simplification"}}}};
}

} // namespace framework
} // namespace o2

#endif
