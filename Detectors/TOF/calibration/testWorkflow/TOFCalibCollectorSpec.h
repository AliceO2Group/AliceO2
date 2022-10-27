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

#ifndef O2_CALIBRATION_TOFCALIB_COLLECTOR_H
#define O2_CALIBRATION_TOFCALIB_COLLECTOR_H

/// @file   TOFCalibCollectorSpec.h
/// @brief  Device to collect information for TOF time slewing calibration.

#include "TOFCalibration/TOFCalibCollector.h"
#include "DetectorsCalibration/Utils.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TOFBase/Geo.h"
#include "DataFormatsTOF/CalibLHCphaseTOF.h"
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"
#include "Framework/CCDBParamSpec.h"

#include <limits>

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class TOFCalibCollectorDevice : public o2::framework::Task
{
  using TimeSlewing = o2::dataformats::CalibTimeSlewingParamTOF;
  using LHCphase = o2::dataformats::CalibLHCphaseTOF;

 public:
  TOFCalibCollectorDevice(std::shared_ptr<o2::base::GRPGeomRequest> req, bool useCCDB) : mCCDBRequest(req), mUseCCDB(useCCDB) {}

  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    bool isTFsendingPolicy = ic.options().get<bool>("tf-sending-policy");
    int maxEnt = ic.options().get<int>("max-number-hits-to-fill-tree");
    bool isTest = ic.options().get<bool>("running-in-test-mode");
    bool absMaxEnt = ic.options().get<bool>("is-max-number-hits-to-fill-tree-absolute");
    auto updateInterval = ic.options().get<uint32_t>("update-interval");
    mCollector = std::make_unique<o2::tof::TOFCalibCollector>(isTFsendingPolicy, maxEnt);
    mCollector->setIsTest(isTest);
    mCollector->setIsMaxNumberOfHitsAbsolute(absMaxEnt);
    mCollector->setFinalizeWhenReady(); // finalize slot once stat is ok and create next one
    mCollector->setCheckIntervalInfiniteSlot(updateInterval);
    mCollector->setMaxSlotsDelay(0);
  }

  //_________________________________________________________________
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    auto data = pc.inputs().get<gsl::span<o2::dataformats::CalibInfoTOF>>("input");

    if (mUseCCDB) { // read calibration objects from ccdb
      const auto lhcPhaseIn = pc.inputs().get<LHCphase*>("tofccdbLHCphase");
      const auto channelCalibIn = pc.inputs().get<TimeSlewing*>("tofccdbChannelCalib");

      float phase = lhcPhaseIn->getLHCphase(0);
      int bcshift = (phase + 5000) * o2::tof::Geo::BC_TIME_INPS_INV;
      mCollector->setLHCphase(phase - bcshift * o2::tof::Geo::BC_TIME_INPS);
    }

    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCollector->getCurrentTFInfo());
    LOG(info) << "Processing TF " << mCollector->getCurrentTFInfo().tfCounter << " with " << data.size() << " tracks";
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
  std::unique_ptr<o2::tof::TOFCalibCollector> mCollector;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  int mMaxNumOfHits = 0;
  bool mUseCCDB = false;

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
      output.snapshot(Output{o2::header::gDataOriginTOF, "COLLECTEDINFO", 0, Lifetime::Timeframe}, collectedInfo);
      output.snapshot(Output{o2::header::gDataOriginTOF, "ENTRIESCH", 0, Lifetime::Timeframe}, entriesPerChannel);
      mCollector->initOutput(); // reset the output for the next round
    }
  }
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getTOFCalibCollectorDeviceSpec(bool useCCDB)
{
  using device = o2::calibration::TOFCalibCollectorDevice;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTOF, "COLLECTEDINFO", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "ENTRIESCH", 0, Lifetime::Timeframe);

  std::vector<InputSpec> inputs;
  inputs.emplace_back("input", "TOF", "CALIBDATA");
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  if (useCCDB) {
    inputs.emplace_back("tofccdbLHCphase", o2::header::gDataOriginTOF, "LHCphase", 0, Lifetime::Condition, ccdbParamSpec("TOF/Calib/LHCphase"));
    inputs.emplace_back("tofccdbChannelCalib", o2::header::gDataOriginTOF, "ChannelCalib", 0, Lifetime::Condition, ccdbParamSpec("TOF/Calib/ChannelCalib"));
  }

  return DataProcessorSpec{
    "calib-tofcalib-collector",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(ccdbRequest, useCCDB)},
    Options{
      {"max-number-hits-to-fill-tree", VariantType::Int, 500, {"maximum number of entries in one channel to trigger teh filling of the tree"}},
      {"is-max-number-hits-to-fill-tree-absolute", VariantType::Bool, false, {"to decide if we want to multiply the max-number-hits-to-fill-tree by the number of channels (when set to true), or not (when set to false) for fast checks"}},
      {"tf-sending-policy", VariantType::Bool, false, {"if we are sending output at every TF; otherwise, we use the max-number-hits-to-fill-tree"}},
      {"running-in-test-mode", VariantType::Bool, false, {"to run in test mode for simplification"}},
      {"update-interval", VariantType::UInt32, 10u, {"number of TF after which to try to finalize calibration"}}}};
}

} // namespace framework
} // namespace o2

#endif
