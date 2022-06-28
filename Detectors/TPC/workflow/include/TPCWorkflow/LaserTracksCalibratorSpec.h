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

#ifndef O2_TPC_LaserTracksCalibratorSpec_H
#define O2_TPC_LaserTracksCalibratorSpec_H

/// @file   LaserTracksCalibratorSpec.h
/// @brief  Device to run tpc laser track calibration

#include "TPCCalibration/LaserTracksCalibrator.h"
#include "DetectorsCalibration/Utils.h"
#include "CommonUtils/MemFileHelper.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsBase/GRPGeomHelper.h"

using namespace o2::framework;

namespace o2::tpc
{

class LaserTracksCalibratorDevice : public o2::framework::Task
{
 public:
  LaserTracksCalibratorDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}
  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    const int minTFs = ic.options().get<int>("min-tfs");
    const auto slotL = ic.options().get<uint32_t>("tf-per-slot");
    const auto delay = ic.options().get<uint32_t>("max-delay");
    const bool debug = ic.options().get<bool>("write-debug");

    mCalibrator = std::make_unique<LaserTracksCalibrator>(minTFs);
    mCalibrator->setSlotLength(slotL);
    mCalibrator->setMaxSlotsDelay(delay);
    mCalibrator->setWriteDebug(debug);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    const auto dph = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("laserTracks").header);
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());
    auto data = pc.inputs().get<gsl::span<TrackTPC>>("laserTracks");
    LOGP(info, "Processing TF {} and {} tracks", mCalibrator->getCurrentTFInfo().tfCounter, data.size());

    mCalibrator->process(data);

    if (mCalibrator->hasCalibrationData()) {
      sendOutput(pc.outputs());
    }
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "LaserTracksCalibratorDevice::endOfStream: Finalizing calibration");
    mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
    sendOutput(ec.outputs());
  }

 private:
  std::unique_ptr<LaserTracksCalibrator> mCalibrator;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    using clbUtils = o2::calibration::Utils;
    const auto& calibrations = mCalibrator->getCalibPerSlot();
    const long timeEnd = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP;
    for (uint32_t iCalib = 0; iCalib < calibrations.size(); ++iCalib) {
      const auto& object = calibrations[iCalib];
      o2::ccdb::CcdbObjectInfo w;
      auto image = o2::ccdb::CcdbApi::createObjectImage(&object, &w);

      w.setPath("TPC/Calib/LaserTracks");
      w.setStartValidityTimestamp(object.firstTime);
      //w.setEndValidityTimestamp(object.lastTime);
      w.setEndValidityTimestamp(timeEnd);

      LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", w.getPath(), w.getFileName(), image->size(), w.getStartValidityTimestamp(), w.getEndValidityTimestamp());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibLtr", iCalib}, *image.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibLtr", iCalib}, w);
    }
    mCalibrator->initOutput();
  }
};

DataProcessorSpec getLaserTracksCalibrator()
{
  using device = o2::tpc::LaserTracksCalibratorDevice;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibLtr"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibLtr"}, Lifetime::Sporadic);
  std::vector<InputSpec> inputs{{"laserTracks", "TPC", "LASERTRACKS"}};
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return DataProcessorSpec{
    "tpc-laser-tracks-calibrator",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(ccdbRequest)},
    Options{
      {"tf-per-slot", VariantType::UInt32, 5000u, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::UInt32, 3u, {"number of slots in past to consider"}},
      {"min-tfs", VariantType::Int, 100, {"minimum number of TFs with enough laser tracks to finalize a slot"}},
      {"write-debug", VariantType::Bool, false, {"write a debug output tree."}},
    }};
}

} // namespace o2::tpc

#endif
