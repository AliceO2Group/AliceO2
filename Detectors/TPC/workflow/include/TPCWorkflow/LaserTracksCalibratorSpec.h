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

using namespace o2::framework;

namespace o2::tpc
{

class LaserTracksCalibratorDevice : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
    const int minTFs = ic.options().get<int>("min-tfs");
    const int slotL = ic.options().get<int>("tf-per-slot");
    const int delay = ic.options().get<int>("max-delay");
    const bool debug = ic.options().get<bool>("write-debug");

    mCalibrator = std::make_unique<LaserTracksCalibrator>(minTFs);
    mCalibrator->setSlotLength(slotL);
    mCalibrator->setMaxSlotsDelay(delay);
    mCalibrator->setWriteDebug(debug);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto dph = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("laserTracks").header);
    const auto startTime = dph->startTime;
    const auto endTime = dph->startTime + dph->duration - 1;

    auto data = pc.inputs().get<gsl::span<TrackTPC>>("laserTracks");
    LOGP(info, "Processing TF with start time {} and {} tracks", startTime, data.size());

    mCalibrator->getSlotForTF(startTime).getContainer()->setTFtimes(startTime, endTime);
    mCalibrator->process(startTime, data);

    if (mCalibrator->hasCalibrationData()) {
      sendOutput(pc.outputs());
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "LaserTracksCalibratorDevice::endOfStream: Finalizing calibration");
    constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
    mCalibrator->checkSlotsToFinalize(INFINITE_TF);
    sendOutput(ec.outputs());
  }

 private:
  std::unique_ptr<LaserTracksCalibrator> mCalibrator;

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    using clbUtils = o2::calibration::Utils;
    const auto& calibrations = mCalibrator->getCalibPerSlot();
    const long timeEnd = 99999999999999;
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
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibLtr"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibLtr"});
  return DataProcessorSpec{
    "tpc-laser-tracks-calibrator",
    Inputs{{"laserTracks", "TPC", "LASERTRACKS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"tf-per-slot", VariantType::Int, 5000, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::Int, 3, {"number of slots in past to consider"}},
      {"min-tfs", VariantType::Int, 100, {"minimum number of TFs with enough laser tracks to finalize a slot"}},
      {"write-debug", VariantType::Bool, false, {"write a debug output tree."}},
    }};
}

} // namespace o2::tpc

#endif
