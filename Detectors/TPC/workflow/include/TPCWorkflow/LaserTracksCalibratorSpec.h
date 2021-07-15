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

#ifndef O2_CALIBRATION_LHCCLOCK_CALIBRATOR_H
#define O2_CALIBRATION_LHCCLOCK_CALIBRATOR_H

/// @file   CalibLaserTracksSpec.h
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
    int minEnt = std::max(300, ic.options().get<int>("min-entries"));
    int slotL = ic.options().get<int>("tf-per-slot");
    int delay = ic.options().get<int>("max-delay");
    mCalibrator = std::make_unique<LaserTracksCalibrator>(minEnt);
    mCalibrator->setSlotLength(slotL);
    mCalibrator->setMaxSlotsDelay(delay);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
    auto data = pc.inputs().get<gsl::span<TrackTPC>>("input");
    LOG(INFO) << "Processing TF " << tfcounter << " with " << data.size() << " tracks";
    mCalibrator->process(tfcounter, data);
    sendOutput(pc.outputs());
    //const auto& infoVec = mCalibrator->getLHCphaseInfoVector();
    //LOG(INFO) << "Created " << infoVec.size() << " objects for TF " << tfcounter;
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(INFO) << "Finalizing calibration";
    constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
    mCalibrator->checkSlotsToFinalize(INFINITE_TF);
    sendOutput(ec.outputs());
  }

 private:
  std::unique_ptr<LaserTracksCalibrator> mCalibrator;

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // TODO in principle, this routine is generic, can be moved to Utils.h
    using clbUtils = o2::calibration::Utils;
    const auto& object = mCalibrator->getDVperSlot();

    o2::ccdb::CcdbObjectInfo w;
    auto image = o2::ccdb::CcdbApi::createObjectImage(&object, &w);

    w.setPath("TPC/Calib/LaserTracks");
    w.setStartValidityTimestamp(object.front().time);
    w.setEndValidityTimestamp(object.back().time);

    LOG(INFO) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
              << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibLtr", 0}, *image.get()); // vector<char>
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibLtr", 0}, w);            // root-serialized
    mCalibrator->initOutput();
  }
};

DataProcessorSpec getCalibLaserTracks()
{
  using device = o2::tpc::LaserTracksCalibratorDevice;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibLtr"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibLtr"});
  return DataProcessorSpec{
    "tpc-calib-laser-tracks",
    Inputs{{"input", "TPC", "TRACKS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"tf-per-slot", VariantType::Int, 5, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::Int, 3, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 500, {"minimum number of entries to fit single time slot"}},
    }};
}

} // namespace o2::tpc

#endif
