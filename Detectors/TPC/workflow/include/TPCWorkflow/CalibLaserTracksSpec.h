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

#ifndef O2_TPC_CalibLaserTracksSpec_H
#define O2_TPC_CalibLaserTracksSpec_H

/// @file   CalibLaserTracksSpec.h
/// @brief  Device to run tpc laser track calibration

#include "TPCCalibration/CalibLaserTracks.h"
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

class CalibLaserTracksDevice : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
    mCalib.setWriteDebugTree(ic.options().get<bool>("write-debug"));
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto dph = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header);
    const auto startTime = dph->startTime;
    const auto endTime = dph->startTime + dph->duration;

    auto data = pc.inputs().get<gsl::span<TrackTPC>>("input");
    mCalib.setTFtimes(startTime, endTime);
    mCalib.fill(data);

    //sendOutput(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "CalibLaserTracksDevice::endOfStream: Finalizing calibration");
    mCalib.finalize();
    mCalib.print();
    sendOutput(ec.outputs());
  }

 private:
  CalibLaserTracks mCalib;

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // TODO in principle, this routine is generic, can be moved to Utils.h
  }
};

DataProcessorSpec getCalibLaserTracks(const std::string inputSpec)
{
  using device = o2::tpc::CalibLaserTracksDevice;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{"TPC", "LtrCalibData"});
  return DataProcessorSpec{
    "tpc-calib-laser-tracks",
    select(inputSpec.data()),
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"write-debug", VariantType::Bool, false, {"write a debug output tree."}},
    }};
}

} // namespace o2::tpc

#endif
