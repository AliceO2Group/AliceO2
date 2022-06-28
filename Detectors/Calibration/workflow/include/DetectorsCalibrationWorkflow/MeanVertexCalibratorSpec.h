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

#ifndef O2_MEANVERTEX_CALIB_DEVICE_H
#define O2_MEANVERTEX_CALIB_DEVICE_H

/// @file   MeanVertexCalibratorSpec.h
/// @brief  Device to calibrate MeanVertex

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DetectorsCalibration/MeanVertexCalibrator.h"

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class MeanVertexCalibDevice : public Task
{
 public:
  MeanVertexCalibDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void sendOutput(DataAllocator& output);

  std::unique_ptr<o2::calibration::MeanVertexCalibrator> mCalibrator;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
};

} // namespace calibration

namespace framework
{
DataProcessorSpec getMeanVertexCalibDeviceSpec();

} // namespace framework
} // namespace o2

#endif
