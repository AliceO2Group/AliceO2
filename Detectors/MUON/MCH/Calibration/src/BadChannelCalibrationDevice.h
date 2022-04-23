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

#ifndef O2_MCH_CALIBRATION_PEDESTAL_CALIBRATION_DEVICE_H
#define O2_MCH_CALIBRATION_PEDESTAL_CALIBRATION_DEVICE_H

#include "MCHCalibration/BadChannelCalibrator.h"
#include "Framework/Task.h"
#include <memory>
#include <string_view>
#include "DetectorsBase/GRPGeomHelper.h"

namespace o2::framework
{
class InitContext;
class EndOfStreamContext;
class ProcessingContext;
class DataAllocator;
} // namespace o2::framework

namespace o2::mch::calibration
{
/**
 * @class BadChannelCalibrationDevice
 * @brief DPL device to compute MCH Bad Channel Map
 * @brief
 *
 */
class BadChannelCalibrationDevice : public o2::framework::Task
{
 public:
  explicit BadChannelCalibrationDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}

  void init(o2::framework::InitContext& ic) final;

  void logStats(size_t dataSize);

  void run(o2::framework::ProcessingContext& pc) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void sendOutput(o2::framework::DataAllocator& output, std::string_view reason);

 private:
  std::unique_ptr<o2::mch::calibration::BadChannelCalibrator> mCalibrator;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  uint64_t mTimeStamp;

  int mLoggingInterval = {0}; /// time interval between statistics logging messages
};

} // namespace o2::mch::calibration
#endif
