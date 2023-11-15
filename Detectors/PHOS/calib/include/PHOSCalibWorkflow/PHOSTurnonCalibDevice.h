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

#ifndef O2_CALIBRATION_PHOSTURNON_CALIBDEV_H
#define O2_CALIBRATION_PHOSTURNON_CALIBDEV_H

/// @file   PHOSTurnonCalibDevice.h
/// @brief  Device to calculate PHOS turn-on curves and trigger map

#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DataFormatsPHOS/Cell.h"
#include "DataFormatsPHOS/Cluster.h"
#include "DataFormatsPHOS/TriggerMap.h"
#include "PHOSCalibWorkflow/PHOSTurnonCalibrator.h"
#include "DetectorsBase/GRPGeomHelper.h"

using namespace o2::framework;

namespace o2
{
namespace phos
{

class PHOSTurnonCalibDevice : public o2::framework::Task
{
 public:
  explicit PHOSTurnonCalibDevice(bool useCCDB, std::shared_ptr<o2::base::GRPGeomRequest> req) : mUseCCDB(useCCDB), mCCDBRequest(req) {}

  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

 protected:
  bool checkFitResult() { return true; } // TODO!! implement true check

 private:
  bool mUseCCDB = false;
  unsigned long mRunStartTime = 0;                   /// start time of the run (ms)
  std::unique_ptr<TriggerMap> mTriggerMap;           /// Final calibration object
  std::unique_ptr<PHOSTurnonCalibrator> mCalibrator; /// Agregator of calibration TimeFrameSlots
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
};

o2::framework::DataProcessorSpec getPHOSTurnonCalibDeviceSpec(bool useCCDB);
} // namespace phos
} // namespace o2

#endif
