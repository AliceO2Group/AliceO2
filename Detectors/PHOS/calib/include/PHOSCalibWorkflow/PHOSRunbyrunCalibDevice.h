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

#ifndef O2_CALIBRATION_PHOSRUNBYRUN_CALIBDEV_H
#define O2_CALIBRATION_PHOSRUNBYRUN_CALIBDEV_H

/// @file   PHOSRunbyrunCalibDevice.h
/// @brief  Device to calculate PHOS energy run by run corrections

#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ProcessingContext.h"
#include "DataFormatsPHOS/Cluster.h"
#include "DataFormatsPHOS/BadChannelsMap.h"
#include "PHOSCalibWorkflow/PHOSRunbyrunCalibrator.h"
#include "DetectorsBase/GRPGeomHelper.h"

using namespace o2::framework;

namespace o2
{
namespace phos
{

class PHOSRunbyrunCalibDevice
{
 public:
  PHOSRunbyrunCalibDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}

  void init(o2::framework::InitContext& ic);

  void run(o2::framework::ProcessingContext& pc);

  void endOfStream(o2::framework::EndOfStreamContext& ec);

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

 protected:
  bool checkFitResult();

 private:
  bool mUseCCDB = false;
  long mRunStartTime = 0;                              /// start time of the run (sec)
  std::array<float, 8> mRunByRun;                      /// Final calibration object
  std::unique_ptr<PHOSRunbyrunCalibrator> mCalibrator; /// Agregator of calibration TimeFrameSlots
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
};

o2::framework::DataProcessorSpec getPHOSRunbyrunCalibDeviceSpec(bool useCCDB);
} // namespace phos
} // namespace o2

#endif
