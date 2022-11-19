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

#ifndef O2_CALIBRATION_PHOSL1PHASE_CALIBDEV_H
#define O2_CALIBRATION_PHOSL1PHASE_CALIBDEV_H

/// @file   PHOSL1phaseCalibDevice.h
/// @brief  Device to calculate PHOS time shift (L1phase)

#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ProcessingContext.h"
#include "PHOSCalibWorkflow/PHOSL1phaseCalibrator.h"
#include "DetectorsBase/GRPGeomHelper.h"

using namespace o2::framework;

namespace o2
{
namespace phos
{

class PHOSL1phaseCalibDevice
{
 public:
  PHOSL1phaseCalibDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}

  void init(o2::framework::InitContext& ic);

  void run(o2::framework::ProcessingContext& pc);

  void endOfStream(o2::framework::EndOfStreamContext& ec);

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

 private:
  unsigned long mRunStartTime = 0;                              /// start time of the run (ms)
  std::unique_ptr<o2::phos::PHOSL1phaseCalibrator> mCalibrator; /// Agregator of calibration TimeFrameSlots
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
};

o2::framework::DataProcessorSpec getPHOSL1phaseCalibDeviceSpec();
} // namespace phos
} // namespace o2

#endif
