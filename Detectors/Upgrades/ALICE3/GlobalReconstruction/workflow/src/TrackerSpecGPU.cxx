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

#include "ALICE3GlobalReconstruction/GPUExternalAllocator.h"
#include "ALICE3GlobalReconstruction/TimeFrameGPU.h"
#include "ALICE3GlobalReconstructionWorkflow/TrackerSpec.h"
#include "ALICE3GlobalReconstructionWorkflow/TrackerSpecImpl.h"
#include "ITStrackingGPU/TrackerTraitsGPU.h"

extern "C" int runALICE3GPUTracking(o2::trk::TrackerDPL* tracker, o2::framework::ProcessingContext* pc)
{
  o2::trk::TimeFrameGPU<11> timeFrame;
  o2::its::TrackerTraitsGPU<11> itsTrackerTraits;
  if (!tracker->getGPUAllocator()) {
    tracker->setGPUAllocator(std::make_shared<o2::trk::GPUExternalAllocator>());
  }
  timeFrame.setFrameworkAllocator(tracker->getGPUAllocator().get());
  tracker->runTracking(*pc, timeFrame, itsTrackerTraits);
  return 0;
}
