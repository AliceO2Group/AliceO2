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
///

#ifndef TRACKINGITSGPU_INCLUDE_TIMEFRAMEGPU_H
#define TRACKINGITSGPU_INCLUDE_TIMEFRAMEGPU_H

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include "ITStracking/TimeFrame.h"

#include "Array.h"
#include "UniquePointer.h"
#include "Vector.h"
#include "GPUCommonDef.h"

namespace o2
{
namespace its
{
namespace gpu
{
template <int NLayers>
class TimeFrameGPU
{
 public:
  TimeFrameGPU();
  ~TimeFrameGPU() = default;
  UniquePointer<TimeFrameGPU<NLayers>> initialise(TimeFrame* tf);

 private:
};
} // namespace gpu
} // namespace its
} // namespace o2
#endif