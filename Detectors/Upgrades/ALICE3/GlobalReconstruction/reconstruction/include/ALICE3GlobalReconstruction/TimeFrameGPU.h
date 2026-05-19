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
/// \file TimeFrameGPU.h
/// \brief GPU TRK TimeFrame wrapper.
///

#ifndef ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAMEGPU_H
#define ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAMEGPU_H

#include "ALICE3GlobalReconstruction/TimeFrameMixin.h"
#include "ITStrackingGPU/TimeFrameGPU.h"

namespace o2::trk
{

template <int nLayers = 11>
class TimeFrameGPU : public TimeFrameMixin<nLayers, o2::its::gpu::TimeFrameGPU<nLayers>>
{
 public:
  TimeFrameGPU() = default;
  ~TimeFrameGPU() override = default;
};

} // namespace o2::trk

#endif // ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAMEGPU_H
