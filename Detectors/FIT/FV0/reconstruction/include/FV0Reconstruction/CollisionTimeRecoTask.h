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

/// \file CollisionTimeRecoTask.h
/// \brief Definition of the FV0 collision time reconstruction task
#ifndef ALICEO2_FV0_COLLISIONTIMERECOTASK_H
#define ALICEO2_FV0_COLLISIONTIMERECOTASK_H

#include "DataFormatsFV0/BCData.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/RecPoints.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
//#include "FV0Calibration/FT0ChannelTimeCalibrationObject.h"
#include "FV0Base/Constants.h"
#include <gsl/span>
#include <bitset>
#include <vector>
#include <array>
#include <TGraph.h>

namespace o2
{
namespace fv0
{
class CollisionTimeRecoTask
{
  //  using offsetCalib = o2::fv0::FV0ChannelTimeCalibrationObject;

 public:
  CollisionTimeRecoTask() = default;
  ~CollisionTimeRecoTask() = default;
  o2::fv0::RecPoints process(o2::fv0::BCData const& bcd,
                             gsl::span<const o2::fv0::ChannelData> inChData,
                             gsl::span<o2::fv0::ChannelDataFloat> outChData);
  void FinishTask();
  //  void SetChannelOffset(o2::fv0::FT0ChannelTimeCalibrationObject* caliboffsets) { mCalibOffset = caliboffsets; };
  //  void SetSlew(std::array<TGraph, Constants::nFv0Channels>* calibslew) { mCalibSlew = calibslew; };
  //  int getOffset(int channel, int amp);

 private:
  //  o2::fv0::FV0ChannelTimeCalibrationObject* mCalibOffset;
  //  std::array<TGraph, Constants::nFv0Channels>* mCalibSlew;

  ClassDefNV(CollisionTimeRecoTask, 3);
};
} // namespace fv0
} // namespace o2
#endif
