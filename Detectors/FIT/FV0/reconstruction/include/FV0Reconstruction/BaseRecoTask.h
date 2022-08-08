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

/// \file BaseRecoTask.h
/// \brief Definition of the FV0 reconstruction task
#ifndef ALICEO2_FV0_BASERECOTASK_H
#define ALICEO2_FV0_BASERECOTASK_H

#include "DataFormatsFV0/Digit.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/RecPoints.h"
#include "DataFormatsFV0/FV0ChannelTimeCalibrationObject.h"
#include <gsl/span>

namespace o2
{
namespace fv0
{
class BaseRecoTask
{
  //  using offsetCalib = o2::fv0::FV0ChannelTimeCalibrationObject;

 public:
  BaseRecoTask() = default;
  ~BaseRecoTask() = default;
  o2::fv0::RecPoints process(o2::fv0::Digit const& bcd,
                             gsl::span<const o2::fv0::ChannelData> inChData,
                             gsl::span<o2::fv0::ChannelDataFloat> outChData);
  void FinishTask();
  void SetChannelOffset(o2::fv0::FV0ChannelTimeCalibrationObject const* caliboffsets) { mCalibOffset = caliboffsets; };
  int getOffset(int channel);

 private:
  o2::fv0::FV0ChannelTimeCalibrationObject const* mCalibOffset = nullptr;

  ClassDefNV(BaseRecoTask, 3);
};
} // namespace fv0
} // namespace o2
#endif
