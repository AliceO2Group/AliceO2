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
/// \brief Definition of the FT0 collision time reconstruction task
#ifndef ALICEO2_FT0_COLLISIONTIMERECOTASK_H
#define ALICEO2_FT0_COLLISIONTIMERECOTASK_H

#include "FT0Base/Geometry.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsFT0/FT0ChannelTimeCalibrationObject.h"
#include <gsl/span>
#include <array>
#include <TGraph.h>

namespace o2
{
namespace ft0
{
class CollisionTimeRecoTask
{
  using offsetCalib = o2::ft0::FT0ChannelTimeCalibrationObject;
  static constexpr int NCHANNELS = o2::ft0::Geometry::Nchannels;

 public:
  enum : int { TimeMean,
               TimeA,
               TimeC,
               Vertex };
  CollisionTimeRecoTask() = default;
  ~CollisionTimeRecoTask() = default;
  o2::ft0::RecPoints process(o2::ft0::Digit const& bcd,
                             gsl::span<const o2::ft0::ChannelData> inChData,
                             gsl::span<o2::ft0::ChannelDataFloat> outChData);
  void FinishTask();
  void SetChannelOffset(o2::ft0::FT0ChannelTimeCalibrationObject const* caliboffsets) { mCalibOffset = caliboffsets; };
  void SetSlew(std::array<TGraph, NCHANNELS>* calibslew)
  {
    LOG(info) << "@@@SetSlew " << calibslew->size();
    mCalibSlew = calibslew;
  };
  int getOffset(int channel, int amp);

 private:
  o2::ft0::FT0ChannelTimeCalibrationObject const* mCalibOffset = nullptr;
  std::array<TGraph, NCHANNELS>* mCalibSlew = nullptr;

  ClassDefNV(CollisionTimeRecoTask, 3);
};
} // namespace ft0
} // namespace o2
#endif
