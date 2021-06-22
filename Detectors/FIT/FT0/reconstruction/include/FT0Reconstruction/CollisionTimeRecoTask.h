// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CollisionTimeRecoTask.h
/// \brief Definition of the FIT collision time reconstruction task
#ifndef ALICEO2_FIT_COLLISIONTIMERECOTASK_H
#define ALICEO2_FIT_COLLISIONTIMERECOTASK_H

#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/RecPoints.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"
#include "FT0Base/Geometry.h"
#include <gsl/span>
#include <bitset>
#include <vector>
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
  void SetChannelOffset(o2::ft0::FT0ChannelTimeCalibrationObject* caliboffsets) { mCalibOffset = caliboffsets; };
  void SetSlew(std::array<TGraph, NCHANNELS>* calibslew) { mCalibSlew = calibslew; };
  int getOffset(int channel, int amp);

 private:
  o2::ft0::FT0ChannelTimeCalibrationObject* mCalibOffset;
  std::array<TGraph, NCHANNELS>* mCalibSlew;

  ClassDefNV(CollisionTimeRecoTask, 3);
};
} // namespace ft0
} // namespace o2
#endif
