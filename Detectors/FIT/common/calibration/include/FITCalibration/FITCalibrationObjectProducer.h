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

#ifndef O2_FITCALIBRATIONOBJECTPRODUCER_H
#define O2_FITCALIBRATIONOBJECTPRODUCER_H

#include "FITCalibration/FITCalibrationApi.h"
#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"
#include "FT0Calibration/FT0ChannelTimeTimeSlotContainer.h"
#include "FT0Calibration/FT0CalibTimeSlewing.h"
#include "FT0Calibration/LHCphaseCalibrationObject.h"
#include "FT0Calibration/LHCClockDataHisto.h"
#include "FV0Calibration/FV0ChannelTimeCalibrationObject.h"
#include "FV0Calibration/FV0ChannelTimeTimeSlotContainer.h"
#include "FITCalibration/FITCalibrationObjectProducer.h"
#include "FT0Calibration/FT0DummyCalibrationObject.h" // delete this
#include "DataFormatsFT0/GlobalOffsetsCalibrationObject.h"
#include "DataFormatsFT0/GlobalOffsetsContainer.h"

namespace o2::fit
{
class FITCalibrationObjectProducer
{
 public:
  template <typename CalibrationObjectType, typename TimeSlotContainerType>
  static CalibrationObjectType generateCalibrationObject(const TimeSlotContainerType& container);
};

template <typename CalibrationObjectType, typename TimeSlotContainerType>
CalibrationObjectType FITCalibrationObjectProducer::generateCalibrationObject(const TimeSlotContainerType& container)
{
  LOG(info)<<" FITCalibrationObjectProducer::generateCalibrationObject";
  static_assert(sizeof(CalibrationObjectType) == 0, "[FITCalibrationObjectProducer] Cannot find specialization provided Calibration Object Type");
  return {};
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
o2::ft0::FT0ChannelTimeCalibrationObject FITCalibrationObjectProducer::generateCalibrationObject<o2::ft0::FT0ChannelTimeCalibrationObject, o2::ft0::FT0ChannelTimeTimeSlotContainer>(const o2::ft0::FT0ChannelTimeTimeSlotContainer& container)
{
  LOG(info)<<"FITCalibrationObjectProducer::generateCalibrationObject";
  return o2::ft0::FT0TimeChannelOffsetCalibrationObjectAlgorithm::generateCalibrationObject(container);
}
template <>
o2::ft0::LHCphaseCalibrationObject FITCalibrationObjectProducer::generateCalibrationObject<o2::ft0::LHCphaseCalibrationObject, o2::ft0::LHCClockDataHisto>(const o2::ft0::LHCClockDataHisto& container)
{
  return o2::ft0::LHCphaseCalibrationObjectAlgorithm::generateCalibrationObject(container);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
o2::ft0::GlobalOffsetsCalibrationObject FITCalibrationObjectProducer::generateCalibrationObject<o2::ft0::GlobalOffsetsCalibrationObject, o2::ft0::GlobalOffsetsContainer>(const o2::ft0::GlobalOffsetsContainer& container)
{
  LOG(info)<<"  FITCalibrationObjectProducer::generateCalibrationObject";
  return o2::ft0::GlobalOffsetsCalibrationObjectAlgorithm::generateCalibrationObject(container);
}

template <>
o2::fv0::FV0ChannelTimeCalibrationObject FITCalibrationObjectProducer::generateCalibrationObject<o2::fv0::FV0ChannelTimeCalibrationObject, o2::fv0::FV0ChannelTimeTimeSlotContainer>(const o2::fv0::FV0ChannelTimeTimeSlotContainer& container)
{
  return o2::fv0::FV0TimeChannelOffsetCalibrationObjectAlgorithm::generateCalibrationObject(container);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

// DUMMY STUFF DELETE IT WHEN EXAMPLE NOT NEEDED ANYMORE

template <>
o2::ft0::FT0DummyCalibrationObject FITCalibrationObjectProducer::generateCalibrationObject<o2::ft0::FT0DummyCalibrationObject, o2::ft0::FT0ChannelTimeTimeSlotContainer>(const o2::ft0::FT0ChannelTimeTimeSlotContainer& container)
{
  //We need additional object here for calibration
  const auto& neededObjectForCalibration = FITCalibrationApi::getMostRecentCalibrationObject<o2::ft0::FT0DummyNeededCalibrationObject>();
  //  const auto& neededObjectForCalibration = FITCalibrationApi::getCalibrationObjectForGivenTimestamp<o2::ft0::FT0DummyNeededCalibrationObject>(FITCalibrationApi::getProcessingTimestamp());
  return o2::ft0::FT0DummyCalibrationObjectAlgorithm::generateCalibrationObject(container, neededObjectForCalibration);
}

// END OF DUMMY STUFF DELETE IT WHEN EXAMPLE NOT NEEDED ANYMORE

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace o2::fit

#endif //O2_FITCALIBRATIONOBJECTPRODUCER_H
