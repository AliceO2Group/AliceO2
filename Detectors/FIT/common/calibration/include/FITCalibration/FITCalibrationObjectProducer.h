// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "FITCalibration/FITCalibrationObjectProducer.h"
#include "FT0Calibration/FT0DummyCalibrationObject.h" // delete this

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
  static_assert(sizeof(CalibrationObjectType) == 0, "[FITCalibrationObjectProducer] Cannot find specialization provided Calibration Object Type");
  return {};
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
o2::ft0::FT0ChannelTimeCalibrationObject FITCalibrationObjectProducer::generateCalibrationObject<o2::ft0::FT0ChannelTimeCalibrationObject, o2::ft0::FT0ChannelTimeTimeSlotContainer>(const o2::ft0::FT0ChannelTimeTimeSlotContainer& container)
{
  return o2::ft0::FT0TimeChannelOffsetCalibrationObjectAlgorithm::generateCalibrationObject(container);
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
