// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FITCalibration/FITCalibrationAlgorithmGetter.h"
#include "FairLogger.h"

using namespace o2::fit;

template <typename CalibrationObjectType, typename TimeSlotContainerType>
void FITCalibrationAlgorithmGetter::doCalibrationAndUpdateCalibrationObject(CalibrationObjectType& calibrationObject, const TimeSlotContainerType& container)
{
  LOG(WARN) << "Unable to find proper calibration algorithm, object will not be modified!\n";
}

template <>
void FITCalibrationAlgorithmGetter::doCalibrationAndUpdateCalibrationObject<o2::ft0::FT0CalibrationObject, o2::ft0::FT0ChannelDataTimeSlotContainer>(o2::ft0::FT0CalibrationObject& calibrationObject, const o2::ft0::FT0ChannelDataTimeSlotContainer& container)
{
  o2::ft0::FT0CalibrationObjectAlgorithm::calibrate(calibrationObject, container);
}
