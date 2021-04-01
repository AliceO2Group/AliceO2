// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FITCALIBRATIONALGORITHMGETTER_H
#define O2_FITCALIBRATIONALGORITHMGETTER_H

#include "FT0Calibration/FT0CalibrationObject.h"
#include "FT0Calibration/FT0ChannelDataTimeSlotContainer.h"

namespace o2::fit
{
class FITCalibrationAlgorithmGetter
{
 public:
  template <typename CalibrationObjectType, typename TimeSlotContainerType>
  static void doCalibrationAndUpdateCalibrationObject(CalibrationObjectType& calibrationObject, const TimeSlotContainerType& container);
};

} // namespace o2::fit

#endif //O2_FITCALIBRATIONALGORITHMGETTER_H
