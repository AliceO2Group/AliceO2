// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FITCALIBRATIONAPI_H
#define O2_FITCALIBRATIONAPI_H


#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"
#include "FT0Calibration/FT0ChannelTimeTimeSlotContainer.h"

namespace o2::fit
{
class FITCalibrationApi
{
 private:

  static constexpr const char* DEFAULT_CCDB_URL = "http://localhost:8080";
 public:

  static void init();

  template <typename CalibrationObjectType>
  [[nodiscard]] static const char* getObjectPath();

  template <typename CalibrationObjectType>
  [[nodiscard]] static const CalibrationObjectType& getMostRecentCalibrationObject();

};

} // namespace o2::fit



#endif //O2_FITCALIBRATIONAPI_H
