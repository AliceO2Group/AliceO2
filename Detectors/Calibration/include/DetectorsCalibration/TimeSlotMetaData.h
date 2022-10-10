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

#ifndef DETECTOR_CALIB_TIMESLOTMETADATA_H_
#define DETECTOR_CALIB_TIMESLOTMETADATA_H_

/// @brief meta-data for the saved content of the timeslot

namespace o2
{
namespace calibration
{
struct TimeSlotMetaData {
  using TFType = uint32_t;

  int startRun = -1;
  int endRun = -1;
  long startTime = -1;
  long endTime = -1;

  ClassDefNV(TimeSlotMetaData, 1);
};

} // namespace calibration
} // namespace o2

#endif
