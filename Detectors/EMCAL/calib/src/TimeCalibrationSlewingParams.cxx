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

#include "EMCALCalib/TimeCalibrationSlewingParams.h"

// using namespace o2::emcal;
namespace o2
{

namespace emcal
{

bool TimeCalibrationSlewingParams::operator==(const TimeCalibrationSlewingParams& other) const
{
  for (int i = 0; i < 4; ++i) {
    if (std::abs(arrParams[i] - other.getTimeSlewingParam(i)) > DBL_EPSILON) {
      return false;
    }
  }
  return true;
}

TimeCalibrationSlewingParams::TimeCalibrationSlewingParams(std::array<double, 4> arr)
{
  arrParams = arr;
}

void TimeCalibrationSlewingParams::addTimeSlewingParam(std::array<double, 4> arr)
{
  arrParams = arr;
}

double TimeCalibrationSlewingParams::getTimeSlewingParam(unsigned int index) const
{
  if (index >= 4) {
    return -1;
  }
  return arrParams[index];
}

double TimeCalibrationSlewingParams::eval(double energy) const
{
  double val = 0;
  for (unsigned int i = 0; i < arrParams.size(); ++i) {
    val += arrParams[i] * std::pow(energy, i);
  }
  return val;
}

} // namespace emcal
} // namespace o2