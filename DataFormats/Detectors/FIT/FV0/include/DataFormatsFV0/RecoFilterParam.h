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

#ifndef ALICEO2_FV0_DIGIT_FILTER_PARAM
#define ALICEO2_FV0_DIGIT_FILTER_PARAM

#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::fv0
{
struct ChargeFilter : o2::conf::ConfigurableParamHelper<ChargeFilter> {
  double AmplitudeLowerThreshold = 24;     // only channels with amplitude higher will participate in calibration and collision time
  double AmplitudeThreholdForMeanTime = 5; // Charge threshold, only above which the time is taken into account in calculating the mean time of all qualifying channels

  bool validForMeanTimeCalculation(double charge) const
  {
    return charge > AmplitudeThreholdForMeanTime;
  }

  bool validForCalibrationAndCollisionTime(double charge) const
  {
    return charge > AmplitudeLowerThreshold;
  }

  O2ParamDef(ChargeFilter, "FV0RecoChargeFilter");
};

struct TimeFilter : o2::conf::ConfigurableParamHelper<TimeFilter> {
  double TimeUpperThershold = 1000.0; // only channels with time below will participate in calibration and collision time
  bool validForCalibrationAndCollisionTime(double time) const
  {
    return time < TimeUpperThershold;
  }

  O2ParamDef(TimeFilter, "FV0RecoTimeFilter");
};

} // namespace o2::fv0

#endif