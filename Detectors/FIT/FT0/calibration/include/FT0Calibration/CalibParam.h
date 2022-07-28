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

#ifndef ALICEO2_CALIB_PARAM_H_
#define ALICEO2_CALIB_PARAM_H_

#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace ft0
{

struct CalibParam : o2::conf::ConfigurableParamHelper<CalibParam> {
  // Logic for obtaining bad channels and for making decision concerning slot finalization
  std::size_t mMinEntriesThreshold = 500;
  std::size_t mMaxEntriesThreshold = 1000;
  uint8_t mNExtraSlots = 1;
  // Fitting ranges
  double mMinFitRange = -200.;
  double mMaxFitRange = 200.;
  // Conditions for checking fit quality, otherwise hist mean will be taken as offset
  double mMaxDiffMean = 20;
  double mMinRMS = 3;
  double mMaxSigma = 30;
  //
  bool mUseDynamicRange = false; // use dynamic ranges [mean-RMS*mRangeInRMS,mean+RMS*mRangeInRMS] for fitting
  double mRangeInRMS = 3;

  O2ParamDef(CalibParam, "FT0CalibParam");
};

} // end namespace ft0
} // end namespace o2

#endif /* FT0_CALIB_PARAM_H_ */