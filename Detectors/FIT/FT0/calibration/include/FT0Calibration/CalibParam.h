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
#include "FT0Base/Geometry.h"

#include <array>

namespace o2
{
namespace ft0
{

struct CalibParam : o2::conf::ConfigurableParamHelper<CalibParam> {
  static constexpr auto Nchannels = o2::ft0::Geometry::Nchannels;
  // Logic for obtaining bad channels and for making decision concerning slot finalization
  std::size_t mMinEntriesThreshold = 500;  // Min number of entries
  std::size_t mMaxEntriesThreshold = 1000; // Max number of entries
  uint8_t mNExtraSlots = 1;                // Number of extra slots
  // Fitting ranges
  double mMinFitRange = -200.; // Min fit range
  double mMaxFitRange = 200.;  // Max fit range
  // Conditions for checking fit quality, otherwise hist mean will be taken as offset
  double mMaxDiffMean = 20;                 // Max differnce between mean and fit result
  double mMinRMS = 3;                       // Min RMS
  double mMaxSigma = 30;                    // Max fit sigma
  int mRebinFactorPerChID[Nchannels] = {0}; //[Nchannels]
  //
  bool mUseDynamicRange = false; // use dynamic ranges [mean-RMS*mRangeInRMS,mean+RMS*mRangeInRMS] for fitting
  double mRangeInRMS = 3;        // Range for RMS in dynamic case

  O2ParamDef(CalibParam, "FT0CalibParam");
};

} // end namespace ft0
} // end namespace o2

#endif /* FT0_CALIB_PARAM_H_ */