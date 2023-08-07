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

// \brief Collect all possible configurable parameters for any QC task

#ifndef O2_CALIBRATION_PARAMS_H
#define O2_CALIBRATION_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace trd
{

/// VDrift and ExB calibration parameters.
struct TRDCalibParams : public o2::conf::ConfigurableParamHelper<TRDCalibParams> {
  unsigned int nTrackletsMin = 5;  ///< minimum amount of tracklets
  unsigned int chi2RedMax = 6;     ///< maximum reduced chi2 acceptable for track quality
  size_t minEntriesChamber = 75;   ///< minimum number of entries per chamber to fit single time slot
  size_t minEntriesTotal = 40'500; ///< minimum total required for meaningful fits

  // For gain calibration
  unsigned int nTrackletsMinGainCalib = 5;
  size_t minEntriesChamberGainCalib = 500;     ///< minimum number of entries per chamber to fit single time slot
  size_t minEntriesTotalGainCalib = 1'000'000; ///< minimum total required for meaningful fits
  // Cuts for selecting clean pion candidates for gain calibration
  float pMin = 0.5; /// minimum p for selected tracks
  float pMax = 1.;
  float dEdxTPCMin = 30.;
  float dEdxTPCMax = 70.;

  // For t0 fits
  size_t minEntriesTotalT0Fit = 1'000'000; ///< minimum number of entries in inclusive distribution for (meaningful) t0 fit
  size_t minEntriesChamberT0Fit = 30'000;  ///< minimum number of entries in one chamber for (meaningful) t0 fit

  // Creation of PH plots
  float pileupCut = 0.7;

  // parameters related to noise calibration
  size_t minNumberOfDigits = 1'000'000'000UL; ///< when reached, noise calibration will be finalized

  // boilerplate
  O2ParamDef(TRDCalibParams, "TRDCalibParams");
};

} // namespace trd
} // namespace o2

#endif
