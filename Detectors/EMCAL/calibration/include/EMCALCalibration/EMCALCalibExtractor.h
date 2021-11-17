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

/// \class EMCALCalibExtractor
/// \brief  Use the EMCal Calibration
/// \author Hannah Bossi, Yale University
/// \ingroup EMCALCalib
/// \since Oct 3, 2021

#ifndef EMCALCALIBEXTRACTOR_H_
#define EMCALCALIBEXTRACTOR_H_

#include <iostream>
#include "CCDB/BasicCCDBManager.h"
#include "EMCALCalib/BadChannelMap.h"
#include "EMCALCalib/TimeCalibrationParams.h"
#include "CommonUtils/BoostHistogramUtils.h"
#include <boost/histogram.hpp>

namespace o2
{
namespace emcal
{

class EMCALCalibExtractor
{

 public:
  EMCALCalibExtractor() = default;
  ~EMCALCalibExtractor() = default;

  int getNsigma() const { return mSigma; }
  void setNsigma(int ns) { mSigma = ns; }

  void setUseScaledHistoForBadChannels(bool useScaledHistoForBadChannels) { mUseScaledHistoForBadChannels = useScaledHistoForBadChannels; }

  /// \brief Function to perform the calibration of bad channels
  template <typename... axes>
  o2::emcal::BadChannelMap calibrateBadChannels(boost::histogram::histogram<axes...>& hist)
  {
    // calculate the mean and sigma of the mEsumHisto
    auto fitValues = o2::utils::fitBoostHistoWithGaus<double>(hist);
    double mean = fitValues.at(1);
    double sigma = fitValues.at(2);

    // calculate the "good cell window from the mean"
    double maxVal = mean + mSigma * sigma;
    double minVal = mean - mSigma * sigma;
    o2::emcal::BadChannelMap mOutputBCM;
    // now loop through the cells and determine the mask for a given cell
    for (int cellID = 0; cellID < 17664; cellID++) {
      double E = hist.at(cellID);
      // if in the good window, mark the cell as good
      // for now we won't do warm cells - the definition of this is unclear.
      if (E == 0) {
        mOutputBCM.addBadChannel(cellID, o2::emcal::BadChannelMap::MaskType_t::DEAD_CELL);
      } else if (E < maxVal || E > minVal) {
        mOutputBCM.addBadChannel(cellID, o2::emcal::BadChannelMap::MaskType_t::GOOD_CELL);
      } else {
        mOutputBCM.addBadChannel(cellID, o2::emcal::BadChannelMap::MaskType_t::BAD_CELL);
      }
    }

    return mOutputBCM;
  }

  /// \brief For now a dummy function to calibrate the time.
  template <typename... axes>
  o2::emcal::TimeCalibrationParams calibrateTime(boost::histogram::histogram<axes...>& hist)
  {
    o2::emcal::TimeCalibrationParams TCP;
    TCP.addTimeCalibParam(1234, 600, 0);
    return TCP;
  }

 private:
  bool mUseScaledHistoForBadChannels = false; ///< variable to specify whether or not we want to use the scaled histo for the claibration of bad channels.
  int mSigma = 4;                             ///< number of sigma used in the calibration to define outliers

  ClassDefNV(EMCALCalibExtractor, 1);
};
} // namespace emcal
} // namespace o2
#endif