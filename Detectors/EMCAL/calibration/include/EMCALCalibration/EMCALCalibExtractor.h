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
#include "EMCALBase/Geometry.h"
#include <boost/histogram.hpp>

#if (defined(WITH_OPENMP) && !defined(__CLING__))
#include <omp.h>
#endif

namespace o2
{
namespace emcal
{

class EMCALCalibExtractor
{
  using boostHisto = boost::histogram::histogram<std::tuple<boost::histogram::axis::regular<double, boost::use_default, boost::use_default, boost::use_default>, boost::histogram::axis::integer<>>, boost::histogram::unlimited_storage<std::allocator<char>>>;

 public:
  EMCALCalibExtractor()
  {
    LOG(info) << "initialized EMCALCalibExtractor";
    if (!mGeometry) {
      mGeometry = o2::emcal::Geometry::GetInstance();
      if (!mGeometry) {
        mGeometry = o2::emcal::Geometry::GetInstanceFromRunNumber(300000); // fallback option
      }
    }
    mNcells = mGeometry->GetNCells();
  };
  ~EMCALCalibExtractor() = default;

  int getNsigma() const { return mSigma; }
  void setNsigma(int ns) { mSigma = ns; }

  void setNThreads(int n) { mNThreads = std::min(n, mNcells); }
  int getNThreads() const { return mNThreads; }

  void setUseScaledHistoForBadChannels(bool useScaledHistoForBadChannels) { mUseScaledHistoForBadChannels = useScaledHistoForBadChannels; }

  /// \brief Average energy per hit is caluclated for each cell.
  /// \param emin -- min. energy for cell amplitudes
  /// \param emax -- max. energy for cell amplitudes
  boostHisto buildHitAndEnergyMean(double emin, double emax, boostHisto mCellAmplitude);
  /// \brief Scaled hits per cell
  /// \param emin -- min. energy for cell amplitudes
  /// \param emax -- max. energy for cell amplitudes
  boostHisto buildHitAndEnergyMeanScaled(double emin, double emax, boostHisto mCellAmplitude);

  /// \brief Function to perform the calibration of bad channels
  template <typename... axes>
  o2::emcal::BadChannelMap calibrateBadChannels(boost::histogram::histogram<axes...>& hist)
  {
    LOG(info) << "In the calibrateBadChannels function";
    std::map<int, std::pair<double, double>> slices = {{0, {0., 0.35}}, {1, {0.35, 0.5}}};
    std::map<int, std::pair<double, double>> accept_energyRange;

    for (auto& [index, slice] : slices) {
      LOG(debug) << "Before running the buildHitAndEnergyMean";
      auto histProjected = buildHitAndEnergyMean(slice.first, slice.second, hist);
      auto fitValues = o2::utils::fitBoostHistoWithGaus<double>(histProjected);
      double mean = fitValues.at(1);
      double sigma = fitValues.at(2);
      LOG(debug) << "Done calculating the mean and the sigma from the EsumHisto Mean: " << mean << " sigma: " << sigma;

      // calculate the "good cell window from the mean"
      double maxVal = mean + mSigma * sigma;
      double minVal = mean - mSigma * sigma;
      accept_energyRange[index] = {minVal, maxVal};
      LOG(info) << "Calculating the good cell window in slice " << slice.first << " , " << slice.second << " from the maxVal" << maxVal << " minVal: " << minVal;
    }

    o2::emcal::BadChannelMap mOutputBCM;
    // now loop through the cells and determine the mask for a given cell

#if (defined(WITH_OPENMP) && !defined(__CLING__))
    if (mNThreads < 1) {
      mNThreads = std::min(omp_get_max_threads(), mNcells);
    }
    LOG(info) << "Number of threads that will be used = " << mNThreads;
#pragma omp parallel for num_threads(mNThreads)
#else
    LOG(info) << "OPEN MP will not be used for the bad channel calibration";
    mNThreads = 1;
#endif

    for (int cellID = 0; cellID < mNcells; cellID++) {
      auto projected = o2::utils::ProjectBoostHistoXFast(hist, cellID, cellID);
      if (std::accumulate(projected.begin(), projected.end(), 0) == 0) {
        mOutputBCM.addBadChannel(cellID, o2::emcal::BadChannelMap::MaskType_t::DEAD_CELL);
      } else {
        bool failed = false;
        for (auto& [index, slice] : slices) {
          auto ranges = accept_energyRange[index];
          auto dummyMean = 0.3; // mean in the slice (range)
          if (dummyMean < ranges.first || dummyMean > ranges.second) {
            failed = true;
            break;
          }
        }
        if (failed) {
          mOutputBCM.addBadChannel(cellID, o2::emcal::BadChannelMap::MaskType_t::BAD_CELL);
        } else {
          mOutputBCM.addBadChannel(cellID, o2::emcal::BadChannelMap::MaskType_t::GOOD_CELL);
        }
      }
    }

    return mOutputBCM;
  }

  /// \brief Calibrate time for all cells
  /// \param hist -- 2d boost histogram: cell-time vs. cell-ID
  /// \param minTime -- min. time considered for fit
  /// \param maxTime -- max. time considered for fit
  template <typename... axes>
  o2::emcal::TimeCalibrationParams calibrateTime(boost::histogram::histogram<axes...>& hist, double minTime = 0, double maxTime = 1000, double restrictFitRangeToMax = 25)
  {

    auto histReduced = boost::histogram::algorithm::reduce(hist, boost::histogram::algorithm::shrink(minTime, maxTime), boost::histogram::algorithm::shrink(0, mNcells));

    o2::emcal::TimeCalibrationParams TCP;

    double mean = 0;

#if (defined(WITH_OPENMP) && !defined(__CLING__))
    if (mNThreads < 1) {
      mNThreads = std::min(omp_get_max_threads(), mNcells);
    }
    LOG(info) << "Number of threads that will be used = " << mNThreads;
#pragma omp parallel for num_threads(mNThreads)
#else
    LOG(info) << "OPEN MP will not be used for the time calibration";
    mNThreads = 1;
#endif

    for (unsigned int i = 0; i < mNcells; ++i) {
      // project boost histogram to 1d just for 1 cell
      auto boostHist1d = o2::utils::ProjectBoostHistoXFast(histReduced, i, i + 1);

      LOG(debug) << "calibrate cell " << i;
      // Restrict fit range to maximum +- restrictFitRangeToMax
      if (restrictFitRangeToMax > 0) {
        int maxElementIndex = std::max_element(boostHist1d.begin(), boostHist1d.end()) - boostHist1d.begin() - 1;
        if (maxElementIndex < 0) {
          maxElementIndex = 0;
        }
        float maxElementCenter = 0.5 * (boostHist1d.axis(0).bin(maxElementIndex).upper() + boostHist1d.axis(0).bin(maxElementIndex).lower());
        float timeInterval = 25; // in ns
        boostHist1d = boost::histogram::algorithm::reduce(boostHist1d, boost::histogram::algorithm::shrink(maxElementCenter - timeInterval, maxElementCenter + timeInterval));
      }

      try {
        auto fitValues = o2::utils::fitBoostHistoWithGaus<double>(boostHist1d);
        mean = fitValues.at(1);
        // add mean to time calib params
        TCP.addTimeCalibParam(i, mean, 0);
      } catch (o2::utils::FitGausError_t) {
        TCP.addTimeCalibParam(i, mean, 0); // take calib value of last cell; or 400 ns shift default value
      }
    }
    return TCP;
  }

 private:
  bool mUseScaledHistoForBadChannels = false; ///< variable to specify whether or not we want to use the scaled histo for the claibration of bad channels.
  int mSigma = 4;                             ///< number of sigma used in the calibration to define outliers
  int mNThreads = 1;                          ///< number of threads used for calibration

  o2::emcal::Geometry* mGeometry = nullptr;
  int mNcells = 17664;

  ClassDefNV(EMCALCalibExtractor, 1);
};
} // namespace emcal
} // namespace o2
#endif