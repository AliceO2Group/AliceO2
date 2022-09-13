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

#include <algorithm>
#include <iostream>
#include "CCDB/BasicCCDBManager.h"
#include "EMCALCalib/BadChannelMap.h"
#include "EMCALCalib/EMCALChannelScaleFactors.h"
#include "EMCALCalib/TimeCalibrationParams.h"
#include "CommonUtils/BoostHistogramUtils.h"
#include "EMCALBase/Geometry.h"
#include <boost/histogram.hpp>

#include <TRobustEstimator.h>

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
  using slice_t = int;
  using cell_t = int;

  /// \brief Stuct for the maps needed for the bad channel calibration
  struct BadChannelCalibInfo {
    std::map<slice_t, std::array<double, 17664>> energyPerHitMap;   // energy/per hit per cell per slice
    std::map<slice_t, std::pair<double, double>> goodCellWindowMap; // for each slice, the emin and the emax of the good cell window
  };

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
    // mNcells = mGeometry->GetNCells();
  };
  ~EMCALCalibExtractor() = default;

  int getNsigma() const { return mSigma; }
  void setNsigma(int ns) { mSigma = ns; }

  void setNThreads(int n) { mNThreads = std::min(n, mNcells); }
  int getNThreads() const { return mNThreads; }

  void setBCMScaleFactors(EMCALChannelScaleFactors* scalefactors) { mBCMScaleFactors = scalefactors; }

  // void setUseScaledHistoForBadChannels(bool useScaledHistoForBadChannels) { mUseScaledHistoForBadChannels = useScaledHistoForBadChannels; }

  /// \brief Scaled hits per cell
  /// \param emin -- min. energy for cell amplitudes
  /// \param emax -- max. energy for cell amplitudes
  boostHisto buildHitAndEnergyMeanScaled(double emin, double emax, boostHisto mCellAmplitude);

  /// \brief Function to perform the calibration of bad channels
  template <typename... axes>
  o2::emcal::BadChannelMap calibrateBadChannels(boost::histogram::histogram<axes...>& hist)
  {
    double time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::map<int, std::pair<double, double>> slices = {{0, {0.1, 0.3}}, {1, {0.3, 0.5}}, {2, {0.5, 1.0}}, {3, {1.0, 4.0}}};

    if (mBCMScaleFactors) {
      LOG(info) << "Rescaling BCM histo";
      // rescale the histogram
      for (int icell = 0; icell < 17644; icell++) {
        for (int ebin = 0; ebin < hist.axis(0).size(); ebin++) {
          double lowerE = hist.axis(0).bin(ebin).lower();
          double upperE = hist.axis(0).bin(ebin).upper();
          double midE = (lowerE + upperE) / 2.;
          hist.at(ebin, icell) = hist.at(ebin, icell) / mBCMScaleFactors->getScaleVal(icell, midE);
        }
      }
    }

    // get all ofthe calibration information that we need in a struct
    BadChannelCalibInfo calibrationInformation = buildHitAndEnergyMean(slices, hist);

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
      if (calibrationInformation.energyPerHitMap[0][cellID] == 0) {
        LOG(debug) << "Cell " << cellID << " is dead.";
        mOutputBCM.addBadChannel(cellID, o2::emcal::BadChannelMap::MaskType_t::DEAD_CELL);
      } else {
        bool failed = false;
        for (auto& [sliceIndex, slice] : slices) {
          auto ranges = calibrationInformation.goodCellWindowMap[sliceIndex];
          auto meanPerCell = calibrationInformation.energyPerHitMap[sliceIndex][cellID];
          LOG(debug) << "Mean per cell is " << meanPerCell << " Good Cell Window: [ " << ranges.first << " , " << ranges.second << " ]";
          if (meanPerCell < ranges.first || meanPerCell > ranges.second) {
            LOG(debug) << "********* FAILED **********";
            failed = true;
            break;
          }
        }
        if (failed) {
          LOG(debug) << "Cell " << cellID << " is bad.";
          mOutputBCM.addBadChannel(cellID, o2::emcal::BadChannelMap::MaskType_t::BAD_CELL);
        } else {
          LOG(debug) << "Cell " << cellID << " is good.";
          mOutputBCM.addBadChannel(cellID, o2::emcal::BadChannelMap::MaskType_t::GOOD_CELL);
        }
      }
    }
    double time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    double diffTransfer = time2 - time1;
    LOG(info) << "Total time" << diffTransfer << " ns";

    return mOutputBCM;
  }

  //___________________________________________________________________________________________________
  /// \brief Average energy per hit is caluclated for each cell.
  /// \param sliceID -- numerical index for the slice of amplitudes
  /// \param emin -- min. energy for cell amplitudes
  /// \param emax -- max. energy for cell amplitudes
  /// \param cellAmplitude -- boost histogram for the cell ID vs. Amplitude
  template <typename... axes>
  BadChannelCalibInfo buildHitAndEnergyMean(std::map<slice_t, std::pair<double, double>> sliceMap, boost::histogram::histogram<axes...>& cellAmplitude)
  {
    // create the output histo
    BadChannelCalibInfo outputInfo;
    std::map<slice_t, std::array<double, mNcells>> outputMapEnergyPerHit;
    // initialize the output maps with 0
    for (const auto& [sliceIndex, sliceLimits] : sliceMap) {
      std::array<double, mNcells> energyPerHit, nHits;
      std::fill(energyPerHit.begin(), energyPerHit.end(), 0.);
      std::fill(nHits.begin(), nHits.end(), 0.);
      outputMapEnergyPerHit[sliceIndex] = energyPerHit;
      //outputMapNHits[sliceIndex] = nHits;
    }
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
      int binCellID = cellAmplitude.axis(1).index(cellID);
      auto projectedSlice = o2::utils::ProjectBoostHistoXFast(cellAmplitude, binCellID, binCellID + 1);
      auto projectedSum = boost::histogram::algorithm::sum(projectedSlice);
      if (projectedSum == 0) {
        continue; // check before loop if the cell is dead
      }
      for (const auto& [sliceIndex, slice] : sliceMap) {
        double emin = slice.first;
        double emax = slice.second;
        int binXLowSlice = cellAmplitude.axis(0).index(emin);
        int binXHighSlice = cellAmplitude.axis(0).index(emax);
        auto slicedHist = o2::utils::ReduceBoostHistoFastSlice1D(projectedSlice, binXLowSlice, binXHighSlice, false);
        double meanVal = o2::utils::getMeanBoost1D(slicedHist);
        double sumVal = boost::histogram::algorithm::sum(slicedHist);
        if (sumVal > 0.) {
          // fill the output map with the desired slicing etc.
          outputMapEnergyPerHit[sliceIndex][cellID] = (meanVal / (sumVal));
        }

      } // end loop over the slices
    }   // end loop over the cells
    for (const auto& [sliceIndex, slice] : sliceMap) {
      Double_t meanPerSlice = 0.0;  // mean energy per slice to be compared to the cell
      Double_t sigmaPerSlice = 0.0; // sigma energy per slice to be compared to the cell
      TRobustEstimator robustEstimator;
      auto& means = outputMapEnergyPerHit[sliceIndex];
      robustEstimator.EvaluateUni(means.size(), means.data(), meanPerSlice, sigmaPerSlice, 0);

      LOG(debug) << "Mean per slice is: " << meanPerSlice << " Sigma Per Slice: " << sigmaPerSlice << " with size " << outputMapEnergyPerHit[sliceIndex].size();
      // calculate the "good cell window from the mean"
      double maxVal = meanPerSlice + 4.0 * sigmaPerSlice;
      double minVal = meanPerSlice - 4.0 * sigmaPerSlice;
      // we need to change this
      outputInfo.goodCellWindowMap[sliceIndex] = {minVal, maxVal};
    }
    // now add these to the calib info struct
    outputInfo.energyPerHitMap = outputMapEnergyPerHit;

    return outputInfo;
  }
  //____________________________________________

  /// \brief Calibrate time for all cells
  /// \param hist -- 2d boost histogram: cell-time vs. cell-ID
  /// \param minTime -- min. time considered for fit
  /// \param maxTime -- max. time considered for fit
  /// \param restrictFitRangeToMax -- restrict the fit range to the maximum entry in the histogram in the range +-restrictFitRangeToMax (default: 25ns)
  template <typename... axes>
  o2::emcal::TimeCalibrationParams calibrateTime(boost::histogram::histogram<axes...>& hist, double minTime = 0, double maxTime = 1000, double restrictFitRangeToMax = 25)
  {

    auto histReduced = boost::histogram::algorithm::reduce(hist, boost::histogram::algorithm::shrink(minTime, maxTime), boost::histogram::algorithm::shrink(0, mNcells));

    o2::emcal::TimeCalibrationParams TCP;

    double mean = 0;

    // #if (defined(WITH_OPENMP) && !defined(__CLING__))
    //     if (mNThreads < 1) {
    //       mNThreads = std::min(omp_get_max_threads(), mNcells);
    //     }
    //     LOG(info) << "Number of threads that will be used = " << mNThreads;
    // #pragma omp parallel for num_threads(mNThreads)
    // #else
    //     LOG(info) << "OPEN MP will not be used for the time calibration";
    //     mNThreads = 1;
    // #endif

    for (unsigned int i = 0; i < mNcells; ++i) {
      // project boost histogram to 1d just for 1 cell
      const int indexLow = histReduced.axis(1).index(i);
      const int indexHigh = histReduced.axis(1).index(i + 1);
      auto boostHist1d = o2::utils::ProjectBoostHistoXFast(histReduced, indexLow, indexHigh);

      LOG(debug) << "calibrate cell time " << i << " of " << mNcells;
      // Restrict fit range to maximum +- restrictFitRangeToMax
      if (restrictFitRangeToMax > 0) {
        int maxElementIndex = std::max_element(boostHist1d.begin(), boostHist1d.end()) - boostHist1d.begin() - 1;
        if (maxElementIndex < 0) {
          maxElementIndex = 0;
        }
        float maxElementCenter = 0.5 * (boostHist1d.axis(0).bin(maxElementIndex).upper() + boostHist1d.axis(0).bin(maxElementIndex).lower());
        boostHist1d = boost::histogram::algorithm::reduce(boostHist1d, boost::histogram::algorithm::shrink(maxElementCenter - restrictFitRangeToMax, maxElementCenter + restrictFitRangeToMax));
      }

      try {
        auto fitValues = o2::utils::fitBoostHistoWithGaus<double>(boostHist1d);
        mean = fitValues.at(1);
        // add mean to time calib params
        TCP.addTimeCalibParam(i, mean, 0);
      } catch (o2::utils::FitGausError_t err) {
        LOG(warning) << createErrorMessageFitGaus(err) << "; for cell " << i << " (Will take the parameter of the previous cell: " << mean << "ns)";
        TCP.addTimeCalibParam(i, mean, 0); // take calib value of last cell; or 400 ns shift default value
      }
    }
    return TCP;
  }

 private:
  //bool mUseScaledHistoForBadChannels = false; ///< variable to specify whether or not we want to use the scaled histo for the claibration of bad channels.
  EMCALChannelScaleFactors* mBCMScaleFactors = nullptr; ///< Scale factors for nentries scaling in bad channel calibration
  int mSigma = 4;                                       ///< number of sigma used in the calibration to define outliers
  int mNThreads = 1;                                    ///< number of threads used for calibration

  o2::emcal::Geometry* mGeometry = nullptr;
  static constexpr int mNcells = 17664;

  ClassDefNV(EMCALCalibExtractor, 1);
};

} // namespace emcal
} // namespace o2
#endif