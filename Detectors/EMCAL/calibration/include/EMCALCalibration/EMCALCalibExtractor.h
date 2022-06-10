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
    std::map<slice_t, std::map<cell_t, double>> numberOfHitsMap;    // number of hits per cell per slice
    std::map<slice_t, std::map<cell_t, double>> energyPerHitMap;    // energy/per hit per cell per slice
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
    mNcells = mGeometry->GetNCells();
  };
  ~EMCALCalibExtractor() = default;

  int getNsigma() const { return mSigma; }
  void setNsigma(int ns) { mSigma = ns; }

  void setNThreads(int n) { mNThreads = std::min(n, mNcells); }
  int getNThreads() const { return mNThreads; }

  void setUseScaledHistoForBadChannels(bool useScaledHistoForBadChannels) { mUseScaledHistoForBadChannels = useScaledHistoForBadChannels; }

  /// \brief Scaled hits per cell
  /// \param emin -- min. energy for cell amplitudes
  /// \param emax -- max. energy for cell amplitudes
  boostHisto buildHitAndEnergyMeanScaled(double emin, double emax, boostHisto mCellAmplitude);

  /// \brief Function to perform the calibration of bad channels
  template <typename... axes>
  o2::emcal::BadChannelMap calibrateBadChannels(boost::histogram::histogram<axes...>& hist)
  {
    LOG(info) << "--------------- In the calibrateBadChannels function ---------------";
    std::map<int, std::pair<double, double>> slices = {{0, {0.1, 0.3}}, {1, {0.3, 0.5}}, {2, {0.5, 1.0}}, {3, {1.0, 4.0}}, {4, {4.0, 10.0}}};

    // get all ofthe calibration information that we need in a struct
    BadChannelCalibInfo calibrationInformation = buildHitAndEnergyMean(slices, hist);
    LOG(info) << "--------------- Done with Build Energy Hit and Mean ---------------";

    o2::emcal::BadChannelMap mOutputBCM;
    // now loop through the cells and determine the mask for a given cell

    // #if (defined(WITH_OPENMP) && !defined(__CLING__))
    //     if (mNThreads < 1) {
    //       mNThreads = std::min(omp_get_max_threads(), mNcells);
    //     }
    //     LOG(info) << "Number of threads that will be used = " << mNThreads;
    // #pragma omp parallel for num_threads(mNThreads)
    // #else
    //     LOG(info) << "OPEN MP will not be used for the bad channel calibration";
    //     mNThreads = 1;
    // #endif

    LOG(info) << "beginning the loop over all the cells to add bad channels";
    for (int cellID = 0; cellID < mNcells; cellID++) {
      LOG(info) << " On cell" << cellID;
      auto projected = o2::utils::ProjectBoostHistoXFast(hist, cellID, cellID);
      if (std::accumulate(projected.begin(), projected.end(), 0) == 0) {
        LOG(info) << "Cell " << cellID << " is dead.";
        mOutputBCM.addBadChannel(cellID, o2::emcal::BadChannelMap::MaskType_t::DEAD_CELL);
      } else {
        bool failed = false;
        for (auto& [index, slice] : slices) {
          auto ranges = calibrationInformation.goodCellWindowMap[index];
          auto meanPerCell = calibrationInformation.energyPerHitMap[index][cellID];
          LOG(info) << "Mean per cell is " << meanPerCell << " Good Cell Window: [ " << ranges.first << " , " << ranges.second << " ]";
          if (meanPerCell < ranges.first || meanPerCell > ranges.second) {
            LOG(info) << "********* FAILED **********";
            failed = true;
            break;
          }
        }
        if (failed) {
          LOG(info) << "Cell " << cellID << " is bad.";
          mOutputBCM.addBadChannel(cellID, o2::emcal::BadChannelMap::MaskType_t::BAD_CELL);
        } else {
          LOG(info) << "Cell " << cellID << " is good.";
          mOutputBCM.addBadChannel(cellID, o2::emcal::BadChannelMap::MaskType_t::GOOD_CELL);
        }
      }
    }

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
    LOG(info) << "In the buildHitAndEnergyMean function";
    // create the output histo
    BadChannelCalibInfo outputInfo;
    std::map<slice_t, std::map<cell_t, double>> outputMapEnergyPerHit;
    std::map<slice_t, std::map<cell_t, double>> outputMapNHits;
    // The outline for this function goes as follows
    // (1) loop over the total number of cells
    // (2) slice the existing histogram for one cell ranging from em    in to emax
    // (3) calculate the mean of that sliced histogram using BoostHistoramUtils.h
    // (4) fill the next histogram with this value
    // #if (defined(WITH_OPENMP))
    //     if (mNThreads < 1) {
    //       mNThreads = std::min(omp_get_max_threads(), mNcells);
    //     }
    //     LOG(info) << "Number of threads that will be used = " << mNThreads;
    // #pragma omp parallel for num_threads(mNThreads)
    // #else
    //     LOG(info) << "OPEN MP will not be used for the bad channel calibration";
    //     mNThreads = 1;
    // #endif
    LOG(info) << "beginning to loop over the slices";
    for (auto& [sliceIndex, slice] : sliceMap) {
      Double_t meanValues[mNcells];
      double emin = slice.first;
      double emax = slice.second;
      LOG(info) << "starting with slice " << sliceIndex << " from [" << emin << " <-> " << emax << "] and beginning to loop over the cells";
      LOG(info) << "number of cells " << mNcells;
      for (int cellID = 0; cellID < 1; cellID++) {
        // create a slice for each cell with energies ranging from emin to emax
        //LOG(info) << "on cell " << cellID;

        auto binYLow = cellAmplitude.axis(1).index(cellID);
        auto binYHigh = cellAmplitude.axis(1).index(cellID);
        auto binXLow = cellAmplitude.axis(0).index(emin);
        auto binXHigh = cellAmplitude.axis(0).index(emax);
        LOG(info) << "Creating a slice from x bins (" << binXLow << " , "
                  << binXHigh << ") and y bins (" << binYLow << " , " << binYHigh
                  << ")";
        auto tempSlice = o2::utils::ReduceBoostHistoFastSlice(cellAmplitude, binXLow, binXHigh, binYLow, binYHigh, false);
        LOG(info) << " after creating a temporary slice with an integral of " << boost::histogram::algorithm::sum(tempSlice);
        LOG(info) << " -------------- Starting to Project -----------";
        auto projectedSlice = o2::utils::ProjectBoostHistoXFast(tempSlice, 0, tempSlice.axis(1).size());
        // project this slice using joshua's function
        LOG(info) << "Making a " << projectedSlice.rank() << "D projection with "
                  << projectedSlice.axis(0).size() << " bins in X and an integral of "
                  << boost::histogram::algorithm::sum(projectedSlice);
        LOG(info) << " -------------- Done Projecting -----------" << std::endl;
        for (int h = 0; h < projectedSlice.axis(0).size(); h++) {
          LOG(info) << " Projected hist at bin " << h
                    << " Lower: " << projectedSlice.axis(0).bin(h).lower()
                    << " <-> Upper: " << projectedSlice.axis(0).bin(h).upper()
                    << " With value of " << projectedSlice.at(h);
        }
        LOG(info) << " after projecting the slice this has an integral of " << boost::histogram::algorithm::sum(projectedSlice);
        LOG(info) << " starting to do the mean";
        // calculate the geometric mean of the slice
        double meanVal = o2::utils::getMeanBoost1D(projectedSlice);
        LOG(info) << " calculating the mean to be " << meanVal;
        double sumVal = boost::histogram::algorithm::sum(projectedSlice);
        LOG(info) << " mean of the slice is  " << meanVal << " and the integral of the slice is " << sumVal;
        //..Set the values only for cells that are not yet marked as bad
        if (sumVal > 0.) {
#if (defined(WITH_OPENMP))
#pragma omp critical
#endif
          // fill the output map with the desired slicing etc.
          LOG(info) << "Filling mean values with " << (meanVal / (sumVal)) << " with Nhits " << sumVal;
          meanValues[cellID] = (meanVal / (sumVal));                        // average energy per hit for the mean calculation
          outputMapEnergyPerHit[sliceIndex][cellID] = (meanVal / (sumVal)); //..average energy per hit
          outputMapNHits[sliceIndex][cellID] = sumVal;                      //..number of hits
        }
        LOG(info) << "now this cell (" << cellID << ") is done, moving onto the next one";
      } // end loop over the cells

      // get the mean per slice using EvaluateUni from the map
      Double_t meanPerSlice; // mean energy per slice to be compared to the cell
      Double_t sigmaPerSlice;
      //   // create the estimator whihc we will then use
      TRobustEstimator robustEstimator;
      robustEstimator.EvaluateUni(sizeof(meanValues) / sizeof(double), meanValues, meanPerSlice, sigmaPerSlice, 0); // mean in the slice

      LOG(info) << "Mean per slice is: " << meanPerSlice << " Sigma Per Slice: " << sigmaPerSlice << " with size " << outputMapEnergyPerHit[sliceIndex].size();
      // calculate the "good cell window from the mean"
      double maxVal = meanPerSlice + mSigma * sigmaPerSlice;
      double minVal = meanPerSlice - mSigma * sigmaPerSlice;
      // we need to change this
      outputInfo.goodCellWindowMap[sliceIndex] = {minVal, maxVal};
    } // end loop over the slices
    // now add these to the calib info struct
    outputInfo.energyPerHitMap = outputMapEnergyPerHit;
    outputInfo.numberOfHitsMap = outputMapNHits;

    return outputInfo;
  }
  //____________________________________________

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