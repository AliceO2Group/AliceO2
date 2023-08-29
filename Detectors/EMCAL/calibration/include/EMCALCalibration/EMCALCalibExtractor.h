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
#include "EMCALCalibration/EMCALCalibParams.h"
#include "CommonUtils/BoostHistogramUtils.h"
#include "EMCALBase/Geometry.h"
#include "EMCALCalibration/EMCALCalibParams.h"
#include <boost/histogram.hpp>

#include <TRobustEstimator.h>

#if (defined(WITH_OPENMP) && !defined(__CLING__))
#include <omp.h>
#endif

namespace o2
{
namespace emcal
{
using boostHisto = boost::histogram::histogram<std::tuple<boost::histogram::axis::regular<double, boost::use_default, boost::use_default, boost::use_default>, boost::histogram::axis::integer<>>, boost::histogram::unlimited_storage<std::allocator<char>>>;
class EMCALCalibExtractor
{
  using slice_t = int;
  using cell_t = int;

  /// \brief Stuct for the maps needed for the bad channel calibration
  struct BadChannelCalibInfo {
    std::map<slice_t, std::array<double, 17664>> energyPerHitMap;        // energy/per hit per cell per slice
    std::map<slice_t, std::array<double, 17664>> nHitsMap;               // number of hits per cell per slice
    std::map<slice_t, std::pair<double, double>> goodCellWindowMap;      // for each slice, the emin and the emax of the good cell window
    std::map<slice_t, std::pair<double, double>> goodCellWindowNHitsMap; // for each slice, the nHitsMin and the mHitsMax of the good cell window
  };

  struct BadChannelCalibTimeInfo {
    std::array<double, 17664> sigmaCell; // sigma value of time distribution for single cells
    double goodCellWindow;               // cut value for good cells
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

  /// \brief Scaled hits per cell
  /// \param emin -- min. energy for cell amplitudes
  /// \param emax -- max. energy for cell amplitudes
  boostHisto buildHitAndEnergyMeanScaled(double emin, double emax, boostHisto mCellAmplitude);

  /// \brief Function to perform the calibration of bad channels
  /// \param hist histogram cell energy vs. cell ID. Main histogram for the bad channel calibration
  /// \param histTime histogram cell time vs. cell ID. If default argument is taken, no calibration based on the timing signal will be performed
  template <typename... axes>
  o2::emcal::BadChannelMap calibrateBadChannels(const boost::histogram::histogram<axes...>& hist, const boost::histogram::histogram<axes...>& histTime = boost::histogram::make_histogram(boost::histogram::axis::variable<>{0., 1.}, boost::histogram::axis::variable<>{0., 1.}))
  {
    double time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::map<int, std::pair<double, double>> slices = {{0, {0.1, 0.3}}, {1, {0.3, 0.5}}, {2, {0.5, 1.0}}, {3, {1.0, 4.0}}, {4, {4.0, 39.0}}};

    auto histScaled = hist;
    if (mBCMScaleFactors) {
      LOG(info) << "Rescaling BCM histo";
      // rescale the histogram
      for (int icell = 0; icell < 17644; icell++) {
        for (int ebin = 0; ebin < hist.axis(0).size(); ebin++) {
          double lowerE = hist.axis(0).bin(ebin).lower();
          double upperE = hist.axis(0).bin(ebin).upper();
          double midE = (lowerE + upperE) / 2.;
          histScaled.at(ebin, icell) = hist.at(ebin, icell) / mBCMScaleFactors->getScaleVal(icell, midE);
        }
      }
    }

    // get all ofthe calibration information that we need in a struct
    BadChannelCalibInfo calibrationInformation = buildHitAndEnergyMean(slices, histScaled);

    // only initialize this if the histo is not the default one
    const bool doIncludeTime = (histTime.axis(0).size() > 1 && EMCALCalibParams::Instance().useTimeInfoForCalib_bc) ? true : false;
    BadChannelCalibTimeInfo calibrationTimeInfo;
    if (doIncludeTime) {
      calibrationTimeInfo = buildTimeMeanAndSigma(histTime);
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
      LOG(debug) << "analysing cell " << cellID;
      if (calibrationInformation.energyPerHitMap[0][cellID] == 0) {
        LOG(debug) << "Cell " << cellID << " is dead.";
        mOutputBCM.addBadChannel(cellID, o2::emcal::BadChannelMap::MaskType_t::DEAD_CELL);
      } else {
        bool failed = false;
        for (auto& [sliceIndex, slice] : slices) {
          auto ranges = calibrationInformation.goodCellWindowMap[sliceIndex];
          auto rangesNHits = calibrationInformation.goodCellWindowNHitsMap[sliceIndex];
          auto meanPerCell = calibrationInformation.energyPerHitMap[sliceIndex][cellID];
          auto meanPerCellNHits = calibrationInformation.nHitsMap[sliceIndex][cellID];
          LOG(debug) << "Energy Per Hit: Mean per cell is " << meanPerCell << " Good Cell Window: [ " << ranges.first << " , " << ranges.second << " ]";
          LOG(debug) << "NHits: Mean per cell is " << meanPerCellNHits << " Good Cell Window: [ " << rangesNHits.first << " , " << rangesNHits.second << " ]";

          // for the cut on the mean number of hits we require at least 2 hits on average
          double meanNHits = 0.5 * (rangesNHits.first + rangesNHits.second);
          if (meanNHits < EMCALCalibParams::Instance().minNHitsForNHitCut || (std::abs(ranges.first) < 0.001 && std::abs(ranges.second) < 0.001)) {
            LOG(debug) << "On average, only " << meanNHits << " found in energy interval [" << slice.first << " - " << slice.second << "]. Will do untight cut on upper limit";
            if (meanPerCellNHits > EMCALCalibParams::Instance().minNHitsForNHitCut * 10) {
              LOG(debug) << "********* FAILED for number of hits **********";
              failed = true;
              break;
            }
            // Case were enough statistics is present
          } else if (meanPerCellNHits < rangesNHits.first || meanPerCellNHits > rangesNHits.second) {
            LOG(debug) << "********* FAILED for mean NHits **********";
            failed = true;
            break;
          }

          // for the cut on the mean energy per hit we require at least 100 hits, as otherwise the distribution is very instable
          if (meanNHits > EMCALCalibParams::Instance().minNHitsForMeanEnergyCut && (meanPerCell < ranges.first || meanPerCell > ranges.second)) {
            LOG(debug) << "********* FAILED for mean energy **********";
            failed = true;
            break;
          }
        }

        // check if the cell is bad due to timing signal.
        if (!failed && doIncludeTime) {
          if (std::abs(calibrationTimeInfo.goodCellWindow) < 0.001) {
            LOG(warning) << "Good cell window for time distribution is 0. Will skip the cut on time distribution";
          } else {
            LOG(debug) << " calibrationTimeInfo.goodCellWindow " << calibrationTimeInfo.goodCellWindow << " calibrationTimeInfo.sigmaCell[cellID] " << calibrationTimeInfo.sigmaCell[cellID];
            if (calibrationTimeInfo.sigmaCell[cellID] > calibrationTimeInfo.goodCellWindow) {
              LOG(debug) << "Cell " << cellID << " is flagged due to time distribution";
              failed = true;
            }
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
    std::map<slice_t, std::array<double, mNcells>> outputMapNHits;
    // initialize the output maps with 0
    for (const auto& [sliceIndex, sliceLimits] : sliceMap) {
      std::array<double, mNcells> energyPerHit, nHits;
      std::fill(energyPerHit.begin(), energyPerHit.end(), 0.);
      std::fill(nHits.begin(), nHits.end(), 0.);
      outputMapEnergyPerHit[sliceIndex] = energyPerHit;
      outputMapNHits[sliceIndex] = nHits;
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
          outputMapEnergyPerHit[sliceIndex][cellID] = meanVal;
          outputMapNHits[sliceIndex][cellID] = sumVal;
        }
      } // end loop over the slices
    }   // end loop over the cells
    for (const auto& [sliceIndex, slice] : sliceMap) {
      Double_t meanPerSlice = 0.0;  // mean energy per slice to be compared to the cell
      Double_t sigmaPerSlice = 0.0; // sigma energy per slice to be compared to the cell
      TRobustEstimator robustEstimator;
      auto& means = outputMapEnergyPerHit[sliceIndex];
      robustEstimator.EvaluateUni(means.size(), means.data(), meanPerSlice, sigmaPerSlice, 0);
      if (std::abs(meanPerSlice) < 0.001 && std::abs(sigmaPerSlice) < 0.001) {
        robustEstimator.EvaluateUni(means.size(), means.data(), meanPerSlice, sigmaPerSlice, means.size() * 0.95);
      }

      Double_t meanPerSlice_NHits = 0.0;  // mean energy per slice to be compared to the cell
      Double_t sigmaPerSlice_NHits = 0.0; // sigma energy per slice to be compared to the cell
      TRobustEstimator robustEstimatorNHits;
      auto& meansNHits = outputMapNHits[sliceIndex];
      robustEstimatorNHits.EvaluateUni(meansNHits.size(), meansNHits.data(), meanPerSlice_NHits, sigmaPerSlice_NHits, 0);
      if (std::abs(meanPerSlice_NHits) < 0.001 && std::abs(sigmaPerSlice_NHits) < 0.001) {
        robustEstimator.EvaluateUni(meansNHits.size(), meansNHits.data(), meanPerSlice_NHits, sigmaPerSlice_NHits, meansNHits.size() * 0.95);
      }

      LOG(debug) << "Energy Per hit: Mean per slice is: " << meanPerSlice << " Sigma Per Slice: " << sigmaPerSlice << " with size " << outputMapEnergyPerHit[sliceIndex].size();
      LOG(debug) << "NHits: Mean per slice is: " << meanPerSlice_NHits << " Sigma Per Slice: " << sigmaPerSlice_NHits << " with size " << outputMapNHits[sliceIndex].size();
      // calculate the "good cell window from the mean"
      double maxVal = meanPerSlice + mSigma * sigmaPerSlice;
      double minVal = meanPerSlice - mSigma * sigmaPerSlice;
      double maxValNHits = meanPerSlice_NHits + mSigma * sigmaPerSlice_NHits;
      double minValNHits = meanPerSlice_NHits - mSigma * sigmaPerSlice_NHits;
      // store in the output maps
      outputInfo.goodCellWindowMap[sliceIndex] = {minVal, maxVal};
      outputInfo.goodCellWindowNHitsMap[sliceIndex] = {minValNHits, maxValNHits};
    }
    // now add these to the calib info struct
    outputInfo.energyPerHitMap = outputMapEnergyPerHit;
    outputInfo.nHitsMap = outputMapNHits;

    return outputInfo;
  }

  //____________________________________________
  /// \brief calculate the sigma of the time distribution for all cells and caluclate the mean of the sigmas
  /// \param histCellTime input histogram cellID vs cell time
  /// \return sigma value for all cells and the upper cut value
  template <typename... axes>
  BadChannelCalibTimeInfo buildTimeMeanAndSigma(const boost::histogram::histogram<axes...>& histCellTime)
  {
    std::array<double, 17664> meanSigma;
    for (int i = 0; i < mNcells; ++i) {
      // calculate sigma per cell
      const int indexLow = histCellTime.axis(1).index(i);
      const int indexHigh = histCellTime.axis(1).index(i + 1);
      auto boostHistCellSlice = o2::utils::ProjectBoostHistoXFast(histCellTime, indexLow, indexHigh);

      int maxElementIndex = std::max_element(boostHistCellSlice.begin(), boostHistCellSlice.end()) - boostHistCellSlice.begin() - 1;
      if (maxElementIndex < 0) {
        maxElementIndex = 0;
      }
      float maxElementCenter = 0.5 * (boostHistCellSlice.axis(0).bin(maxElementIndex).upper() + boostHistCellSlice.axis(0).bin(maxElementIndex).lower());
      meanSigma[i] = std::sqrt(o2::utils::getVarianceBoost1D(boostHistCellSlice, -999999, maxElementCenter - 50, maxElementCenter + 50));
    }

    // get the mean sigma and the std. deviation of the sigma distribution
    // those will be the values we cut on
    double avMean = 0, avSigma = 0;
    TRobustEstimator robustEstimator;
    robustEstimator.EvaluateUni(meanSigma.size(), meanSigma.data(), avMean, avSigma, 0.5 * meanSigma.size());
    // protection for the following case: For low statistics cases, it can happen that more than half of the cells is in one bin
    // in that case the sigma will be close to zero. In that case, we take 95% of the data to calculate the truncated mean
    if (std::abs(avMean) < 0.001 && std::abs(avSigma) < 0.001) {
      robustEstimator.EvaluateUni(meanSigma.size(), meanSigma.data(), avMean, avSigma, 0.95 * meanSigma.size());
    }

    BadChannelCalibTimeInfo timeInfo;
    timeInfo.sigmaCell = meanSigma;
    timeInfo.goodCellWindow = avMean + (avSigma * o2::emcal::EMCALCalibParams::Instance().sigmaTime_bc); // only upper limit needed

    return timeInfo;
  }

  //____________________________________________

  /// \brief Calibrate time for all cells
  /// \param hist -- 2d boost histogram: cell-time vs. cell-ID
  /// \param minTime -- min. time considered for fit
  /// \param maxTime -- max. time considered for fit
  /// \param restrictFitRangeToMax -- restrict the fit range to the maximum entry in the histogram in the range +-restrictFitRangeToMax (default: 25ns)
  template <typename... axes>
  o2::emcal::TimeCalibrationParams calibrateTime(const boost::histogram::histogram<axes...>& hist, double minTime = 0, double maxTime = 1000, double restrictFitRangeToMax = 25)
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
      int maxElementIndex = std::max_element(boostHist1d.begin(), boostHist1d.end()) - boostHist1d.begin() - 1;
      if (maxElementIndex < 0) {
        maxElementIndex = 0;
      }
      float maxElementCenter = 0.5 * (boostHist1d.axis(0).bin(maxElementIndex).upper() + boostHist1d.axis(0).bin(maxElementIndex).lower());
      // Restrict fit range to maximum +- restrictFitRangeToMax
      if (restrictFitRangeToMax > 0) {
        boostHist1d = boost::histogram::algorithm::reduce(boostHist1d, boost::histogram::algorithm::shrink(maxElementCenter - restrictFitRangeToMax, maxElementCenter + restrictFitRangeToMax));
      }

      try {
        auto fitValues = o2::utils::fitBoostHistoWithGaus<double>(boostHist1d);
        if (maxElementCenter + EMCALCalibParams::Instance().maxAllowedDeviationFromMax < fitValues.at(1) || maxElementCenter - EMCALCalibParams::Instance().maxAllowedDeviationFromMax > fitValues.at(1)) {
          mean = maxElementCenter;
        } else {
          mean = fitValues.at(1);
        }
        // add mean to time calib params
        TCP.addTimeCalibParam(i, mean, false);                                                // highGain calib factor
        TCP.addTimeCalibParam(i, mean + EMCALCalibParams::Instance().lowGainOffset_tc, true); // lowGain calib factor
      } catch (o2::utils::FitGausError_t err) {
        LOG(warning) << createErrorMessageFitGaus(err) << "; for cell " << i << " (Will take the parameter of the previous cell: " << mean << "ns)";
        TCP.addTimeCalibParam(i, mean, false);                                                // take calib value of last cell; or 400 ns shift default value
        TCP.addTimeCalibParam(i, mean + EMCALCalibParams::Instance().lowGainOffset_tc, true); // take calib value of last cell; or 400 ns shift default value
      }
    }
    return TCP;
  }

 private:
  EMCALChannelScaleFactors* mBCMScaleFactors = nullptr; ///< Scale factors for nentries scaling in bad channel calibration
  int mSigma = 5;                                       ///< number of sigma used in the calibration to define outliers
  int mNThreads = 1;                                    ///< number of threads used for calibration

  o2::emcal::Geometry* mGeometry = nullptr;
  static constexpr int mNcells = 17664;

  ClassDefNV(EMCALCalibExtractor, 1);
};

} // namespace emcal
} // namespace o2
#endif