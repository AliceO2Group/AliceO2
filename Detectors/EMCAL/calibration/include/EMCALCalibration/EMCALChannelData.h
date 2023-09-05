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

/// \class EMCALChannelData
/// \brief  Perform the EMCAL bad channel calibration
/// \author Hannah Bossi, Yale University
/// \ingroup EMCALCalib
/// \since Feb 11, 2021

#ifndef CHANNEL_DATA_H_
#define CHANNEL_DATA_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
// #include "CommonUtils/BoostHistogramUtils.h"
// #include "EMCALCalib/BadChannelMap.h"
// #include "EMCALCalib/TimeCalibrationParams.h"
#include "EMCALCalib/GainCalibrationFactors.h"
#include "EMCALCalib/EMCALChannelScaleFactors.h"
#include "EMCALCalibration/EMCALCalibExtractor.h"
#include "EMCALCalibration/EMCALCalibParams.h"
#include "DataFormatsEMCAL/Cell.h"
#include "EMCALBase/Geometry.h"
// #include "CCDB/CcdbObjectInfo.h"

// #include "Framework/Logger.h"
// #include "CommonUtils/MemFileHelper.h"
// #include "CCDB/CcdbApi.h"
// #include "DetectorsCalibration/Utils.h"
#include <boost/histogram.hpp>
#include <boost/histogram/ostream.hpp>
// #include <boost/format.hpp>

#include <thread>

// #include <array>

namespace o2
{
namespace emcal
{
class EMCALCalibExtractor;

class EMCALChannelData
{
  // using Slot = o2::calibration::TimeSlot<o2::emcal::EMCALChannelData>;
  using Cells = o2::emcal::Cell;
  using boostHisto = boost::histogram::histogram<std::tuple<boost::histogram::axis::regular<double, boost::use_default, boost::use_default, boost::use_default>, boost::histogram::axis::regular<double, boost::use_default, boost::use_default, boost::use_default>>>;
  using BadChannelMap = o2::emcal::BadChannelMap;

 public:
  // NCELLS includes DCal, treat as one calibration
  o2::emcal::Geometry* mGeometry = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);
  int NCELLS = mGeometry->GetNCells();

  EMCALChannelData() : mNBins(EMCALCalibParams::Instance().nBinsEnergyAxis_bc), mRange(EMCALCalibParams::Instance().maxValueEnergyAxis_bc), mNBinsTime(EMCALCalibParams::Instance().nBinsTimeAxis_bc), mRangeTimeLow(EMCALCalibParams::Instance().rangeTimeAxisLow_bc), mRangeTimeHigh(EMCALCalibParams::Instance().rangeTimeAxisHigh_bc), mNThreads(EMCALCalibParams::Instance().nThreads_bc)
  {

    // NCELLS includes DCal, treat as one calibration
    o2::emcal::Geometry* mGeometry = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);
    int NCELLS = mGeometry->GetNCells();

    mVecNEntriesInHisto.resize(mNThreads);
    mHisto.resize(mNThreads);
    mHistoTime.resize(mNThreads);
    for (size_t i = 0; i < mNThreads; ++i) {
      mHisto[i] = boost::histogram::make_histogram(boost::histogram::axis::regular<>(mNBins, 0., mRange), boost::histogram::axis::regular<>(NCELLS, -0.5, NCELLS - 0.5));
      mHistoTime[i] = boost::histogram::make_histogram(boost::histogram::axis::regular<>(mNBinsTime, mRangeTimeHigh, mRangeTimeLow), boost::histogram::axis::regular<>(NCELLS, -0.5, NCELLS - 0.5));
      mVecNEntriesInHisto[i] = 0;
    }
  }

  ~EMCALChannelData() = default;

  /// \brief Print relevant info for EMCALChannelData on a given stream
  /// \param stream Stream on which the info is printed on
  /// The function is called in the operator<< providing direct access
  /// to protected members. Explicit calls by the users is normally not
  /// necessary.
  void PrintStream(std::ostream& stream) const;
  /// \brief Print a useful message about the container.
  void print();
  /// \brief Fill the container with the cell ID and amplitude.
  void fill(const gsl::span<const o2::emcal::Cell> data);
  /// \brief Merge the data of two slots.
  void merge(EMCALChannelData* prev);
  // int findBin(float v) const;
  /// \brief Check if enough stataistics was accumulated to perform calibration
  bool hasEnoughData() const;
  /// \brief Get current calibration histogram
  const boostHisto& getHisto()
  {
    // set the summed histogram to one of the existing histograms
    mHistoSummed = mHisto[0];
    // reset the histogram
    mHistoSummed.reset();
    // Sum up all entries
    for (const auto& h : mHisto) {
      mHistoSummed += h;
    }
    return mHistoSummed;
  }

  /// \brief Set new calibration histogram
  void setHisto(boostHisto hist, int nthr = 0) { mHisto[nthr] = hist; }

  /// \brief Get current calibration histogram with time information
  const boostHisto& getHistoTime()
  {
    mHistoTimeSummed = mHistoTime[0];
    mHistoTimeSummed.reset();
    for (const auto& h : mHistoTime) {
      mHistoTimeSummed += h;
    }
    return mHistoTimeSummed;
  }

  /// \brief Set new calibration histogram with timing info
  void setHistoTime(boostHisto hist, int nthr = 0) { mHistoTime[nthr] = hist; }

  /// \brief Peform the calibration and flag the bad channel map
  /// Average energy per hit histogram is fitted with a gaussian
  /// good area is +-mSigma
  /// cells beyond that value are flagged as bad.
  void analyzeSlot();

  float getRange() const { return mRange; }
  void setRange(float r) { mRange = r; }

  int getNbins() const { return mNBins; }
  void setNbins(int nb) { mNBins = nb; }

  int getNEvents() const { return mEvents; }
  void setNEvents(int nevt) { mEvents = nevt; }

  long unsigned int getNEntriesInHisto() const { return mNEntriesInHisto; }
  void setNEntriesInHisto(long unsigned int n) { mNEntriesInHisto = n; }
  void addEntriesInHisto(long unsigned int n) { mNEntriesInHisto += n; }

  void setGainCalibFactors(o2::emcal::GainCalibrationFactors* calibFactors)
  {
    mGainCalibFactors = calibFactors;
    for (unsigned int i = 0; i < mArrGainCalibFactors.size(); ++i) {
      mArrGainCalibFactors[i] = mGainCalibFactors->getGainCalibFactors(i);
    }
    mApplyGainCalib = true;
  }

 private:
  float mRange = 10;                                    ///< Maximum energy range of boost histogram (will be overwritten by values in the EMCALCalibParams)
  int mNBins = 1000;                                    ///< Number of bins in the boost histogram (will be overwritten by values in the EMCALCalibParams)
  size_t mNThreads = 1;                                 ///< Number of threads used for filling the boost histograms
  std::vector<boostHisto> mHisto;                       ///< vector of 2d boost histogram with cellID vs cell energy
  boostHisto mHistoSummed;                              ///< summed 2d boost histogram (sum of mHisto)
  int mNBinsTime = 1000;                                ///< Number of time bins in boost histogram (cell time vs. cell ID)
  float mRangeTimeLow = -500;                           ///< lower bound of time axis of mHistoTime
  float mRangeTimeHigh = 500;                           ///< upper bound of time axis of mHistoTime
  std::vector<boostHisto> mHistoTime;                   ///< vector of 2d boost histogram with cellID vs cell time
  boostHisto mHistoTimeSummed;                          ///< Summed 2d boost histogram with cellID vs cell time
  int mEvents = 0;                                      ///< event counter
  long unsigned int mNEntriesInHisto = 0;               ///< Number of entries in the histogram
  std::vector<long unsigned int> mVecNEntriesInHisto;   ///< Number of entries in the histogram for each thread per event
  boostHisto mEsumHisto;                                ///< contains the average energy per hit for each cell
  boostHisto mEsumHistoScaled;                          ///< contains the average energy (scaled) per hit for each cell
  boostHisto mCellAmplitude;                            ///< is the input for the calibration, hist of cell E vs. ID
  bool mTest = false;                                   ///< flag to be used when running in test mode: it simplify the processing
  BadChannelMap mOutputBCM;                             ///< output bad channel map for the calibration
  bool mApplyGainCalib = false;                         ///< Switch if gain calibration is applied or not
  o2::emcal::GainCalibrationFactors* mGainCalibFactors; ///< Gain calibration factors applied to the data before filling the histograms
  std::array<double, 17664> mArrGainCalibFactors;       ///< array of gain calibration factors
  std::shared_ptr<EMCALCalibExtractor> mCalibExtractor; ///< calib extractor

  ClassDefNV(EMCALChannelData, 1);
};
/// \brief Printing EMCALChannelData on the stream
/// \param in Stream where the EMCALChannelData is printed on
/// \param emcdata EMCALChannelData to be printed
std::ostream& operator<<(std::ostream& in, const EMCALChannelData& emcdata);

} // end namespace emcal
} // end namespace o2

#endif /*CHANNEL_DATA_H_ */
