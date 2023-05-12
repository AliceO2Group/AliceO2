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

// #include <array>

namespace o2
{
namespace emcal
{
class EMCALCalibExtractor;

class EMCALChannelData
{
  //using Slot = o2::calibration::TimeSlot<o2::emcal::EMCALChannelData>;
  using Cells = o2::emcal::Cell;
  using boostHisto = boost::histogram::histogram<std::tuple<boost::histogram::axis::variable<double, boost::use_default, boost::use_default, std::allocator<double>>, boost::histogram::axis::variable<double, boost::use_default, boost::use_default, std::allocator<double>>>>;
  using BadChannelMap = o2::emcal::BadChannelMap;

 public:
  // NCELLS includes DCal, treat as one calibration
  o2::emcal::Geometry* mGeometry = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);
  int NCELLS = mGeometry->GetNCells();

  EMCALChannelData() : mNBins(EMCALCalibParams::Instance().nBinsEnergyAxis_bc), mRange(EMCALCalibParams::Instance().maxValueEnergyAxis_bc), mNBinsTime(EMCALCalibParams::Instance().nBinsTimeAxis_bc), mRangeTimeLow(EMCALCalibParams::Instance().rangeTimeAxisLow_bc), mRangeTimeHigh(EMCALCalibParams::Instance().rangeTimeAxisHigh_bc)
  {

    // NCELLS includes DCal, treat as one calibration
    o2::emcal::Geometry* mGeometry = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);
    int NCELLS = mGeometry->GetNCells();

    // boost histogram with amplitude vs. cell ID, specify the range and binning of the amplitude axis
    std::vector<double> binEdgesCells;
    for (int i = 0; i <= NCELLS; i++) {
      binEdgesCells.push_back(i);
    }
    std::vector<double> binEdgesEnergy;
    for (int i = 0; i <= mNBins; i++) {
      binEdgesEnergy.push_back(static_cast<double>(i) * mRange / mNBins);
    }
    std::vector<double> binEdgesTime;
    for (int i = 0; i <= mNBinsTime; i++) {
      binEdgesTime.push_back(mRangeTimeLow + (static_cast<double>(i) * std::abs(mRangeTimeHigh - mRangeTimeLow)) / mNBinsTime);
    }

    mHisto = boost::histogram::make_histogram(boost::histogram::axis::variable<>(binEdgesEnergy), boost::histogram::axis::variable<>(binEdgesCells));
    mHistoTime = boost::histogram::make_histogram(boost::histogram::axis::variable<>(binEdgesTime), boost::histogram::axis::variable<>(binEdgesCells));
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
  void merge(const EMCALChannelData* prev);
  // int findBin(float v) const;
  /// \brief Check if enough stataistics was accumulated to perform calibration
  bool hasEnoughData() const;
  /// \brief Get current calibration histogram
  boostHisto& getHisto() { return mHisto; }
  const boostHisto& getHisto() const { return mHisto; }

  /// \brief Set new calibration histogram
  void setHisto(boostHisto hist) { mHisto = hist; }

  /// \brief Get current calibration histogram with timing information
  boostHisto& getHistoTime() { return mHistoTime; }
  const boostHisto& getHistoTime() const { return mHistoTime; }

  /// \brief Set new calibration histogram with timing info
  void setHistoTime(boostHisto hist) { mHistoTime = hist; }

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

 private:
  float mRange = 10;                                    ///< Maximum energy range of boost histogram (will be overwritten by values in the EMCALCalibParams)
  int mNBins = 1000;                                    ///< Number of bins in the boost histogram (will be overwritten by values in the EMCALCalibParams)
  boostHisto mHisto;                                    ///< 2d boost histogram with cellID vs cell energy
  int mNBinsTime = 1000;                                ///< Number of time bins in boost histogram (cell time vs. cell ID)
  float mRangeTimeLow = -500;                           ///< lower bound of time axis of mHistoTime
  float mRangeTimeHigh = 500;                           ///< upper bound of time axis of mHistoTime
  boostHisto mHistoTime;                                ///< 2d boost histogram with cellID vs cell time
  int mEvents = 0;                                      ///< event counter
  long unsigned int mNEntriesInHisto = 0;               ///< Number of entries in the histogram
  boostHisto mEsumHisto;                                ///< contains the average energy per hit for each cell
  boostHisto mEsumHistoScaled;                          ///< contains the average energy (scaled) per hit for each cell
  boostHisto mCellAmplitude;                            ///< is the input for the calibration, hist of cell E vs. ID
  bool mTest = false;                                   ///< flag to be used when running in test mode: it simplify the processing
  BadChannelMap mOutputBCM;                             ///< output bad channel map for the calibration
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
