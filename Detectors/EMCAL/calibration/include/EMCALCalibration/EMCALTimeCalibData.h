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

/// \class EMCALTimeCalibData
/// \brief  Perform the EMCAL time calibration
/// \author Joshua Koenig
/// \ingroup EMCALCalib
/// \since Jul 25, 2021

#ifndef EMCAL_TIME_DATA_H_
#define EMCAL_TIME_DATA_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsEMCAL/Cell.h"
#include "EMCALBase/Geometry.h"
#include "CCDB/CcdbObjectInfo.h"
#include "EMCALCalib/TimeCalibrationParams.h"
#include "EMCALCalibration/EMCALCalibParams.h"

#include "Framework/Logger.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include <boost/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/format.hpp>

#include <array>
#include <boost/histogram.hpp>
namespace o2
{
namespace emcal
{

// class containing the initialization parameters for histograms (time bins/range etc.)
struct TimeCalibInitParams {
 public:
  unsigned int mTimeBins = 1500;
  std::array<float, 2> mTimeRange = {-500., 1000.}; // time range in ns
  unsigned int mEnergyBins = 5000;
  std::array<float, 2> mEnergyRange = {0., 50.}; // energy range in GeV
};

class EMCALTimeCalibData
{
 public:
  using Cells = o2::emcal::Cell;
  using boostHisto = boost::histogram::histogram<std::tuple<boost::histogram::axis::regular<double, boost::use_default, boost::use_default, boost::use_default>, boost::histogram::axis::regular<>>, boost::histogram::unlimited_storage<std::allocator<char>>>;

  o2::emcal::Geometry* mGeometry = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);
  int NCELLS = mGeometry->GetNCells();

  EMCALTimeCalibData(const TimeCalibInitParams& par)
  {
    // boost histogram with amplitude vs. cell ID, specify the range and binning of the amplitude axis
    mTimeHisto = boost::histogram::make_histogram(boost::histogram::axis::regular<>(par.mTimeBins, par.mTimeRange.at(0), par.mTimeRange.at(1), "t (ns)"), boost::histogram::axis::regular<>(NCELLS, -0.5, NCELLS - 0.5, "CELL ID"));
    LOG(debug) << "initialize time histogram with " << NCELLS << " cells";
  }

  ~EMCALTimeCalibData() = default;

  /// \brief Fill the container with the cell ID and amplitude and time information.
  void fill(const gsl::span<const o2::emcal::Cell> data);

  /// \brief Merge the data of two slots.
  void merge(const EMCALTimeCalibData* prev);

  /// \brief Check if enough data for calibration has been accumulated
  bool hasEnoughData() const;

  /// \brief Print a useful message about the container.
  void print();

  /// \brief Set number of events available for calibration
  void setNEvents(int nevt) { mEvents = nevt; }
  /// \brief Add number of events available for calibration
  void AddEvents(int nevt) { mEvents += nevt; }
  /// \brief Get number of events currently available for calibration
  int getNEvents() const { return mEvents; }

  /// \brief Get the number of entries in histogram
  long unsigned int getNEntriesInHisto() const { return mNEntriesInHisto; }
  /// \brief Set the number of entries in histogram
  void setNEntriesInHisto(long unsigned int n) { mNEntriesInHisto = n; }

  /// \brief Get current histogram
  boostHisto& getHisto() { return mTimeHisto; }
  const boostHisto& getHisto() const { return mTimeHisto; }

  void PrintStream(std::ostream& stream) const;

  /// \brief Actual function where calibration is done. Has to be called in has enough data when enough data is there
  o2::emcal::TimeCalibrationParams process();

 private:
  boostHisto mTimeHisto;                ///< histogram with cell time vs. cell ID
  TimeCalibInitParams mTimeCalibParams; ///< initialization parameters for histogram

  int mEvents = 0;                        ///< current number of events
  long unsigned int mNEntriesInHisto = 0; ///< number of entries in histogram

  ClassDefNV(EMCALTimeCalibData, 1);
};

} // end namespace emcal
} // end namespace o2

#endif /*EMCAL_TIME_DATA_H_ */