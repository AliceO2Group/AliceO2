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

/// \class TimeCalibrationSlewingParams
/// \brief CCDB container for the time calibration slewing coefficients
/// \ingroup EMCALcalib
/// \author Joshua Konig <joshua.konig@cern.ch>, Goethe-University Frankfurt
/// \since Apr 5th, 2023

#ifndef TIMECALIBRATIONSLEWINGPARAMS_H_
#define TIMECALIBRATIONSLEWINGPARAMS_H_

#include <array>
#include <cmath>
#include <numeric>
#include <cfloat>
#include <Rtypes.h>

namespace o2
{

namespace emcal
{

class TimeCalibrationSlewingParams
{

 public:
  /// \brief Constructor
  TimeCalibrationSlewingParams() = default;

  /// \brief Constructor
  /// \param arr parameters of third oder polynomial function
  TimeCalibrationSlewingParams(std::array<double, 4> arr);

  /// \brief Destructor
  ~TimeCalibrationSlewingParams() = default;

  /// \brief Comparison of two time calibration coefficients
  /// \return true if the two list of time calibration coefficients are the same, false otherwise
  bool operator==(const TimeCalibrationSlewingParams& other) const;

  /// \brief Add time calibration slewing function to the container
  /// \param arr parameter for 3rd order polynomial function
  void addTimeSlewingParam(const std::array<double, 4> arr);

  /// \brief Get time calibration slewing parameter index of function
  /// \param index index for parameter
  double getTimeSlewingParam(const unsigned int index) const;

  /// \brief Get value of time calib slewing for a given energy
  /// \param energy energy of cell
  double eval(const double energy) const;

 private:
  std::array<double, 4> arrParams; ///< parameter for 3rd order polynomial

  // ClassDefOverride(TimeCalibrationSlewingParams, 1);
  ClassDefNV(TimeCalibrationSlewingParams, 1);
};

} // namespace emcal

} // namespace o2
#endif // TIMECALIBRATIONSLEWINGPARAMS_H_
