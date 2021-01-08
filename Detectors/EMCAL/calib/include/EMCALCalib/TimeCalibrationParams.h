// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \class TimeCalibrationParams
/// \brief CCDB container for the time calibration coefficients
/// \ingroup EMCALcalib
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since July 16th, 2019
///
/// The time calibration coefficienct can be added for each channel by
/// ~~~.{cxx}
/// o2::emcal::TimeCalibrationParams TCP;
/// TCP.addTimeCalibParam(1234, 600, 0);
/// For the High Gain cells the last parameter should be set to 0, for low
/// gain it should be set to 1.
/// ~~~
///
/// One can read the time calibration coefficient by calling
/// ~~~.{cxx}
/// auto param = TCP.getTimeCalibParam(1234, 0);
/// This will return the time calibration coefficient for a certain HG cell.
/// For low gain cells you have to set the last parameter 1
/// ~~~
///

#ifndef TIMECALIBRATIONPARAMS_H_
#define TIMECALIBRATIONPARAMS_H_

#include <iosfwd>
#include <array>
#include <Rtypes.h>

class TH1;

namespace o2
{

namespace emcal
{

class TimeCalibrationParams
{
 public:
  /// \brief Constructor
  TimeCalibrationParams() = default;

  /// \brief Destructor
  ~TimeCalibrationParams() = default;

  /// \brief Comparison of two time calibration coefficients
  /// \return true if the two list of time calibration coefficients are the same, false otherwise
  bool operator==(const TimeCalibrationParams& other) const;

  /// \brief Add time calibration coefficients to the container
  /// \param cellID Absolute ID of cell
  /// \param time is the calibration coefficient
  /// \param isLowGain is flag whether this cell is LG or HG
  void addTimeCalibParam(unsigned short cellID, short time, bool isLowGain);

  /// \brief Get the time calibration coefficient for a certain cell
  /// \param cellID Absolute ID of cell
  /// \param isLowGain is flag whether this cell is LG or HG
  /// \return time calibration coefficient of the cell
  short getTimeCalibParam(unsigned short cellID, bool isLowGain) const;

  /// \brief Convert the time calibration coefficient array to a histogram
  /// \param isLowGain is flag whether to draw for LG or HG
  /// \return Histogram representation for time calibration coefficient
  TH1* getHistogramRepresentation(bool isLowGain) const;

 private:
  std::array<short, 17664> mTimeCalibParamsHG; ///< Container for the time calibration coefficient for the High Gain cells
  std::array<short, 17664> mTimeCalibParamsLG; ///< Container for the time calibration coefficient for the Low Gain cells

  ClassDefNV(TimeCalibrationParams, 1);
};

} // namespace emcal

} // namespace o2
#endif
