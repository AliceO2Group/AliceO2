// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \class TempCalibrationParams
/// \brief CCDB container for the temperature calibration coefficients
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since July 16th, 2019
///
/// The temperature calibration coefficienct can be added for each channel by
/// ~~~.{cxx}
/// o2::emcal::TempCalibrationParams TCP;
/// TCP.addTempCalibParam(1234, x, y);
/// The temperature calibration coefficiencts are the slope and A0 param.
/// ~~~
///
/// One can read the temperature calibration coefficient by calling
/// ~~~.{cxx}
/// auto param = TCP.getTempCalibParamSlope(1234);
/// This will return the slope for a certain cell.
/// and
/// auto param = TCP.getTempCalibParamA0(1234);
/// This will return the A0 param for a certain cell.
/// ~~~
///

#ifndef TEMPCALIBRATIONPARAMS_H_
#define TEMPCALIBRATIONPARAMS_H_

#include <iosfwd>
#include <array>
#include <Rtypes.h>

class TH1;

namespace o2
{

namespace emcal
{

class TempCalibrationParams
{
 public:
  /// \brief Constructor
  TempCalibrationParams() = default;

  /// \brief Destructor
  ~TempCalibrationParams() = default;

  /// \brief Comparison of two temperature calibration coefficients
  /// \return true if the two list of Temp calibration coefficients are the same, false otherwise
  bool operator==(const TempCalibrationParams& other) const;

  /// \brief Add temperature calibration coefficients to the container
  /// \param cellID Absolute ID of cell
  /// \param Slope is the slope coefficient
  /// \param ParamA0 is the A0 param
  void addTempCalibParam(unsigned short cellID, float Slope, float ParamA0);

  /// \brief Get the temperature calibration coefficient (slope) for a certain cell
  /// \param cellID Absolute ID of cell
  /// \return slope of the cell
  float getTempCalibParamSlope(unsigned short cellID) const;

  /// \brief Get the temperature calibration coefficient (A0 param) for a certain cell
  /// \param cellID Absolute ID of cell
  /// \return A0 param of the cell
  float getTempCalibParamA0(unsigned short cellID) const;

  /// \brief Convert the temperature calibration coefficient arrays to histograms
  /// \return Histogram representation for temperature calibration coefficient
  TH1* getHistogramRepresentationSlope() const;
  TH1* getHistogramRepresentationA0() const;

 private:
  std::array<float, 17664> mTempCalibParamsSlope; ///< Temp calibration: Slope
  std::array<float, 17664> mTempCalibParamsA0;    ///< Temp calibration coefficient: A0 param

  ClassDefNV(TempCalibrationParams, 1);
};

} // namespace emcal

} // namespace o2
#endif
