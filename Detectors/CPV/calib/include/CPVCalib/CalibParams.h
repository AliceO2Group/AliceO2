// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \class CalibParams
/// \brief CCDB container for the full set of CPV calibration coefficients
/// \author Dmitri Peresunko, RRC Kurchatov institute
/// \since Aug. 1, 2019
///
///

#ifndef CPV_CALIBPARAMS_H
#define CPV_CALIBPARAMS_H

#include <array>
#include "TObject.h"

class TH2;

namespace o2
{

namespace cpv
{

class CalibParams
{
 public:
  /// \brief Constructor
  CalibParams() = default;

  /// \brief Constructor for tests
  CalibParams(short test);

  /// \brief Destructor
  ~CalibParams() = default;

  /// \brief Get High Gain energy calibration coefficients
  /// \param cellID Absolute ID of cell
  /// \return high gain energy calibration coefficient of the cell
  float getGain(unsigned short cellID) const { return mGainCalib[cellID]; }

  /// \brief Set High Gain energy calibration coefficient
  /// \param cellID Absolute ID of cell
  /// \param c is the calibration coefficient
  void setGain(unsigned short cellID, float c) { mGainCalib[cellID] = c; }

  /// \brief Set High Gain energy calibration coefficients for one module in the form of 2D histogram
  /// \param 2D(64,56) histogram with calibration coefficients
  /// \param module number
  /// \return Is successful
  bool setGain(TH2* h, short module);

 private:
  static constexpr short NCHANNELS = 28673; ///< Number of channels starting from 1
  std::array<float, NCHANNELS> mGainCalib;  ///< Container for the gain calibration coefficients
  ClassDefNV(CalibParams, 1);
};

} // namespace cpv

} // namespace o2
#endif
