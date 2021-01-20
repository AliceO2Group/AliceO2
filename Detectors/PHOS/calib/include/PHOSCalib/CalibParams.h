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
/// \brief CCDB container for the full set of PHOS calibration coefficients
/// \author Dmitri Peresunko, RRC Kurchatov institute
/// \since Aug. 1, 2019
///
///

#ifndef PHOS_CALIBPARAMS_H
#define PHOS_CALIBPARAMS_H

#include <array>
#include "TObject.h"

class TH2;

namespace o2
{

namespace phos
{

class CalibParams
{
 public:
  /// \brief Constructor
  CalibParams() = default;

  /// \brief Constructor for tests
  CalibParams(int test);

  /// \brief Destructor
  ~CalibParams() = default;

  /// \brief Get High Gain energy calibration coefficients
  /// \param cellID Absolute ID of cell
  /// \return high gain energy calibration coefficient of the cell
  float getGain(short cellID) const { return mGainCalib.at(cellID - OFFSET); }

  /// \brief Set High Gain energy calibration coefficient
  /// \param cellID Absolute ID of cell
  /// \param c is the calibration coefficient
  void setGain(short cellID, float c) { mGainCalib[cellID - OFFSET] = c; }

  /// \brief Set High Gain energy calibration coefficients for one module in the form of 2D histogram
  /// \param 2D(64,56) histogram with calibration coefficients
  /// \param module number
  /// \return Is successful
  bool setGain(TH2* h, char module);

  /// \brief Get High Gain to Low Gain ratio calibration coefficients
  /// \param cellID Absolute ID of cell
  /// \return High Gain to Low Gain ratio of the cell
  float getHGLGRatio(short cellID) const { return mHGLGRatio.at(cellID - OFFSET); }

  /// \brief Set High Gain to Low Gain ratio
  /// \param cellID Absolute ID of cell
  /// \param r is the calibration coefficient
  void setHGLGRatio(short cellID, float r) { mHGLGRatio[cellID - OFFSET] = r; }

  /// \brief Set High Gain to Low Gain ratio for one module in the form of 2D histogram
  /// \param 2D(64,56) histogram with High Gain to Low Gain ratio
  /// \param module number
  /// \return Is successful
  bool setHGLGRatio(TH2* h, char module);

  /// \brief Get High Gain time calibration coefficients
  /// \param cellID Absolute ID of cell
  /// \return high gain time calibration coefficient of the cell
  float getHGTimeCalib(short cellID) const { return mHGTimeCalib.at(cellID - OFFSET); }

  /// \brief Set High Gain time calibration coefficient
  /// \param cellID Absolute ID of cell
  /// \param t is the calibration coefficient
  void setHGTimeCalib(short cellID, float t) { mHGTimeCalib[cellID - OFFSET] = t; }

  /// \brief Set High Gain time calibration coefficients for one module in the form of 2D histogram
  /// \param 2D(64,56) histogram with calibration coefficients
  /// \param module number
  /// \return Is successful
  bool setHGTimeCalib(TH2* h, char module);

  /// \brief Get Low Gain time calibration coefficient
  /// \param cellID Absolute ID of cell
  /// \return low gain time calibration coefficient of the cell
  float getLGTimeCalib(short cellID) const { return mLGTimeCalib.at(cellID - OFFSET); }

  /// \brief Set time calibration coefficient
  /// \param cellID Absolute ID of cell
  /// \param t is the calibration coefficient
  void setLGTimeCalib(short cellID, float t) { mLGTimeCalib[cellID - OFFSET] = t; }

  /// \brief Set Low Gain time calibration coefficients for one module in the form of 2D histogram
  /// \param 2D(64,56) histogram with calibration coefficients
  /// \param module number
  /// \return Is successful
  bool setLGTimeCalib(TH2* h, char module);

 private:
  static constexpr short NCHANNELS = 14337;  ///< Number of channels starting from 1
  static constexpr short OFFSET = 5377;      ///< Non-existing channels 56*64*1.5+1
  std::array<float, NCHANNELS> mGainCalib;   ///< Container for the gain calibration coefficients
  std::array<float, NCHANNELS> mHGLGRatio;   ///< Container for the High Gain to Low Gain ratios
  std::array<float, NCHANNELS> mHGTimeCalib; ///< Container for the High Gain time calibration coefficients
  std::array<float, NCHANNELS> mLGTimeCalib; ///< Container for the Low Gain time calibration coefficients

  ClassDefNV(CalibParams, 1);
};

} // namespace phos

} // namespace o2
#endif
