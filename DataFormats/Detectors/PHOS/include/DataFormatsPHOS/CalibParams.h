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

  /// \brief Constructor for tests
  CalibParams(CalibParams& a) = default;

  CalibParams& operator=(const CalibParams& other) = default;

  /// \brief Destructor
  ~CalibParams() = default;

  /// \brief Get High Gain energy calibration coefficients
  /// \param cellID Absolute ID of cell
  /// \return high gain energy calibration coefficient of the cell
  float getGain(short cellID) const { return mGainCalib.at(cellID - OFFSET); }

  /// \brief Set High Gain energy calibration coefficient
  /// \param cellID Absolute ID of cell
  /// \param c is the calibration coefficient
  void setGain(short cellID, float c) { mGainCalib.at(cellID - OFFSET) = c; }

  /// \brief Set High Gain energy calibration coefficients for one module in the form of 2D histogram
  /// \param 2D(64,56) histogram with calibration coefficients
  /// \param module number
  /// \return Is successful
  bool setGain(TH2* h, int8_t module);

  /// \brief Get High Gain to Low Gain ratio calibration coefficients
  /// \param cellID Absolute ID of cell
  /// \return High Gain to Low Gain ratio of the cell
  [[nodiscard]] float getHGLGRatio(short cellID) const { return mHGLGRatio.at(cellID - OFFSET); }

  /// \brief Set High Gain to Low Gain ratio
  /// \param cellID Absolute ID of cell
  /// \param r is the calibration coefficient
  void setHGLGRatio(short cellID, float r) { mHGLGRatio.at(cellID - OFFSET) = r; }

  /// \brief Set High Gain to Low Gain ratio for one module in the form of 2D histogram
  /// \param 2D(64,56) histogram with High Gain to Low Gain ratio
  /// \param module number
  /// \return Is successful
  bool setHGLGRatio(TH2* h, int8_t module);

  /// \brief Get High Gain time calibration coefficients
  /// \param cellID Absolute ID of cell
  /// \return high gain time calibration coefficient of the cell
  [[nodiscard]] float getHGTimeCalib(short cellID) const { return mHGTimeCalib.at(cellID - OFFSET); }

  /// \brief Set High Gain time calibration coefficient
  /// \param cellID Absolute ID of cell
  /// \param t is the calibration coefficient
  void setHGTimeCalib(short cellID, float t) { mHGTimeCalib.at(cellID - OFFSET) = t; }

  /// \brief Set High Gain time calibration coefficients for one module in the form of 2D histogram
  /// \param 2D(64,56) histogram with calibration coefficients
  /// \param module number
  /// \return Is successful
  bool setHGTimeCalib(TH2* h, int8_t module);

  /// \brief Get Low Gain time calibration coefficient
  /// \param cellID Absolute ID of cell
  /// \return low gain time calibration coefficient of the cell
  [[nodiscard]] float getLGTimeCalib(short cellID) const { return mLGTimeCalib.at(cellID - OFFSET); }

  /// \brief Set time calibration coefficient
  /// \param cellID Absolute ID of cell
  /// \param t is the calibration coefficient
  void setLGTimeCalib(short cellID, float t) { mLGTimeCalib.at(cellID - OFFSET) = t; }

  /// \brief Set Low Gain time calibration coefficients for one module in the form of 2D histogram
  /// \param 2D(64,56) histogram with calibration coefficients
  /// \param module number
  /// \return Is successful
  bool setLGTimeCalib(TH2* h, int8_t module);

 private:
  static constexpr short NCHANNELS = 12544;  ///< Number of channels = 14336-1792
  static constexpr short OFFSET = 1793;      ///< Non-existing channels 56*64*0.5+1
  std::array<float, NCHANNELS> mGainCalib;   ///< Container for the gain calibration coefficients
  std::array<float, NCHANNELS> mHGLGRatio;   ///< Container for the High Gain to Low Gain ratios
  std::array<float, NCHANNELS> mHGTimeCalib; ///< Container for the High Gain time calibration coefficients
  std::array<float, NCHANNELS> mLGTimeCalib; ///< Container for the Low Gain time calibration coefficients

  ClassDefNV(CalibParams, 1);
};

} // namespace phos

} // namespace o2
#endif
