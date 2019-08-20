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

#ifndef CALIBPARAMS_H
#define CALIBPARAMS_H

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

  /// \brief Destructor
  ~CalibParams() = default;

  /// \brief Get High Gain energy calibration coefficients
  /// \param cellID Absolute ID of cell
  /// \return high gain energy calibration coefficient of the cell
  float getGain(unsigned short cellID) { return mGainCalib.at(cellID); }

  /// \brief Set High Gain energy calibration coefficient
  /// \param cellID Absolute ID of cell
  /// \param c is the calibration coefficient
  void setGain(unsigned short cellID, float c) { mGainCalib[cellID] = c; }

  /// \brief Set High Gain energy calibration coefficients for one module in the form of 2D histogram
  /// \param 2D(64,56) histogram with calibration coefficients
  /// \param module number
  /// \return Is successful
  bool setGain(TH2* h, int module);

  /// \brief Get High Gain to Low Gain ratio calibration coefficients
  /// \param cellID Absolute ID of cell
  /// \return High Gain to Low Gain ratio of the cell
  float getHGLGRatio(unsigned short cellID) { return mHGLGRatio.at(cellID); }

  /// \brief Set High Gain to Low Gain ratio
  /// \param cellID Absolute ID of cell
  /// \param r is the calibration coefficient
  void setHGLGRatio(unsigned short cellID, float r) { mHGLGRatio[cellID] = r; }

  /// \brief Set High Gain to Low Gain ratio for one module in the form of 2D histogram
  /// \param 2D(64,56) histogram with High Gain to Low Gain ratio
  /// \param module number
  /// \return Is successful
  bool setHGLGRatio(TH2* h, int module);

  /// \brief Get High Gain time calibration coefficients
  /// \param cellID Absolute ID of cell
  /// \return high gain time calibration coefficient of the cell
  float getHGTimeCalib(unsigned short cellID) { return mHGTimeCalib.at(cellID); }

  /// \brief Set High Gain time calibration coefficient
  /// \param cellID Absolute ID of cell
  /// \param t is the calibration coefficient
  void setHGTimeCalib(unsigned short cellID, float t) { mHGTimeCalib[cellID] = t; }

  /// \brief Set High Gain time calibration coefficients for one module in the form of 2D histogram
  /// \param 2D(64,56) histogram with calibration coefficients
  /// \param module number
  /// \return Is successful
  bool setHGTimeCalib(TH2* h, int module);

  /// \brief Get Low Gain time calibration coefficient
  /// \param cellID Absolute ID of cell
  /// \return low gain time calibration coefficient of the cell
  float getLGTimeCalib(unsigned short cellID) { return mLGTimeCalib.at(cellID); }

  /// \brief Set time calibration coefficient
  /// \param cellID Absolute ID of cell
  /// \param t is the calibration coefficient
  void setLGTimeCalib(unsigned short cellID, float t) { mLGTimeCalib[cellID] = t; }

  /// \brief Set Low Gain time calibration coefficients for one module in the form of 2D histogram
  /// \param 2D(64,56) histogram with calibration coefficients
  /// \param module number
  /// \return Is successful
  bool setLGTimeCalib(TH2* h, int module);

 private:
  static constexpr int NCHANNELS = 14337;    ///< Number of channels starting from 1
  std::array<float, NCHANNELS> mGainCalib;   ///< Container for the gain calibration coefficients
  std::array<float, NCHANNELS> mHGLGRatio;   ///< Container for the High Gain to Low Gain ratios
  std::array<float, NCHANNELS> mHGTimeCalib; ///< Container for the High Gain time calibration coefficients
  std::array<float, NCHANNELS> mLGTimeCalib; ///< Container for the Low Gain time calibration coefficients

  ClassDefNV(CalibParams, 1);
};

} // namespace phos

} // namespace o2
#endif
