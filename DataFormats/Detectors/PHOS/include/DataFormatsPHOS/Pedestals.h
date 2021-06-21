// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \class Pedestals
/// \brief CCDB container for the full set of PHOS calibration coefficients
/// \author Dmitri Peresunko, RRC Kurchatov institute
/// \since Aug. 1, 2019
///
///

#ifndef PHOS_PEDESTALS_H
#define PHOS_PEDESTALS_H

#include <array>
#include "TObject.h"

class TH1;

namespace o2
{

namespace phos
{

class Pedestals
{
 public:
  //numbering of PHOS channels described in Geometry, repeat it here
  // module numbering:
  //  start from module 0 (non-existing), 1 (half-module), 2 (bottom),... 4(highest)
  // absId:
  // start from 1 till 5*64*56 =14336. Numbering in each module starts at bottom left and first go in z direction:
  //  56   112   3584
  //  ...  ...    ...
  //  1    57 ...3529
  //  relid[3]: (module number[0...4], iphi[1...64], iz[1...56])

  /// \brief Constructor
  Pedestals() = default;

  /// \brief Constructor for tests
  Pedestals(int test);

  Pedestals& operator=(const Pedestals& other) = default;

  /// \brief Destructor
  ~Pedestals() = default;

  /// \brief Get pedestal
  /// \param cellID Absolute ID of cell
  /// \return pedestal for the cell
  short getHGPedestal(short cellID) const { return short(mHGPedestals[cellID - OFFSET]); }

  /// \brief Set pedestal
  /// \param cellID Absolute ID of cell
  /// \param c is the pedestal (expected to be in range <254)
  void setHGPedestal(short cellID, short c) { mHGPedestals[cellID - OFFSET] = static_cast<unsigned char>(c); }

  /// \brief Get pedestal
  /// \param cellID Absolute ID of cell
  /// \return pedestal for the cell
  short getLGPedestal(short cellID) const { return short(mLGPedestals[cellID - OFFSET]); }

  /// \brief Set pedestal
  /// \param cellID Absolute ID of cell
  /// \param c is the pedestal (expected to be in range <254)
  void setLGPedestal(short cellID, short c) { mLGPedestals[cellID - OFFSET] = static_cast<unsigned char>(c); }

  /// \brief Set pedestals from 1D histogram with cell absId in x axis
  /// \param 1D(NCHANNELS) histogram with calibration coefficients
  /// \return Is successful
  bool setHGPedestals(TH1* h);

  /// \brief Set pedestals from 1D histogram with cell absId in x axis
  /// \param 1D(NCHANNELS) histogram with calibration coefficients
  /// \return Is successful
  bool setLGPedestals(TH1* h);

  /// \brief Get pedestal RMS
  /// \param cellID Absolute ID of cell
  /// \return pedestal RMS for the cell
  float getHGRMS(short cellID) const { return float(mHGRMS[cellID - OFFSET]) / RMSCOMPRESS; }

  /// \brief Set pedestal RMS
  /// \param cellID Absolute ID of cell
  /// \param c is the pedestal RMS (expected to be in range 0..5, larger values=bad channel=overflow)
  void setHGRMS(short cellID, float c) { mHGRMS[cellID - OFFSET] = static_cast<unsigned char>(c * RMSCOMPRESS); }

  /// \brief Get pedestal
  /// \param cellID Absolute ID of cell
  /// \return pedestal RMS for the LG cell
  float getLGRMS(short cellID) const { return float(mLGRMS[cellID - OFFSET]) / RMSCOMPRESS; }

  /// \brief Set LG pedestal RMS
  /// \param cellID Absolute ID of cell
  /// \param c is the pedestal RMS (expected to be in range 0..5, larger values=bad channel=overflow)
  void setLGRMS(short cellID, float c) { mLGRMS[cellID - OFFSET] = static_cast<unsigned char>(c * RMSCOMPRESS); }

 private:
  static constexpr short NCHANNELS = 12544;          ///< Number of channels = 14336-1792
  static constexpr short OFFSET = 1793;              ///< Non-existing channels 56*64*1.5+1
  static constexpr short RMSCOMPRESS = 50;           ///< Conversion to store float RMS in range ~[0..5] in uchar
  std::array<unsigned char, NCHANNELS> mHGPedestals; ///< Container for HG pedestals
  std::array<unsigned char, NCHANNELS> mLGPedestals; ///< Container for LG pedestals
  std::array<unsigned char, NCHANNELS> mHGRMS;       ///< Container for RMS of HG pedestals
  std::array<unsigned char, NCHANNELS> mLGRMS;       ///< Container for RMS of LG pedestals

  ClassDefNV(Pedestals, 3);
};

} // namespace phos

} // namespace o2
#endif
