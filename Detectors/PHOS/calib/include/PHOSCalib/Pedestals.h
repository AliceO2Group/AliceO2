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
  // start from 1 till 5*64*56. Numbering in each module starts at bottom left and first go in z direction:
  //  56   112   3584
  //  ...  ...    ...
  //  1    57 ...3529
  //  relid[3]: (module number[0...4], iphi[1...64], iz[1...56])

  /// \brief Constructor
  Pedestals() = default;

  /// \brief Constructor for tests
  Pedestals(int test);

  /// \brief Destructor
  ~Pedestals() = default;

  /// \brief Get pedestal
  /// \param cellID Absolute ID of cell
  /// \return pedestal for the cell
  short getHGPedestal(short cellID) const { return short(mHGPedestals.at(cellID - OFFSET)); }

  /// \brief Set pedestal
  /// \param cellID Absolute ID of cell
  /// \param c is the pedestal (expected to be in range <254)
  void setHGPedestal(short cellID, short c) { mHGPedestals[cellID - OFFSET] = static_cast<unsigned char>(c); }

  /// \brief Get pedestal
  /// \param cellID Absolute ID of cell
  /// \return pedestal for the cell
  short getLGPedestal(short cellID) const { return short(mLGPedestals.at(cellID - OFFSET)); }

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

 private:
  static constexpr short NCHANNELS = 14337;          ///< Number of channels starting from 1
  static constexpr short OFFSET = 5377;              ///< Non-existing channels 56*64*1.5+1
  std::array<unsigned char, NCHANNELS> mHGPedestals; ///< Container for pedestals
  std::array<unsigned char, NCHANNELS> mLGPedestals; ///< Container for pedestals

  ClassDefNV(Pedestals, 2);
};

} // namespace phos

} // namespace o2
#endif
