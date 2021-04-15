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
/// \brief CCDB container for the full set of CPV calibration coefficients
/// \author Dmitri Peresunko, RRC Kurchatov institute
/// \since Aug. 1, 2019
///
///

#ifndef CPV_PEDESTALS_H
#define CPV_PEDESTALS_H

#include <array>
#include "TObject.h"

class TH1;
class TH1F;

namespace o2
{

namespace cpv
{

class Pedestals
{
 public:
  /// \brief Constructor
  Pedestals() = default;

  /// \brief Constructor for tests
  Pedestals(int test);

  /// \brief Destructor
  ~Pedestals() = default;

  /// \brief Get pedestal
  /// \param cellID Absolute ID of cell
  /// \return pedestal for the cell
  short getPedestal(short cellID) const { return short(mPedestals.at(cellID)); }
  float getPedSigma(short cellID) const { return mPedSigmas.at(cellID); }

  /// \brief Set pedestal
  /// \param cellID Absolute ID of cell
  /// \param c is the pedestal (expected to be in range <254)
  void setPedestal(short cellID, short c) { mPedestals[cellID] = (c > 0 && c < 511) ? c : mPedestals[cellID]; }
  void setPedSigma(short cellID, float c) { mPedSigmas[cellID] = (c > 0) ? c : mPedSigmas[cellID]; }

  /// \brief Set pedestals from 1D histogram with cell absId in x axis
  /// \param 1D(NCHANNELS) histogram with calibration coefficients
  /// \return Is successful
  bool setPedestals(TH1* h);
  bool setPedSigmas(TH1F* h);

 private:
  static constexpr short NCHANNELS = 23040; ///< Number of channels in 3 modules starting from 0
  std::array<short, NCHANNELS> mPedestals;  ///< Container for pedestals
  std::array<float, NCHANNELS> mPedSigmas;  ///< Container for pedestal sigmas

  ClassDefNV(Pedestals, 2);
};

} // namespace cpv

} // namespace o2
#endif
