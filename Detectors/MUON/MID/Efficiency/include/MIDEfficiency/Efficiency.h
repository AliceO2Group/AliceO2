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

/// \file   MIDEfficiency/Efficiency.h
/// \brief  Computes the MID chamber efficiency
/// \author Livia Terlizzi <Livia.Terlizzi at cern.ch>
/// \date   20 September 2022

#ifndef O2_MID_EFFICIENCY_H
#define O2_MID_EFFICIENCY_H

#include <gsl/span>
#include <array>
#include "DataFormatsMID/Track.h"
#include "MIDEfficiency/ChamberEfficiency.h"
namespace o2
{
namespace mid
{

/// Class to estimate the MID chamber efficiency
class Efficiency
{
 public:
  enum class ElementType {
    Board, /// Efficiency per board
    RPC,   /// Efficiency per RPC
    Plane  /// Efficiency per chamber plane
  };

  /// @brief Fill the counters to estimate the chamber efficiency
  /// @param midTracks Reconstructed tracks
  void process(gsl::span<const mid::Track> midTracks);

  /// @brief Returns the chamber efficiency
  /// @param et Element type
  /// @return Chamber efficiency for the specified type
  const ChamberEfficiency& getChamberEfficiency(ElementType et) const { return mEff[static_cast<int>(et)]; }

 private:
  std::array<ChamberEfficiency, 3> mEff; /// Efficiency handler
};

} // namespace mid
} // namespace o2

#endif // O2_MID_EFFICIENCY_H
