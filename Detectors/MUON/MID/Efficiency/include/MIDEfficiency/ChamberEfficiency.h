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

/// \file   MIDEfficiency/ChamberEfficiency.h
/// \brief  Measured values of the RPC efficiency
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   03 March 2019
#ifndef O2_MID_CHAMBEREFFICIENCY_H
#define O2_MID_CHAMBEREFFICIENCY_H

#include <cstdint>
#include <vector>
#include <unordered_map>

#include "DataFormatsMID/ChEffCounter.h"

namespace o2
{
namespace mid
{

class ChamberEfficiency
{
 public:
  enum class EffType {
    BendPlane,    ///< Bending plane efficiency
    NonBendPlane, ///< Non-bending plane efficiency
    BothPlanes    ///< Both plane efficiency
  };

  /// Sets the efficiency from a vector of counters
  /// \param counters Vector of efficiency counters
  void setFromCounters(const std::vector<ChEffCounter>& counters);

  /// Gets the efficiency
  /// \param deId Detection element ID
  /// \param columnId Column ID
  /// \param lineId Line of the local board in the RPC
  /// \param type Efficiency type
  double getEfficiency(int deId, int columnId, int lineId, EffType type) const;

  /// Adds an entry
  /// \param isEfficientBP Bending plane was efficient
  /// \param isEfficientNBP Non-bending plane was efficient
  /// \param deId Detection element ID
  /// \param columnId Column ID
  /// \param lineId line of the local board in the RPC
  void addEntry(bool isEfficientBP, bool isEfficientNBP, int deId, int columnId, int lineId);

  /// Returns the efficiency counters
  /// \return vector of chamber efficiency counters
  std::vector<ChEffCounter> getCountersAsVector() const;

 private:
  /// Convert EffType to EffCountType
  /// \param type Efficiency type
  /// \return Efficiency counter type
  EffCountType convert(EffType type) const;
  std::unordered_map<uint16_t, ChEffCounter> mCounters; ///< Efficiency counters
};

ChamberEfficiency createDefaultChamberEfficiency();

} // namespace mid
} // namespace o2

#endif /* O2_MID_CHAMBEREFFICIENCY_H */
