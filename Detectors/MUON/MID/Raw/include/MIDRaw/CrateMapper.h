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

/// \file   MIDRaw/CrateMapper.h
/// \brief  Mapper to convert the RO Ids to a format suitable for O2
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   15 November 2019
#ifndef O2_MID_CRATEMAPPER_H
#define O2_MID_CRATEMAPPER_H

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace o2
{
namespace mid
{
class CrateMapper
{
 public:
  CrateMapper();
  ~CrateMapper() = default;

  /// Builds the local board ID in the detection element
  static uint16_t deBoardId(uint8_t rpcLineId, uint8_t columnId, uint8_t lineId) { return (rpcLineId | (columnId << 4) | (lineId << 7)); }
  /// Gets the RPC line from the DE board ID
  static uint8_t getRPCLine(uint16_t deBoardId) { return deBoardId & 0xF; }
  /// Gets the column ID from the DE board ID
  static uint8_t getColumnId(uint16_t deBoardId) { return (deBoardId >> 4) & 0x7; }
  /// Gets the line ID from the DE board ID
  static uint8_t getLineId(uint16_t deBoardId) { return (deBoardId >> 7) & 0x3; }
  uint8_t deLocalBoardToRO(uint8_t deId, uint8_t columnId, uint8_t lineId) const;

  uint16_t roLocalBoardToDE(uint8_t uniqueLocId) const;

  /// Checks if local board ID has direct input from FEE y strips
  bool hasDirectInputY(uint8_t uniqueLocId) const { return mLocIdsWithDirectInputY.find(getROBoardIdRight(uniqueLocId)) != mLocIdsWithDirectInputY.end(); }

  /// Gets the local boards with a direct input from FEE y strips
  std::unordered_set<uint8_t> getLocalBoardsWithDirectInputY() const { return mLocIdsWithDirectInputY; }

  /// Returns the list of readout local board IDs
  /// \param gbtUniqueId Limit the query to the links belonging to the specified gbtUniqueId. If gbtUniqueId is 0xFFFF, return all
  /// \return Sorted vector of local board unique IDs
  std::vector<uint8_t> getROBoardIds(uint16_t gbtUniqueId = 0xFFFF) const;

 private:
  /// Initializes the crate mapping
  void init();
  /// Returns the unique Loc ID in the right side
  uint8_t getROBoardIdRight(uint8_t uniqueLocId) const { return uniqueLocId % 0x80; }
  std::unordered_map<uint8_t, uint16_t> mROToDEMap;    /// Correspondence between RO and DE board
  std::unordered_map<uint16_t, uint8_t> mDEToROMap;    /// Correspondence between DE and RO board
  std::unordered_set<uint8_t> mLocIdsWithDirectInputY; /// IDs of the local board with direct input from FEE y strips
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CRATEMAPPER_H */
