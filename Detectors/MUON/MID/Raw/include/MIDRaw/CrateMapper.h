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

  /// @brief Converts the LOC ID expressed in the offline convention into the readout convention
  /// @param deId Detection element ID
  /// @param columnId Column ID
  /// @param lineId Line ID
  /// @return LOC ID for readout electronics
  uint8_t deLocalBoardToRO(uint8_t deId, uint8_t columnId, uint8_t lineId) const;

  /// @brief Converts the LOC ID expressed in readout convention into the LOC ID in MT11 right in the offline convention
  /// @param uniqueLocId LOC ID for the readout electronics
  /// @return LOC ID in the offline way
  uint16_t roLocalBoardToDE(uint8_t uniqueLocId) const;

  /// @brief Checks if local board ID (RO convention) has direct input from FEE y strips
  /// @param uniqueLocId LOC ID in the RO convention
  /// @returns true if local board ID has direct input from FEE y strips
  bool hasDirectInputY(uint8_t uniqueLocId) const { return mLocIdsWithDirectInputY.find(getROBoardIdRight(uniqueLocId)) != mLocIdsWithDirectInputY.end(); }

  /// @brief Gets the local boards with a direct input from FEE y strips
  /// @return An unordered set with the local boards IDs (offline convention) with a direct input from FEE y strips
  std::unordered_set<uint8_t> getLocalBoardsWithDirectInputY() const { return mLocIdsWithDirectInputY; }

  /// @brief Returns the list of local board IDs (RO convention)
  /// @param gbtUniqueId Limit the query to the links belonging to the specified gbtUniqueId. If gbtUniqueId is 0xFFFF, return all
  /// @return Sorted vector of local board unique IDs (offline convention)
  std::vector<uint8_t> getROBoardIds(uint16_t gbtUniqueId = 0xFFFF) const;

 private:
  /// @brief Initializes the crate mapping
  void init();

  /// @brief Returns the unique Loc ID in the right side (offline convention)
  uint8_t getROBoardIdRight(uint8_t uniqueLocId) const { return uniqueLocId % 0x80; }

  std::unordered_map<uint8_t, uint16_t> mROToDEMap;    /// Correspondence between boards in the RO and Offline convention
  std::unordered_map<uint16_t, uint8_t> mDEToROMap;    /// Correspondence between boards in the Offline and RO convention
  std::unordered_set<uint8_t> mLocIdsWithDirectInputY; /// IDs of the local board (offline convention) with direct input from FEE y strips
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CRATEMAPPER_H */
