// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <map>

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
  uint16_t deLocalBoardToRO(uint8_t deId, uint8_t columnId, uint8_t lineId) const;

  uint16_t roLocalBoardToDE(uint8_t crateId, uint8_t boardId) const;

 private:
  void init();
  std::map<uint16_t, uint16_t> mROToDEMap; /// Correspondence between RO and DE board
  std::map<uint16_t, uint16_t> mDEToROMap; /// Correspondence between DE and RO board
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CRATEMAPPER_H */
