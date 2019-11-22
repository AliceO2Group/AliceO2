// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/CrateParameters.h
/// \brief  MID RO crate parameters
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   19 November 2019
#ifndef O2_MID_CRATEPARAMETERS_H
#define O2_MID_CRATEPARAMETERS_H

#include <cstdint>
#include <array>

namespace o2
{
namespace mid
{
namespace crateparams
{
static constexpr unsigned int sNCratesPerSide = 8;
static constexpr unsigned int sNCrates = 2 * sNCratesPerSide;
static constexpr unsigned int sNGBTsPerCrate = 2;
static constexpr unsigned int sNGBTsPerSide = sNGBTsPerCrate * sNCratesPerSide;
static constexpr unsigned int sNGBTs = 2 * sNGBTsPerSide;
static constexpr unsigned int sMaxNBoardsInLink = 8;
static constexpr unsigned int sMaxNBoardsInCrate = sMaxNBoardsInLink * sNGBTsPerCrate;
static constexpr unsigned int sNELinksPerGBT = 10;

/// Builds the RO ID from the crate ID and the GBT ID in the crate
inline uint16_t makeROId(uint8_t crateId, uint8_t gbtId) { return sNGBTsPerCrate * crateId + gbtId; }
/// Gets the crate ID from the RO ID
inline uint8_t getCrateIdFromROId(uint16_t roId) { return roId / sNGBTsPerCrate; }
/// Gets the link ID in crate from the RO ID
inline uint8_t getGBTIdInCrate(uint16_t roId) { return roId % sNGBTsPerCrate; }
/// Gets the link ID in crate from the board ID
inline uint8_t getGBTIdFromBoardInCrate(uint16_t locId) { return locId / sMaxNBoardsInLink; }
/// Gets the absolute crate ID
inline uint8_t getCrateId(bool isRightSide, uint8_t crateIdOneSide) { return isRightSide ? crateIdOneSide : crateIdOneSide + sNCratesPerSide; }
/// Tests if the crate is in the right side
inline bool isRightSide(uint8_t crateId) { return (crateId / sNCratesPerSide) == 0; }
/// Builds the unique loc ID from the crate ID and the loc ID in the crate
inline uint8_t makeUniqueLocID(uint8_t crateId, uint8_t locId) { return locId | (crateId << 4); }
/// Gets the crate ID from the absolute local board ID
inline uint8_t getCrateId(uint8_t uniqueLocId) { return (uniqueLocId >> 4) & 0xF; }
/// Gets the loc ID in the crate from the unique local board ID
inline uint8_t getLocId(uint16_t uniqueLocId) { return uniqueLocId & 0xF; }
/// Tests the local card bit
inline bool isLoc(uint8_t statusWord) { return (statusWord >> 6) & 0x1; }
/// Tests if the calibration bit of the card
inline bool isCalibration(uint8_t eventWord) { return ((eventWord & 0xc) == 0x8); }
/// Tests if this is a Front End Test event
inline bool isFET(uint8_t eventWord) { return ((eventWord & 0xc) == 0xc); }
} // namespace crateparams
} // namespace mid
} // namespace o2

#endif /* O2_MID_CRATEPARAMETERS_H */
