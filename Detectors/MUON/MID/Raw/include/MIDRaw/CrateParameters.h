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

/// Builds the GBT unique ID from the crate ID and the GBT ID in the crate
inline uint16_t makeGBTUniqueId(uint8_t crateId, uint8_t gbtId) { return sNGBTsPerCrate * crateId + gbtId; }
/// Gets the crate ID from the GBT unique ID
inline uint8_t getCrateIdFromGBTUniqueId(uint16_t gbtUniqueId) { return gbtUniqueId / sNGBTsPerCrate; }
/// Gets the link ID in crate from the RO ID
inline uint8_t getGBTIdInCrate(uint16_t gbtUniqueId) { return gbtUniqueId % sNGBTsPerCrate; }
/// Gets the link ID in crate from the board ID
inline uint8_t getGBTIdFromBoardInCrate(uint16_t locId) { return locId / sMaxNBoardsInLink; }
/// Gets the absolute crate ID
inline uint8_t getCrateId(bool isRightSide, uint8_t crateIdOneSide) { return isRightSide ? crateIdOneSide : crateIdOneSide + sNCratesPerSide; }
/// Tests if the crate is in the right side
inline bool isRightSide(uint8_t crateId) { return (crateId / sNCratesPerSide) == 0; }
} // namespace crateparams
} // namespace mid
} // namespace o2

#endif /* O2_MID_CRATEPARAMETERS_H */
