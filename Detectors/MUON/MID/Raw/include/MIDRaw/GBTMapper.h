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

/// \file   MIDRaw/GBTMapper.h
/// \brief  MID GBT mapping
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   16 November 2021
#ifndef O2_MID_GBTMAPPER_H
#define O2_MID_GBTMAPPER_H

#include "DataFormatsMID/ROBoard.h"
#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{
/// Functions to relate the unique local board ID with the corresponding GBT ID
namespace gbtmapper
{
/// Gets the GBT ID from the unique loc ID
/// \param uniqueLocId Unique local board ID
/// \return GBT ID
inline uint16_t getGBTIdFromUniqueLocId(uint8_t uniqueLocId) { return crateparams::makeGBTUniqueId(raw::getCrateId(uniqueLocId), crateparams::getGBTIdFromBoardInCrate(raw::getLocId(uniqueLocId))); }
/// Check if the board is in the GBT
/// \param uniqueLocId Unique local board ID
/// \param gbtUniqueId Unique GBT ID
/// \return True if the local board is in the GBT
inline bool isBoardInGBT(uint8_t uniqueLocId, uint16_t gbtUniqueId) { return (raw::getCrateId(uniqueLocId) == crateparams::getCrateIdFromGBTUniqueId(gbtUniqueId) && crateparams::getGBTIdFromBoardInCrate(raw::getLocId(uniqueLocId)) == crateparams::getGBTIdInCrate(gbtUniqueId)); }
} // namespace gbtmapper
} // namespace mid
} // namespace o2

#endif /* O2_MID_GBTMAPPER_H */
