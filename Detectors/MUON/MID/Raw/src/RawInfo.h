// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/RawInfo.h
/// \brief  Raw data format MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019
#ifndef O2_MID_RAWINFO_H
#define O2_MID_RAWINFO_H

#include <cstdint>

namespace o2
{
namespace mid
{
// Local board patterns
static constexpr unsigned int sNBitsLocId = 4;
static constexpr unsigned int sNBitsFiredStrips = 16;

// Column info
static constexpr unsigned int sNBitsCrateId = 4;
static constexpr unsigned int sNBitsNFiredBoards = 4;

// Chamber info
static constexpr unsigned int sNBitsFiredChambers = 4;

// Local clock
static constexpr unsigned int sNBitsLocalClock = 16;

// Event info
static constexpr unsigned int sNBitsEventWord = 8;
static constexpr uint16_t sDelayCalibToFET = 10;
static constexpr uint16_t sDelayBCToLocal = 0;

// Buffer info
static constexpr unsigned int sMaxPageSize = 0x2000;
} // namespace mid
} // namespace o2

#endif /* O2_MID_RAWINFO_H */
