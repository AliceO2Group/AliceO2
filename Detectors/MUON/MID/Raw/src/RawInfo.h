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
static constexpr unsigned int sNBitsBoardId = 2;
static constexpr unsigned int sNBitsFiredStrips = 16;

// Column info
static constexpr unsigned int sNBitsColumnId = 3;
static constexpr unsigned int sNBitsNFiredBoards = 3;

// RPC info
static constexpr unsigned int sNBitsRPCId = 7;
static constexpr unsigned int sNBitsNFiredColumns = 3;

static constexpr unsigned int sNBitsNFiredRPCs = 7;

// Local clock
static constexpr unsigned int sNBitsLocalClock = 16;

// Trigger info
static constexpr unsigned int sNBitsEventType = 2;
static constexpr unsigned int sStandardEvent = 0;
static constexpr unsigned int sSoftwareTriggerEvent = 1;
static constexpr unsigned int sFEEEvent = 2;

} // namespace mid
} // namespace o2

#endif /* O2_MID_RAWINFO_H */
