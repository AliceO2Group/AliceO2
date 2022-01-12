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

/// \file   MIDRaw/ROBoardConfig.h
/// \brief  Configuration for the readout local board
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   12 November 2021
#ifndef O2_MID_ROBOARDCONFIG_H
#define O2_MID_ROBOARDCONFIG_H

#include <cstdint>
#include "DataFormatsMID/ROBoard.h"

namespace o2
{
namespace mid
{
struct ROBoardConfig {
  uint32_t configWord = 0;            // Readout board config word
  uint8_t boardId{0};                 // Board ID
  std::array<uint16_t, 4> masksBP{};  // Bending plane mask
  std::array<uint16_t, 4> masksNBP{}; // Non-bending plane mask
};

std::ostream& operator<<(std::ostream& os, const ROBoardConfig& cfg);

namespace crateconfig
{
static constexpr uint32_t sTxDataMask = 0x10000;
static constexpr uint32_t sMonmoff = 0x2;
static constexpr uint32_t sXorY = 0x400;
} // namespace crateconfig

} // namespace mid
} // namespace o2

#endif /* O2_MID_ROBOARDCONFIG_H */
