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

/// \file   MID/Raw/src/ROBoardConfig.cxx
/// \brief  Configuration for the readout local board
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   12 November 2021

#include "MIDRaw/ROBoardConfig.h"

#include <iostream>
#include "fmt/format.h"

namespace o2
{
namespace mid
{
std::ostream& operator<<(std::ostream& os, const ROBoardConfig& cfg)
{
  /// Stream operator for ROBoardConfig
  os << fmt::format("{:02x} {:08x}", cfg.boardId, cfg.configWord);
  for (int ich = 0; ich < 4; ++ich) {
    os << fmt::format(" {:04x}{:04x}", cfg.masksBP[ich], cfg.masksNBP[ich]);
  }
  return os;
}

} // namespace mid
} // namespace o2