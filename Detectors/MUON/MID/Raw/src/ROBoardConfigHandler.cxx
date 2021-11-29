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

/// \file   MID/Raw/src/ROBoardConfigHandler.cxx
/// \brief  Handler for readout local board configuration
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   12 November 2021

#include "MIDRaw/ROBoardConfigHandler.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "fmt/format.h"
#include "MIDRaw/CrateMapper.h"

namespace o2
{
namespace mid
{

ROBoardConfigHandler::ROBoardConfigHandler()
{
  /// Default constructor
  set(makeDefaultROBoardConfig());
}

ROBoardConfigHandler::ROBoardConfigHandler(const char* filename)
{
  /// Construct from file
  load(filename);
}

ROBoardConfigHandler::ROBoardConfigHandler(const std::vector<ROBoardConfig>& configurations)
{
  set(configurations);
}

const ROBoardConfig ROBoardConfigHandler::getConfig(uint8_t uniqueLocId) const
{
  auto cfgIt = mROBoardConfigs.find(uniqueLocId);
  if (cfgIt == mROBoardConfigs.end()) {
    throw std::runtime_error(fmt::format("Cannot find local board {:02x}", uniqueLocId));
  }
  return cfgIt->second;
}

bool ROBoardConfigHandler::load(const char* filename)
{
  std::ifstream inFile(filename);
  if (!inFile.is_open()) {
    return false;
  }
  std::vector<ROBoardConfig> configurations;
  std::string line, token;
  while (std::getline(inFile, line)) {
    if (std::count(line.begin(), line.end(), ' ') < 5) {
      continue;
    }
    if (line.find('#') < line.find(' ')) {
      continue;
    }
    ROBoardConfig cfg;
    std::stringstream ss;
    ss << line;
    std::getline(ss, token, ' ');
    cfg.boardId = static_cast<uint8_t>(std::strtol(token.c_str(), nullptr, 16));
    std::getline(ss, token, ' ');
    cfg.configWord = static_cast<uint32_t>(std::strtol(token.c_str(), nullptr, 16));
    for (int ich = 0; ich < 4; ++ich) {
      std::getline(ss, token, ' ');
      auto mask = static_cast<uint32_t>(std::strtol(token.c_str(), nullptr, 16));
      cfg.masksBP[ich] = (mask >> 16) & 0xFFFF;
      cfg.masksNBP[ich] = (mask & 0xFFFF);
    }
    configurations.emplace_back(cfg);
    inFile.close();
  }
  set(configurations);
  return true;
}

void ROBoardConfigHandler::write(const char* filename) const
{
  /// Writes the masks to a configuration file
  std::vector<ROBoardConfig> configs;
  for (auto& cfgIt : mROBoardConfigs) {
    configs.emplace_back(cfgIt.second);
  }

  std::sort(configs.begin(), configs.end(), [](const ROBoardConfig& cfg1, const ROBoardConfig& cfg2) { return cfg1.boardId < cfg2.boardId; });

  std::ofstream outFile(filename);
  for (auto& cfg : configs) {
    outFile << cfg << std::endl;
  }
  outFile.close();
}

void ROBoardConfigHandler::set(const std::vector<ROBoardConfig>& configurations)
{
  mROBoardConfigs.clear();
  for (auto& cfg : configurations) {
    mROBoardConfigs.emplace(cfg.boardId, cfg);
  }
}

void ROBoardConfigHandler::updateMasks(const std::vector<ROBoard>& masks)
{
  for (auto& mask : masks) {
    auto cfgIt = mROBoardConfigs.find(mask.boardId);

    // First we check if some patterns has zeros.
    // When set xORy for boards with no Y input.
    // So in this case we explicitly mask Y.
    bool isMasked = ((cfgIt->second.configWord & crateconfig::sXorY) != 0);
    for (int ich = 0; ich < 4; ++ich) {
      if (mask.patternsBP[ich] != 0xFFFF || mask.patternsNBP[ich] != 0xFFFF) {
        isMasked = true;
      }
    }

    if (isMasked) {
      cfgIt->second.configWord |= crateconfig::sMonmoff;
      cfgIt->second.masksBP = mask.patternsBP;
      if ((cfgIt->second.configWord & crateconfig::sXorY) == 0) {
        cfgIt->second.masksNBP = mask.patternsNBP;
      }
    }
  }
}

std::vector<ROBoardConfig> makeDefaultROBoardConfig(uint16_t gbtUniqueId)
{
  // FIXME: in the current configuration, when an Y strip covers several local boards
  // the signal is sent to the first board only and no copy of the signal is sent to the others
  // In this case we cannot implement zero suppression to boards that only receive the X signal.
  // Originally we applied zero suppression to the board receiving both X and Y signals,
  // but this leads to a bias.
  // When the neighbour board is fired, indeed we have that:
  // - the board with no Y signal transmits its X signal
  // - the board with Y signal has no X signal. If we require X AND Y we lose the Y signal.
  // For the moment we decide to require X OR Y by default to all board.
  // This is equivalent to no zero suppression applied.
  // In the future, we might apply X AND Y to cases were the Y strip belong to 1 local board only.
  std::vector<ROBoardConfig> configurations;
  CrateMapper crateMapper;
  auto locIds = crateMapper.getROBoardIds(gbtUniqueId);
  for (auto& locId : locIds) {
    ROBoardConfig cfg;
    cfg.configWord = crateconfig::sTxDataMask | crateconfig::sXorY;
    cfg.boardId = locId;
    configurations.emplace_back(cfg);
  }
  return configurations;
}

std::vector<ROBoardConfig> makeNoZSROBoardConfig(uint16_t gbtUniqueId)
{
  // FIXME: notice that the current default implies no zero suppression
  // so this is equivalent to the default
  // We still keep this in case we change the default behaviour in the future
  auto configurations = makeDefaultROBoardConfig(gbtUniqueId);
  for (auto& cfg : configurations) {
    cfg.configWord |= crateconfig::sXorY;
  }
  return configurations;
}

} // namespace mid
} // namespace o2
