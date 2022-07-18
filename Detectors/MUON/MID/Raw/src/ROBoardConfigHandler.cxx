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
  set(makeDefaultROBoardConfig());
}

ROBoardConfigHandler::ROBoardConfigHandler(const char* filename)
{
  load(filename);
}

ROBoardConfigHandler::ROBoardConfigHandler(std::istream& in)
{
  load(in);
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

void ROBoardConfigHandler::load(std::istream& in)
{
  std::vector<ROBoardConfig> configurations;
  std::string line, token;
  while (std::getline(in, line)) {
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
  }
  set(configurations);
}

bool ROBoardConfigHandler::load(const char* filename)
{
  std::ifstream inFile(filename);
  if (!inFile.is_open()) {
    return false;
  }
  load(inFile);
  return true;
}

void ROBoardConfigHandler::write(std::ostream& out) const
{
  std::vector<ROBoardConfig> configs;
  for (auto& cfgIt : mROBoardConfigs) {
    configs.emplace_back(cfgIt.second);
  }

  std::sort(configs.begin(), configs.end(), [](const ROBoardConfig& cfg1, const ROBoardConfig& cfg2) { return cfg1.boardId < cfg2.boardId; });

  for (auto& cfg : configs) {
    out << cfg << "\n";
  }
}

void ROBoardConfigHandler::write(const char* filename) const
{
  std::ofstream outFile(filename);
  write(outFile);
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

    bool isMasked = false;
    for (int ich = 0; ich < 4; ++ich) {
      if (mask.patternsBP[ich] != 0xFFFF || mask.patternsNBP[ich] != 0xFFFF) {
        isMasked = true;
      }
    }

    if (isMasked) {
      cfgIt->second.configWord |= crateconfig::sMonmoff;
      cfgIt->second.masksBP = mask.patternsBP;
      cfgIt->second.masksNBP = mask.patternsNBP;
    }
  }
}

std::vector<ROBoardConfig> makeZSROBoardConfig(uint16_t gbtUniqueId)
{
  // In this configuration, data from one local board
  // (reading the output from 4 detection planes)
  // is transmitted if at least one strip in X AND Y are fired on the same detection plane.
  std::vector<ROBoardConfig> configurations;
  CrateMapper crateMapper;
  auto locIds = crateMapper.getROBoardIds(gbtUniqueId);
  for (auto& locId : locIds) {
    ROBoardConfig cfg;
    cfg.configWord = crateconfig::sTxDataMask;
    cfg.boardId = locId;
    configurations.emplace_back(cfg);
  }
  return configurations;
}

std::vector<ROBoardConfig> makeNoZSROBoardConfig(uint16_t gbtUniqueId)
{
  // In this configuration, no zero suppression is performed.
  // Data from one local board are transmitted as soon as one strip in X OR Y is fired.
  auto configurations = makeZSROBoardConfig(gbtUniqueId);
  for (auto& cfg : configurations) {
    cfg.configWord |= crateconfig::sXorY;
  }
  return configurations;
}

std::vector<ROBoardConfig> makeDefaultROBoardConfig(uint16_t gbtUniqueId)
{
  // Originally, the electronics was configured to apply the zero suppression as explained in makeZSROBoardConfig.
  // However, this logic implies that, when a Y strip covers several local boards,
  // the signal is copied to all local boards.
  // This is not the case in the current electronics setup: when an Y strip covers several local boards
  // the signal is sent to the first board only and no copy of the signal is sent to the others.
  // If we try to apply zero suppression with this setup, we have a bias.
  // Indeed, when a board without direct Y signal is fired, we have that:
  // - the board only transmits its X signal
  // - the neighbour board with Y signal has no X signal. If we require X AND Y we lose the Y signal.
  // If we do not change the setup, we are therefore obliged to release the zero suppression.
  return makeNoZSROBoardConfig(gbtUniqueId);
}

} // namespace mid
} // namespace o2
