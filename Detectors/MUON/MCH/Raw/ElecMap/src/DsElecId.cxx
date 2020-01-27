// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawElecMap/DsElecId.h"
#include "Assertions.h"
#include <fmt/format.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

namespace o2::mch::raw
{
DsElecId::DsElecId(uint16_t solarId, uint8_t elinkGroupId, uint8_t elinkIndex)
  : mSolarId{solarId}, mElinkGroupId{elinkGroupId}, mElinkIndexInGroup{elinkIndex}
{
  impl::assertIsInRange("elinkGroupId", mElinkGroupId, 0, 7);
  impl::assertIsInRange("elinkIndex", mElinkIndexInGroup, 0, 4);
}

uint16_t encode(const DsElecId& id)
{
  return (id.solarId() & 0x3FF) | ((id.elinkGroupId() & 0x7) << 10) |
         ((id.elinkIndexInGroup() & 0x7) << 13);
}

std::optional<DsElecId> decodeDsElecId(uint16_t code)
{
  uint16_t solarId = code & 0x3FF;

  uint8_t groupId = (code & 0x1C00) >> 10;

  uint8_t index = (code & 0xE000) >> 13;

  if (groupId > 7) {
    return std::nullopt;
  }
  if (index > 4) {
    return std::nullopt;
  }
  return DsElecId(solarId, groupId, index);
}

std::optional<DsElecId> decodeDsElecId(std::string rep)
{
  std::istringstream is(rep);
  std::string line;
  std::vector<std::string> tokens;
  while (getline(is, line, '-')) {
    tokens.emplace_back(line);
  }
  if (tokens.size() < 3) {
    // not a valid representation of a DsElecId
    return std::nullopt;
  }
  if (tokens[0].empty() || tokens[0][0] != 'S') {
    // token[0] is not a valid representation of a solarId
    return std::nullopt;
  }
  if (tokens[1].empty() || tokens[1][0] != 'J') {
    // token[1] is not a valid representation of a groupId
    return std::nullopt;
  }
  if (tokens[2].size() < 3 || tokens[2][0] != 'D' || tokens[2][1] != 'S') {
    // token is not a valid representation of a DS
    return std::nullopt;
  }
  uint16_t solarId = std::atoi(tokens[0].substr(1).c_str());
  uint8_t groupId = std::atoi(tokens[1].substr(1).c_str());
  uint8_t index = std::atoi(tokens[2].substr(2).c_str());
  return DsElecId(solarId, groupId, index);
}

std::optional<uint8_t> decodeChannelId(std::string rep)
{
  auto dsElecId = decodeDsElecId(rep);
  if (!dsElecId.has_value()) {
    // must be a valid dsElecId
    return std::nullopt;
  }
  std::istringstream is(rep);
  std::string line;
  std::vector<std::string> tokens;
  while (getline(is, line, '-')) {
    tokens.emplace_back(line);
  }
  if (tokens.size() < 4) {
    // not a valid representation of a {DsElecId,ChannelId}
    return std::nullopt;
  }
  if (tokens[3].size() < 3 || tokens[3][0] != 'C' || tokens[3][1] != 'H') {
    // token[3] is not a valid representation of a CH
    return std::nullopt;
  }
  auto chId = std::atoi(tokens[3].substr(2).c_str());
  if (chId >= 0 && chId <= 63) {
    return chId;
  }
  return std::nullopt;
}

std::ostream& operator<<(std::ostream& os, const DsElecId& id)
{
  std::cout << fmt::format("DsElecId(SOLAR=S{:4d} GROUP=J{:2d} INDEX=DS{:2d}) CODE={:8d}",
                           id.solarId(), id.elinkGroupId(), id.elinkIndexInGroup(), encode(id));
  return os;
}

std::string asString(DsElecId dsId)
{
  return fmt::format("S{}-J{}-DS{}", dsId.solarId(), dsId.elinkGroupId(), dsId.elinkIndexInGroup());
}

std::optional<uint8_t> groupFromElinkId(uint8_t elinkId)
{
  if (elinkId < 40) {
    return elinkId / 5;
  }
  return std::nullopt;
}

std::optional<uint8_t> indexFromElinkId(uint8_t elinkId)
{
  if (elinkId < 40) {
    return elinkId - (elinkId / 5) * 5;
  }
  return std::nullopt;
}
} // namespace o2::mch::raw
