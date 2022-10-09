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

#include "MCHConditions/Number.h"
#include <fmt/core.h>

namespace
{

int parseNumber(std::string_view alias, std::string_view key, uint8_t len)
{
  auto pos = alias.find(key);
  std::string s{alias.substr(pos + key.size(), len)};
  return std::stoi(s);
}

} // namespace

namespace o2::mch::dcs
{
int aliasToNumber(std::string_view alias)
{
  if (alias.find("Slat") != std::string_view::npos) {
    return parseNumber(alias, "Slat", 2);
  } else if (alias.find("Quad") != std::string_view::npos) {
    auto quad = parseNumber(alias, "Quad", 1);
    auto sector = parseNumber(alias, "Sect", 1);
    return quad * 10 + sector;
  } else if (alias.find("SolCh") != std::string_view::npos) {
    return parseNumber(alias, "Cr", 2);
  } else if (alias.find("Group") != std::string_view::npos) {
    return parseNumber(alias, "Group", 2);
  }
  throw std::invalid_argument(fmt::format("Cannot extract number from alias={}", alias));
}
} // namespace o2::mch::dcs
