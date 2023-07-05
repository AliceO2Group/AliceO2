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

#include "Quadrant.h"

#include "LV.h"
#include "MCHConditions/Chamber.h"
#include "MCHConditions/Number.h"
#include "MCHConditions/Side.h"
#include <array>
#include <fmt/core.h>
#include <set>

namespace o2::mch::dcs::quadrant
{
Cathode lvAliasToCathode(std::string_view alias)
{
  const auto chamber = aliasToChamber(alias);
  const auto side = aliasToSide(alias);
  const auto number = aliasToNumber(alias);
  int base = (1 + toInt(chamber)) * 100;
  if (isStation1(chamber)) {
    // Station 1
    if (side == Side::Left) {
      switch (number) {
        case 4:
          return {Cathode{base + 1, Plane::NonBending}};
        case 2:
          return {Cathode{base + 1, Plane::Bending}};
        case 3:
          return {Cathode{base + 2, Plane::Bending}};
        case 1:
          return {Cathode{base + 2, Plane::NonBending}};
        default:
          throw std::invalid_argument(fmt::format("wrong alias {}", alias));
      }
    } else {
      switch (number) {
        case 4:
          return {Cathode{base + 0, Plane::Bending}};
        case 2:
          return {Cathode{base + 0, Plane::NonBending}};
        case 3:
          return {Cathode{base + 3, Plane::NonBending}};
        case 1:
          return {Cathode{base + 3, Plane::Bending}};
        default:
          throw std::invalid_argument(fmt::format("wrong alias {}", alias));
      }
    }
  } else if (isStation2(chamber)) {
    // Station 2
    if (side == Side::Left) {
      switch (number) {
        case 2:
          return {Cathode{base + 1, Plane::NonBending}};
        case 1:
          return {Cathode{base + 1, Plane::Bending}};
        case 4:
          return {Cathode{base + 2, Plane::Bending}};
        case 3:
          return {Cathode{base + 2, Plane::NonBending}};
        default:
          throw std::invalid_argument(fmt::format("wrong alias {}", alias));
      }
    } else {
      switch (number) {
        case 2:
          return {Cathode{base + 0, Plane::Bending}};
        case 1:
          return {Cathode{base + 0, Plane::NonBending}};
        case 4:
          return {Cathode{base + 3, Plane::NonBending}};
        case 3:
          return {Cathode{base + 3, Plane::Bending}};
        default:
          throw std::invalid_argument(fmt::format("wrong alias {}", alias));
      }
    }
  } else {
    throw std::invalid_argument(fmt::format("wrong alias {} (expecting station12 one", alias));
  }
}

std::set<int> solarAliasToDsIndices(std::string_view alias)
{
  /** For St12, to get the solar alias to dual sampas relationship
   * we "just" have to convert the solar crate number to a LV
   * group (as used in the analog and digital measurements),
   * and then reuse the LV group logic.
   */
  // lvid.type = dcs::MeasurementType::LV_V_FEE_ANALOG;
  static const std::array<int, 4> crateToGroupSt1 = {4, 2, 3, 1};
  static const std::array<int, 4> crateToGroupSt2 = {2, 1, 4, 3};

  auto chamber = aliasToChamber(alias);
  auto solarCrate = aliasToNumber(alias);

  int group;
  if (isStation1(chamber)) {
    group = crateToGroupSt1[solarCrate - 1];
  } else {
    group = crateToGroupSt2[solarCrate - 1];
  }

  auto side = aliasToSide(alias);
  auto lvAlias = fmt::format("MchHvLv{}/Chamber{}{}/Group{}an",
                             name(side),
                             toInt(chamber),
                             name(side),
                             group);

  return lvAliasToDsIndices(lvAlias);
}
} // namespace o2::mch::dcs::quadrant
