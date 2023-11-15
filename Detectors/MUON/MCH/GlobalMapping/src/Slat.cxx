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

#include "MCHConditions/DetectionElement.h"
#include "MCHConditions/Number.h"
#include "MCHConditions/SolarCrate.h"
#include "MCHGlobalMapping/DsIndex.h"
#include "MCHRawElecMap/Mapper.h"
#include "Slat.h"
#include <vector>

namespace o2::mch::dcs::slat
{

std::set<dcs::Cathode> lvAliasToCathode(std::string_view alias)
{
  std::vector<int> slats;
  const auto chamber = aliasToChamber(alias);
  const auto number = aliasToNumber(alias);
  const auto side = aliasToSide(alias);
  if (chamber == Chamber::Ch04 ||
      chamber == Chamber::Ch05) {
    switch (number) {
      case 1:
        slats.emplace_back(0);
        slats.emplace_back(1);
        slats.emplace_back(2);
        break;
      case 5:
        slats.emplace_back(6);
        slats.emplace_back(7);
        slats.emplace_back(8);
        break;
      default:
        slats.emplace_back(number + 1);
    };
  }
  if (
    chamber == Chamber::Ch06 ||
    chamber == Chamber::Ch07 ||
    chamber == Chamber::Ch08 ||
    chamber == Chamber::Ch09) {
    switch (number) {
      case 1:
        slats.emplace_back(0);
        slats.emplace_back(1);
        slats.emplace_back(2);
        slats.emplace_back(3);
        break;
      case 7:
        slats.emplace_back(9);
        slats.emplace_back(10);
        slats.emplace_back(11);
        slats.emplace_back(12);
        break;
      default:
        slats.emplace_back(number + 2);
    };
  }
  std::set<dcs::Cathode> cathodes;
  for (const auto slat : slats) {
    int deId = detElemId(chamber, side, slat);
    cathodes.emplace(dcs::Cathode{deId, dcs::Plane::Bending});
    cathodes.emplace(dcs::Cathode{deId, dcs::Plane::NonBending});
  }
  return cathodes;
}

std::set<int> solarAliasToDsIndices(std::string_view alias)
{
  /** For St345 the relation solar alias to dual sampas
   * is a bit more involved than for quadrants.
   * We must have the mapping from alias to solar crate number
   * and then from solar crate number to list of solars in that crate.
   * Finally from the list of solars we know the list of dual sampas
   * from other mapper(s).
   */
  int solarCrate = aliasToSolarCrate(alias);
  static auto solarIds = raw::getSolarUIDs<raw::ElectronicMapperGenerated>();
  std::set<int> dsIndices;
  for (auto solarId : solarIds) {
    if (solarId / 8 == solarCrate) {
      auto dsDetIds = raw::getDualSampas<raw::ElectronicMapperGenerated>(solarId);
      for (auto dsDetId : dsDetIds) {
        auto index = getDsIndex(dsDetId);
        dsIndices.emplace(index);
      }
    }
  }
  return dsIndices;
}
} // namespace o2::mch::dcs::slat
