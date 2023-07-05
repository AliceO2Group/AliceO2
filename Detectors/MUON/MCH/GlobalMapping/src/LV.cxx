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

#include "LV.h"

#include "MCHConditions/Cathode.h"
#include "MCHConditions/Chamber.h"
#include "MCHConditions/Plane.h"
#include "MCHConstants/DetectionElements.h"
#include "MCHGlobalMapping/DsIndex.h"
#include "MCHRawElecMap/Mapper.h"
#include "Quadrant.h"
#include "Slat.h"

namespace o2::mch::dcs
{

std::set<Cathode> lvAliasToCathode(std::string_view alias)
{
  const auto chamber = aliasToChamber(alias);
  if (dcs::isQuadrant(chamber)) {
    return {quadrant::lvAliasToCathode(alias)};
  } else {
    return slat::lvAliasToCathode(alias);
  }
}

std::set<int> lvAliasToDsIndices(std::string_view alias)
{
  auto cathodes = lvAliasToCathode(alias);
  std::set<int> dsIndices;
  for (auto cathode : cathodes) {
    auto deId = cathode.deId;
    auto plane = cathode.plane;
    if (constants::isValidDetElemId(deId)) {
      auto solarIds = raw::getSolarUIDs<raw::ElectronicMapperGenerated>(deId);
      for (auto solarId : solarIds) {
        auto dsDetIds = raw::getDualSampas<raw::ElectronicMapperGenerated>(solarId);
        for (auto dsDetId : dsDetIds) {
          if (dsDetId.dsId() >= 1024 && plane == dcs::Plane::Bending ||
              dsDetId.dsId() < 1024 && plane == dcs::Plane::NonBending) {
            continue;
          }
          dsIndices.emplace(getDsIndex(dsDetId));
        }
      }
    }
  }
  return dsIndices;
}
} // namespace o2::mch::dcs
