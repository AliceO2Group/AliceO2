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

#include "MCHConditions/Plane.h"

#include "MCHConditions/Chamber.h"
#include "MCHConditions/MeasurementType.h"
#include "MCHConditions/Number.h"
#include "MCHConditions/Side.h"
#include <algorithm>
#include <fmt/core.h>
#include <stdexcept>

namespace o2::mch::dcs
{

Plane aliasToPlane(std::string_view alias)
{
  auto chamber = aliasToChamber(alias);
  if (isQuadrant(chamber)) {
    // only in St12 can we get a dcs alias that is plane-dependent
    // and only if the alias is for FEE LV
    auto type = aliasToMeasurementType(alias);
    if (type != MeasurementType::LV_V_FEE_ANALOG &&
        type != MeasurementType::LV_V_FEE_DIGITAL) {
      return Plane::Both;
    }
    auto number = aliasToNumber(alias);
    auto side = aliasToSide(alias);
    if (side == Side::Left) {
      switch (number) {
        case 1:
        case 4:
          return Plane::Bending;
        case 2:
        case 3:
          return Plane::NonBending;
        default:
          throw std::invalid_argument(fmt::format("invalid number {} in alias={}", number, alias));
      };
    } else {
      switch (number) {
        case 2:
        case 3:
          return Plane::Bending;
        case 1:
        case 4:
          return Plane::NonBending;
        default:
          throw std::invalid_argument(fmt::format("invalid number {} in alias={}", number, alias));
      };
    }
  }
  return Plane::Both;
}

std::string name(Plane p)
{
  switch (p) {
    case Plane::Bending:
      return "B";
    case Plane::NonBending:
      return "NB";
    case Plane::Both:
      return "";
  }
  return "INVALID";
}
} // namespace o2::mch::dcs
