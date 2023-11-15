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
#include "MCHConditions/MeasurementType.h"
#include "MCHConditions/Number.h"

namespace o2::mch::dcs
{
namespace slat
{
int nofDetectionElementsInChamber(Chamber chamber)
{
  switch (chamber) {
    case Chamber::Ch06:
    case Chamber::Ch07:
    case Chamber::Ch08:
    case Chamber::Ch09:
      return 26;
    default:
      return 18;
  }
}

int detElemId(Chamber chamber, Side side, int number)
{
  auto nofDE = nofDetectionElementsInChamber(chamber);
  auto half = nofDE / 2;
  auto quarter = nofDE / 4;
  auto threeQuarter = half + quarter;
  auto dcsNumber = half - number;
  int de;
  if (side == dcs::Side::Left) {
    de = threeQuarter + 1 - dcsNumber;
  } else {
    if (dcsNumber <= quarter) {
      de = dcsNumber + threeQuarter;
    } else {
      de = dcsNumber - quarter - 1;
    }
  }
  return (toInt(chamber) + 1) * 100 + de;
}
} // namespace slat

namespace quadrant
{
int detElemId(Chamber chamber, int number)
{
  int quad = number / 10;
  return 100 * (toInt(chamber) + 1) + quad;
}
} // namespace quadrant

int detElemId(Chamber chamber, Side side, int number)
{
  if (isSlat(chamber)) {
    return slat::detElemId(chamber, side, number);
  } else {
    return quadrant::detElemId(chamber, number);
  }
}

std::optional<int> aliasToDetElemId(std::string_view dcsAlias)
{
  const auto m = aliasToMeasurementType(dcsAlias);
  if (m == MeasurementType::HV_V ||
      m == MeasurementType::HV_I) {
    const auto chamber = aliasToChamber(dcsAlias);
    int number = aliasToNumber(dcsAlias);
    if (isSlat(chamber)) {
      return slat::detElemId(chamber,
                             aliasToSide(dcsAlias),
                             number);
    } else {
      return quadrant::detElemId(chamber, number);
    }
  }
  return std::nullopt;
}

} // namespace o2::mch::dcs
