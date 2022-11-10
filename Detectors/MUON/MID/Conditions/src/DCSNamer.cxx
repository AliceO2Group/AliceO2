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

#include "MIDConditions/DCSNamer.h"
#include "MIDBase/DetectorParameters.h"
#include <array>
#include <fmt/printf.h>
#include <iostream>
#include <set>
#include <optional>
#include <stdexcept>
#include <fmt/format.h>

namespace
{
const std::string detPrefix = "MID";

// Run2 (0..71) vs Run3 (11xx, 12xx, 13xx, 14xx) detection element ids
//
// INSIDE: 0..35
//
// 1100,1101,1102,1103,1104,
//    4,   5,   6,   7,   8,
// 1114,1115,1116,1117,
//    0,   1,   2,   3,
// 1200,1201,1202,1203,1204,
//   17,  16,  15,  14,  13,
// 1214,1215,1216,1217,
//    9,  10,  11,  12,
// 1300,1301,1302,1303,1304,
//   26,  25,  24,  23,  22,
// 1314,1315,1316,1317,
//   18,  19,  20,  21,
// 1400,1401,1402,1403,1404,
//   35,  34,  33,  32,  31,
// 1414,1415,1416,1417,
//   27,  28,  29, 30,
//
// OUTSIDE : 36..71
//
// 1105,1106,1107,1108,1109,
//   44,  43,  42,  41,  40,
// 1110,1111,1112,1113,
//   39,  38,  37,  36
// 1205,1206,1207,1208,1209,
//   53,  52,  51,  50,  49,
// 1210,1211,1212,1213,
//   48,  47,  46,  45,
// 1305,1306,1307,1308,1309,
//   62,  61,  60,  59,
// 1310,1311,1312,1313,
//   58,  57,  56,  55,  54,
// 1405,1406,1407,1408,1409,
//   71,  70,  69,  68,  67,
// 1410,1411,1412,1413,
//   66,  65,  64,  63,

std::string sideName(o2::mid::dcs::Side side)
{
  if (side == o2::mid::dcs::Side::Inside) {
    return "INSIDE";
  }
  if (side == o2::mid::dcs::Side::Outside) {
    return "OUTSIDE";
  }
  return "INVALID";
}

std::string measurementName(o2::mid::dcs::MeasurementType m)
{
  switch (m) {
    case o2::mid::dcs::MeasurementType::HV_V:
      return "vEff";
    case o2::mid::dcs::MeasurementType::HV_I:
      return "actual.iMon";
  }
  return "INVALID";
}

} // namespace

namespace o2::mid::dcs
{
std::optional<ID> detElemId2DCS(int deId)
{
  if (deId < 0 || deId > 71) {
    return std::nullopt;
  }
  int chamberId = detparams::getChamber(deId);
  int id = 1 + detparams::getRPCLine(deId);
  Side side = detparams::isRightSide(deId) ? Side::Inside : Side::Outside;

  switch (chamberId) {
    case 0:
      chamberId = 11;
      break;
    case 1:
      chamberId = 12;
      break;
    case 2:
      chamberId = 21;
      break;
    case 3:
      chamberId = 22;
      break;
    default:
      throw std::runtime_error(fmt::format("invalid chamberId={} : expected between 0 and 3", chamberId));
  }

  return std::optional<ID>{{id, side, chamberId}};
}

std::string detElemId2DCSAlias(int deId, MeasurementType type)
{
  auto id = detElemId2DCS(deId);
  return fmt::format("{}_{}_MT{}_RPC{}_HV.", detPrefix, sideName(id->side), id->chamberId, id->number) + measurementName(type);
}

std::vector<std::string> aliases(std::vector<MeasurementType> types)
{

  std::vector<std::string> aliases;
  for (auto& type : types) {
    for (auto deId = 0; deId < detparams::NDetectionElements; ++deId) {
      aliases.emplace_back(detElemId2DCSAlias(deId, type));
    }
  }
  return aliases;
}
} // namespace o2::mid::dcs
