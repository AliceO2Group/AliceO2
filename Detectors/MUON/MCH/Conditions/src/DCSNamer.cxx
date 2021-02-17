// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHConditions/DCSNamer.h"

#include <array>
#include <fmt/printf.h>
#include <iostream>
#include <set>

namespace
{
const uint8_t invalidGroup{99};

std::array<int, 156> detElemIds = {
  100, 101, 102, 103,
  200, 201, 202, 203,
  300, 301, 302, 303,
  400, 401, 402, 403,
  500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517,
  600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617,
  700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725,
  800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825,
  900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925,
  1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025};
bool isQuadrant(int deId)
{
  return deId < 500;
}

bool isSlat(int deId)
{
  return deId >= 500;
}

int nofDetectionElementsInChamber(int chamberId)
{
  // chamberId must be between 4 and 9 (i.e. slats)
  if (chamberId < 6) {
    return 18;
  }
  return 26;
}

std::string basePattern(o2::mch::dcs::Side side)
{
  std::string sside = side == o2::mch::dcs::Side::Left ? "Left" : "Right";
  return "MchHvLv" + sside + "/Chamber%02d" + sside + "/";
}

std::string basePattern(o2::mch::dcs::ID id)
{
  return basePattern(id.side);
}

std::string quadrantHV(int deId, o2::mch::dcs::ID id, int sector)
{
  const auto pattern = basePattern(id) + "Quad%dSect%d";
  return fmt::sprintf(pattern, id.chamberId, id.number, sector);
}

std::string slatHV(int deId, o2::mch::dcs::ID id)
{
  const auto pattern = basePattern(id) + "Slat%02d";
  return fmt::sprintf(pattern, id.chamberId, id.number);
}

std::string hvPattern(int deId, int sector = -1)
{
  auto id = o2::mch::dcs::detElemId2DCS(deId);
  if (!id.has_value()) {
    return "INVALID";
  }
  std::string base;
  if (isQuadrant(deId)) {
    base = quadrantHV(deId, id.value(), sector);
  } else {
    base = slatHV(deId, id.value());
  }
  return base + ".actual.%s";
}

uint8_t quadrantLVGroup(int deId, bool bending)
{
  uint8_t group{invalidGroup};

  // For Chamber 1 to 4 Left the relationship between DCS GUI names and groups is:
  // Quad2B    --> Group3 = DE x01 Non Bending
  // Quad2F    --> Group1 = DE x01 Bending
  // Quad3B    --> Group4 = DE x02 Bending
  // Quad3F    --> Group2 = DE x02 Non Bending
  // for Chamber 1 to 4 Right the relationship is:
  // Quad1B    --> Group3 = DE x00 Bending
  // Quad1F    --> Group1 = DE x00 Non  Bending
  // Quad4B    --> Group4 = DE x03 Non Bending
  // Quad4F    --> Group2 = DE x03 Bending
  // where x = 1,2,3,4
  // and Quad#B = Back = towards IP = cath1
  // while Quad#F = Front = towards muon trigger = cath0
  //
  int remnant = deId % 100;
  switch (remnant) {
    case 0: // DE x00
      group = bending ? 3 : 1;
      break;
    case 1: // DE x01
      group = bending ? 1 : 3;
      break;
    case 2: // DE x02
      group = bending ? 4 : 2;
      break;
    case 3: // DE x03
      group = bending ? 2 : 4;
      break;
    default:
      break;
  }
  return group;
}

uint8_t slatLVGroup(int deId)
{
  uint8_t group{invalidGroup};
  auto id = o2::mch::dcs::detElemId2DCS(deId).value();
  int dcsSlatNumber = 1 + id.number;
  if (id.chamberId == 4 || id.chamberId == 5) {
    switch (dcsSlatNumber) {
      case 1:
      case 2:
      case 3:
        group = 1;
        break;
      case 4:
      case 5:
      case 6:
        group = dcsSlatNumber - 2;
        break;
      case 7:
      case 8:
      case 9:
        group = 5;
        break;
      default:
        break;
    }
  } else {
    switch (dcsSlatNumber) {
      case 1:
      case 2:
      case 3:
      case 4:
        group = 1;
        break;
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
        group = dcsSlatNumber - 3;
        break;
      case 10:
      case 11:
      case 12:
      case 13:
        group = 7;
        break;
      default:
        break;
    }
  }
  return group;
}

std::string lvFEEPattern(int deId, bool bending = true)
{
  auto group = isQuadrant(deId) ? quadrantLVGroup(deId, bending) : slatLVGroup(deId);
  auto id = o2::mch::dcs::detElemId2DCS(deId);
  if (!id.has_value()) {
    return "INVALID";
  }
  auto base = basePattern(id.value()) + "Group%02d";
  auto pattern = fmt::sprintf(base, id->chamberId, group);
  pattern += "%s.MeasurementSenseVoltage";
  return pattern;
}

std::string measurementName(o2::mch::dcs::MeasurementType m)
{
  switch (m) {
    case o2::mch::dcs::MeasurementType::HV_V:
      return "vMon";
    case o2::mch::dcs::MeasurementType::HV_I:
      return "iMon";
    case o2::mch::dcs::MeasurementType::LV_V_FEE_ANALOG:
      return "an";
    case o2::mch::dcs::MeasurementType::LV_V_FEE_DIGITAL:
      return "di";
    case o2::mch::dcs::MeasurementType::LV_V_SOLAR:
      return "Sol";
  }
  return "INVALID";
}

std::vector<std::string> lvSolarPattern(int chamberId)
{
  std::vector<std::string> patterns;

  int nofCrates{0};
  switch (chamberId / 2) {
    case 0:
    case 1:
      nofCrates = 4;
      break;
    case 2:
      nofCrates = 5;
      break;
    case 3:
      nofCrates = 7;
      break;
    case 4:
      nofCrates = 8;
      break;
  };

  for (int i = 1; i <= nofCrates; i++) {
    for (auto side : {o2::mch::dcs::Side::Left, o2::mch::dcs::Side::Right}) {
      auto sideLetter = (side == o2::mch::dcs::Side::Left ? 'L' : 'R');
      auto base = basePattern(side) + "%sCh%02d%cCr%02d.MeasurementSenseVoltage";
      patterns.emplace_back(fmt::sprintf(base, chamberId, measurementName(o2::mch::dcs::MeasurementType::LV_V_SOLAR), chamberId, sideLetter, i));
    }
  }

  return patterns;
}

std::vector<std::string> measurement(const std::vector<std::string>& patterns,
                                     o2::mch::dcs::MeasurementType m)
{
  std::vector<std::string> result;

  result.resize(patterns.size());

  std::transform(patterns.begin(), patterns.end(), result.begin(), [&m](std::string s) {
    auto name = fmt::sprintf(s, measurementName(m));
    return name.substr(0, 62); // name coming from ADAPOS are <= 62 characters
  });

  std::sort(result.begin(), result.end());
  return result;
}

std::vector<std::string> aliasesForHV(std::vector<o2::mch::dcs::MeasurementType> types)
{
  std::vector<std::string> patterns;

  /// 188 HV channels
  ///
  /// St 1 ch  1 : 12 channels
  ///      ch  2 : 12 channels
  /// St 2 ch  3 : 12 channels
  ///      ch  4 : 12 channels
  /// St 3 ch  5 : 18 channels
  ///      ch  6 : 18 channels
  /// St 4 ch  7 : 26 channels
  ///      ch  8 : 26 channels
  /// St 5 ch  9 : 26 channels
  ///      ch 10 : 26 channels
  ///
  for (auto deId : detElemIds) {
    if (isQuadrant(deId)) {
      for (auto sector = 0; sector < 3; ++sector) {
        patterns.emplace_back(hvPattern(deId, sector));
      }
    } else {
      patterns.emplace_back(hvPattern(deId));
    }
  }

  std::vector<std::string> aliases;

  for (auto m : types) {
    if (m == o2::mch::dcs::MeasurementType::HV_V || m == o2::mch::dcs::MeasurementType::HV_I) {
      auto aliasPerType = measurement(patterns, m);
      aliases.insert(aliases.begin(), aliasPerType.begin(), aliasPerType.end());
    }
  }

  return aliases;
}

std::vector<std::string> aliasesForLVFEE(std::vector<o2::mch::dcs::MeasurementType> types)
{
  std::set<std::string> patterns;

  /// 108 aliases per voltage (analog or digital) for front end card
  ///
  /// St 1 ch  1 left or right : 4 groups
  ///      ch  2 left or right : 4 groups
  /// St 2 ch  3 left or right : 4 groups
  ///      ch  4 left or right : 4 groups
  /// St 3 ch  5 left or right : 5 groups
  ///      ch  6 left or right : 5 groups
  /// St 4 ch  7 left or right : 7 groups
  ///      ch  8 left or right : 7 groups
  /// St 5 ch  9 left or right : 7 groups
  ///      ch 10 left or right : 7 groups
  ///
  for (auto deId : detElemIds) {
    if (isQuadrant(deId)) {
      for (auto bending : {true, false}) {
        patterns.insert(lvFEEPattern(deId, bending));
      }
    } else {
      patterns.insert(lvFEEPattern(deId));
    }
  }

  std::vector<std::string> uniquePatterns(patterns.begin(), patterns.end());
  std::vector<std::string> aliases;

  for (auto m : types) {
    if (m == o2::mch::dcs::MeasurementType::LV_V_FEE_ANALOG || m == o2::mch::dcs::MeasurementType::LV_V_FEE_DIGITAL) {
      auto aliasPerType = measurement(uniquePatterns, m);
      aliases.insert(aliases.begin(), aliasPerType.begin(), aliasPerType.end());
    }
  }

  return aliases;
}

std::vector<std::string> aliasesForLVSolar(std::vector<o2::mch::dcs::MeasurementType> types)
{
  if (std::find(types.begin(), types.end(), o2::mch::dcs::MeasurementType::LV_V_SOLAR) != types.end()) {
    std::vector<std::string> patterns;

    /// 112 voltages for SOLAR cards
    for (auto chamberId = 0; chamberId < 10; chamberId++) {
      auto pats = lvSolarPattern(chamberId);
      patterns.insert(patterns.begin(), pats.begin(), pats.end());
    }

    return measurement(patterns, o2::mch::dcs::MeasurementType::LV_V_SOLAR);
  }
  return {};
}

std::vector<std::string> aliasesForLV(std::vector<o2::mch::dcs::MeasurementType> types)
{
  auto fee = aliasesForLVFEE(types);
  auto solar = aliasesForLVSolar(types);

  std::vector<std::string> aliases{fee};
  aliases.insert(aliases.begin(), solar.begin(), solar.end());
  return aliases;
}
} // namespace

namespace o2::mch::dcs
{
std::optional<ID> detElemId2DCS(int deId)
{
  if (std::find(detElemIds.begin(), detElemIds.end(), deId) == detElemIds.end()) {
    return std::nullopt;
  }
  int chamberId = deId / 100 - 1;
  int id = deId - (chamberId + 1) * 100;

  Side side = Side::Left;

  if (isQuadrant(deId)) {
    if (id == 0 || id == 3) {
      side = Side::Right;
    } else {
      side = Side::Left;
    }
  } else {
    int nofDe = nofDetectionElementsInChamber(chamberId);
    int quarter = nofDe / 4;
    int half = nofDe / 2;
    int threeQuarter = quarter + half;
    if (id <= quarter) {
      id += quarter + 1;
      side = Side::Right;
    } else if (id <= threeQuarter) {
      id = (threeQuarter - id + 1);
      side = Side::Left;
    } else {
      id -= threeQuarter;
      side = Side::Right;
    }
    // dcs convention change : numbering from top, not from bottom
    id = half - id;
  }

  return std::optional<ID>{{id, side, chamberId}};
}

std::vector<std::string> aliases(std::vector<MeasurementType> types)
{
  auto hv = aliasesForHV(types);
  auto lv = aliasesForLV(types);

  auto aliases = hv;
  aliases.insert(aliases.begin(), lv.begin(), lv.end());

  return aliases;
}

} // namespace o2::mch::dcs
