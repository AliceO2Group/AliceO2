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

#define BOOST_TEST_MODULE Test MCHConditions DCSNamer
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MCHConditions/DCSAliases.h"
#include "MCHConstants/DetectionElements.h"
#include <algorithm>
#include <fmt/format.h>
#include <map>
#include <string>

using namespace o2::mch::dcs;

namespace
{
struct ID {
  o2::mch::dcs::Chamber chamber;
  o2::mch::dcs::MeasurementType measurement;
  int number;
  o2::mch::dcs::Plane plane;
  o2::mch::dcs::Side side;
  ID(Chamber ch, MeasurementType m, int n, Plane p, Side s) : chamber{ch},
                                                              measurement{m},
                                                              number{n},
                                                              side{s},
                                                              plane{p}
  {
  }
};

bool operator==(const ID& i1, const ID& i2)
{
  return i1.side == i2.side &&
         i1.chamber == i2.chamber &&
         i1.number == i2.number &&
         i1.measurement == i2.measurement &&
         i1.plane == i2.plane;
}

bool operator<(const ID& i1, const ID& i2)
{
  if (i1.plane == i2.plane) {
    if (i1.measurement == i2.measurement) {
      int s1 = i1.side == Side::Left ? 0 : 1;
      int s2 = i2.side == Side::Left ? 0 : 1;
      if (s1 == s2) {
        if (i1.chamber == i2.chamber) {
          return i1.number < i2.number;
        } else {
          return toInt(i1.chamber) < toInt(i2.chamber);
        }
      } else {
        return s1 < s2;
      }
    } else {
      return (int)i1.measurement < (int)i2.measurement;
    }
  } else {
    return (int)i1.plane < (int)i2.plane;
  }
}

} // namespace

namespace o2::mch::dcs
{

extern std::vector<std::string> expectedHVAliasesVoltages;
extern std::vector<std::string> expectedHVAliasesCurrents;
extern std::vector<std::string> expectedLVAliasesFeeAnalog;
extern std::vector<std::string> expectedLVAliasesFeeDigital;
extern std::vector<std::string> expectedLVAliasesSolar;
} // namespace o2::mch::dcs

BOOST_AUTO_TEST_SUITE(o2_mch_conditions)

BOOST_AUTO_TEST_SUITE(dcsnamer)

BOOST_AUTO_TEST_CASE(NumberOfHVAliasesVoltagesIs188)
{
  auto result = aliases({MeasurementType::HV_V});
  BOOST_CHECK_EQUAL(result.size(), expectedHVAliasesVoltages.size());
  BOOST_CHECK_EQUAL(188, expectedHVAliasesVoltages.size());
  bool permutation = std::is_permutation(begin(result), end(result), begin(expectedHVAliasesVoltages));
  BOOST_CHECK_EQUAL(permutation, true);
}

BOOST_AUTO_TEST_CASE(NumberOfHVAliasesCurrentsIs188)
{
  auto result = aliases({MeasurementType::HV_I});
  BOOST_CHECK_EQUAL(result.size(), expectedHVAliasesCurrents.size());
  BOOST_CHECK_EQUAL(188, expectedHVAliasesCurrents.size());
  bool permutation = std::is_permutation(begin(result), end(result), begin(expectedHVAliasesCurrents));
  BOOST_CHECK_EQUAL(permutation, true);
}

BOOST_AUTO_TEST_CASE(NumberOfLVAliasesFeeAnalogIs108)
{
  auto result = aliases({MeasurementType::LV_V_FEE_ANALOG});
  BOOST_CHECK_EQUAL(result.size(), expectedLVAliasesFeeAnalog.size());
  BOOST_CHECK_EQUAL(108, expectedLVAliasesFeeAnalog.size());
  bool permutation = std::is_permutation(begin(result), end(result), begin(expectedLVAliasesFeeAnalog));
  BOOST_CHECK_EQUAL(permutation, true);
}

BOOST_AUTO_TEST_CASE(NumberOfLVAliasesFeeDigitalIs108)
{
  auto result = aliases({MeasurementType::LV_V_FEE_DIGITAL});
  BOOST_CHECK_EQUAL(result.size(), expectedLVAliasesFeeDigital.size());
  BOOST_CHECK_EQUAL(108, expectedLVAliasesFeeDigital.size());
  bool permutation = std::is_permutation(begin(result), end(result), begin(expectedLVAliasesFeeDigital));
  BOOST_CHECK_EQUAL(permutation, true);
}

BOOST_AUTO_TEST_CASE(NumberOfLVAliasesSolarIs112)
{
  auto result = aliases({MeasurementType::LV_V_SOLAR});
  BOOST_CHECK_EQUAL(result.size(), expectedLVAliasesSolar.size());
  BOOST_CHECK_EQUAL(112, expectedLVAliasesSolar.size());
  bool permutation = std::is_permutation(begin(result), end(result), begin(expectedLVAliasesSolar));
  BOOST_CHECK_EQUAL(permutation, true);
}

BOOST_AUTO_TEST_CASE(AliasNameIsShortEnough)
{
  auto result = aliases();
  std::map<size_t, int> sizes;
  constexpr size_t maxLen{62};
  for (auto& a : result) {
    sizes[a.size()]++;
    if (a.size() > maxLen) {
      std::cout << fmt::format("Alias is too long : {:2d} characters, while {:2d} max are allowed : {}\n",
                               a.size(), maxLen, a);
    }
  }
  size_t len{0};

  for (auto p : sizes) {
    std::cout << fmt::format("{:3d} aliases of size {:2d}\n",
                             p.second, p.first);
    len = std::max(len, p.first);
  }
  BOOST_CHECK(len <= 62);
}

BOOST_AUTO_TEST_CASE(AllAliasesShouldBeValid)
{
  auto all = aliases();
  for (auto a : all) {
    BOOST_CHECK_EQUAL(isValid(a), true);
  }
}

std::map<std::string, ID> expected = {
  {"MchHvLvRight/Chamber06Right/Slat08.actual.vMon", {Chamber::Ch06, MeasurementType::HV_V, 8, Plane::Both, Side::Right}},

  {"MchHvLvLeft/Chamber06Left/Group02an.SenseVoltage", {Chamber::Ch06, MeasurementType::LV_V_FEE_ANALOG, 2, Plane::Both, Side::Left}},

  {"MchHvLvRight/Chamber01Right/Quad3Sect2.actual.iMon", {Chamber::Ch01, MeasurementType::HV_I, 32, Plane::Both, Side::Right}},

  {"MchHvLvRight/Chamber01Right/Group04an", {Chamber::Ch01, MeasurementType::LV_V_FEE_ANALOG, 4, Plane::NonBending, Side::Right}},

  {"MchHvLvLeft/Chamber00Left/SolCh00LCr01.SenseVoltage", {Chamber::Ch00, MeasurementType::LV_V_SOLAR, 1, Plane::Both, Side::Left}}

};

BOOST_AUTO_TEST_CASE(AliasToXXX)
{
  for (const auto e : expected) {
    const auto alias = e.first;
    const auto c = aliasToChamber(alias);
    const auto m = aliasToMeasurementType(alias);
    const auto n = aliasToNumber(alias);
    const auto p = aliasToPlane(alias);
    const auto s = aliasToSide(alias);
    BOOST_CHECK_EQUAL(c, e.second.chamber);
    BOOST_CHECK_EQUAL(m, e.second.measurement);
    BOOST_CHECK_EQUAL(n, e.second.number);
    BOOST_CHECK_EQUAL(p, e.second.plane);
    BOOST_CHECK_EQUAL(s, e.second.side);
  }
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
