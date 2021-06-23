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
#include "MCHConditions/DCSNamer.h"
#include "HVAliases.h"
#include "LVAliases.h"
#include <algorithm>
#include <fmt/format.h>

using namespace o2::mch::dcs;

BOOST_AUTO_TEST_SUITE(o2_mch_conditions)

BOOST_AUTO_TEST_SUITE(dcsnamer)

BOOST_AUTO_TEST_CASE(detElemId2DCSMustReturnNothingIfDetElemIdIsNotValid)
{
  BOOST_CHECK_EQUAL(detElemId2DCS(0).has_value(), false);
  BOOST_CHECK_EQUAL(detElemId2DCS(10).has_value(), false);
  BOOST_CHECK_EQUAL(detElemId2DCS(1026).has_value(), false);
  BOOST_CHECK_EQUAL(detElemId2DCS(-1).has_value(), false);
}

BOOST_AUTO_TEST_CASE(DE100isRight)
{
  BOOST_CHECK(detElemId2DCS(100)->side == Side::Right);
}
BOOST_AUTO_TEST_CASE(DE102isLeft)
{
  BOOST_CHECK(detElemId2DCS(102)->side == Side::Left);
}

BOOST_AUTO_TEST_CASE(detElemId2DCSMustReturnChamberIdAndSideIfDetElemIdIsValid)
{
  auto v = detElemId2DCS(1025);
  BOOST_CHECK_EQUAL(v.has_value(), true);
  BOOST_CHECK_EQUAL(v->chamberId, 9);
  BOOST_CHECK(v->side == Side::Right);
}

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
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
