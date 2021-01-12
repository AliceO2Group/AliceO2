// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MIDConditions DCSNamer
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MIDConditions/DCSNamer.h"
#include "MIDBase/DetectorParameters.h"
#include "HVAliases.h"
#include <algorithm>

using namespace o2::mid::dcs;

BOOST_AUTO_TEST_SUITE(o2_mid_conditions)

BOOST_AUTO_TEST_SUITE(dcsnamer)

BOOST_AUTO_TEST_CASE(detElemId2DCSMustReturnNothingIfDetElemIdIsNotValid)
{
  BOOST_CHECK_EQUAL(detElemId2DCS(72).has_value(), false);
  BOOST_CHECK_EQUAL(detElemId2DCS(1026).has_value(), false);
  BOOST_CHECK_EQUAL(detElemId2DCS(-1).has_value(), false);
}

BOOST_AUTO_TEST_CASE(DE4isInside)
{
  BOOST_CHECK(detElemId2DCS(4)->side == Side::Inside);
}
BOOST_AUTO_TEST_CASE(DE36isOutside)
{
  BOOST_CHECK(detElemId2DCS(36)->side == Side::Outside);
}

BOOST_AUTO_TEST_CASE(detElemId2DCSMustReturnChamberIdAndSideIfDetElemIdIsValid)
{
  auto v = detElemId2DCS(24);
  BOOST_CHECK_EQUAL(v.has_value(), true);
  BOOST_CHECK_EQUAL(v->chamberId, 21);
  BOOST_CHECK(v->side == Side::Inside);
}

BOOST_AUTO_TEST_CASE(NumberOfHVAliasesVoltagesIs72)
{
  auto result = aliases({MeasurementType::HV_V});
  BOOST_CHECK_EQUAL(result.size(), expectedHVAliasesVoltages.size());
  BOOST_CHECK_EQUAL(72, expectedHVAliasesVoltages.size());
  bool permutation = std::is_permutation(begin(result), end(result), begin(expectedHVAliasesVoltages));
  BOOST_CHECK_EQUAL(permutation, true);
}

BOOST_AUTO_TEST_CASE(NumberOfHVAliasesCurrentsIs72)
{
  auto result = aliases({MeasurementType::HV_I});
  BOOST_CHECK_EQUAL(result.size(), expectedHVAliasesCurrents.size());
  BOOST_CHECK_EQUAL(72, expectedHVAliasesCurrents.size());
  bool permutation = std::is_permutation(begin(result), end(result), begin(expectedHVAliasesCurrents));
  BOOST_CHECK_EQUAL(permutation, true);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
