// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHRaw SampaCluster
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <fstream>
#include <fmt/printf.h>
#include "MCHRawCommon/SampaCluster.h"
#include <array>

using namespace o2::mch::raw;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(sampacluster)

uint16_t defaultTimestamp{0x3FF};
uint32_t defaultChargeSum{0xFFFF};
std::vector<uint16_t> defaultSamples = {0x3FF, 0x3FF, 0x3FF};

BOOST_AUTO_TEST_CASE(CtorWithValidArgumentsMustNotThrow)
{
  BOOST_CHECK_NO_THROW(SampaCluster sc(defaultTimestamp, defaultChargeSum));
  BOOST_CHECK_NO_THROW(SampaCluster sc(defaultTimestamp, defaultSamples));
}

BOOST_AUTO_TEST_CASE(CtorWithInvalidTimeStampMustThrow)
{
  BOOST_CHECK_THROW(SampaCluster sc(1 << 10, defaultChargeSum), std::invalid_argument);
  BOOST_CHECK_THROW(SampaCluster sc(1 << 10, defaultSamples), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(ElementarySizeShouldBe40BitsInClusterSumMode)
{
  SampaCluster sc(defaultTimestamp, defaultChargeSum);
  BOOST_CHECK_EQUAL(sc.nof10BitWords(), 4);
}

BOOST_AUTO_TEST_CASE(CtorWithInvalidChargeSumMustThrow)
{
  BOOST_CHECK_THROW(SampaCluster sc(defaultTimestamp, 0x1FFFFF), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(CtorWithInvalidSamplesMustThrow)
{
  std::vector<uint16_t> invalidSamples = {0x3FF, 0x3FF, 0x4FF};
  BOOST_CHECK_THROW(SampaCluster sc(defaultTimestamp, invalidSamples), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
