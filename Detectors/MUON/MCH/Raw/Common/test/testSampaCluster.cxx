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
uint32_t defaultBunchCrossing{0xFFFFF};
uint32_t defaultChargeSum{0xFFFFF};
uint32_t defaultChargeSumSize{0x3FF};
std::vector<uint16_t> defaultSamples = {0x3FF, 0x3FF, 0x3FF};

BOOST_AUTO_TEST_CASE(CtorWithValidArgumentsMustNotThrow)
{
  BOOST_CHECK_NO_THROW(SampaCluster sc(defaultTimestamp, defaultBunchCrossing, defaultChargeSum, defaultChargeSumSize));
  BOOST_CHECK_NO_THROW(SampaCluster sc(defaultTimestamp, defaultBunchCrossing, defaultSamples));
}

BOOST_AUTO_TEST_CASE(CtorWithInvalidTimeStampMustThrow)
{
  BOOST_CHECK_THROW(SampaCluster sc(1 << 10, defaultBunchCrossing, defaultChargeSum, defaultChargeSumSize), std::invalid_argument);
  BOOST_CHECK_THROW(SampaCluster sc(1 << 10, defaultBunchCrossing, defaultSamples), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(CtorWithInvalidBunchCrossingMustThrow)
{
  BOOST_CHECK_THROW(SampaCluster sc(defaultTimestamp, 1 << 20, defaultChargeSum, defaultChargeSumSize), std::invalid_argument);
  BOOST_CHECK_THROW(SampaCluster sc(defaultTimestamp, 1 << 20, defaultSamples), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(ElementarySizeShouldBe40BitsInClusterSumMode)
{
  SampaCluster sc(defaultTimestamp, defaultBunchCrossing, defaultChargeSum, defaultChargeSumSize);
  BOOST_CHECK_EQUAL(sc.nof10BitWords(), 4);
}

BOOST_AUTO_TEST_CASE(CtorWithInvalidChargeSumMustThrow)
{
  BOOST_CHECK_THROW(SampaCluster sc(defaultTimestamp, defaultBunchCrossing, 0x1FFFFF, defaultChargeSumSize), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(CtorWithInvalidChargeSumSizeMustThrow)
{
  BOOST_CHECK_THROW(SampaCluster sc(defaultTimestamp, defaultBunchCrossing, defaultChargeSum, 1 << 10), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(CtorWithInvalidSamplesMustThrow)
{
  std::vector<uint16_t> invalidSamples = {0x3FF, 0x3FF, 0x4FF};
  BOOST_CHECK_THROW(SampaCluster sc(defaultTimestamp, defaultBunchCrossing, invalidSamples), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(CtorWithNoSamplesMustThrow)
{
  std::vector<uint16_t> empty;
  BOOST_CHECK_THROW(SampaCluster sc(defaultTimestamp, defaultBunchCrossing, empty), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(AssertNotMixingShouldThrowIfClustersOfMixedSampleType)
{
  std::vector<SampaCluster> clusters;
  clusters.emplace_back(defaultTimestamp, defaultBunchCrossing, defaultChargeSum, defaultChargeSumSize);
  clusters.emplace_back(defaultTimestamp, defaultBunchCrossing, defaultSamples);
  BOOST_CHECK_THROW(assertNotMixingClusters<ChargeSumMode>(clusters), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(AssertNotMixingShouldThrowIfClustersOfDifferentBunchCrossing)
{
  std::vector<SampaCluster> clusters;
  clusters.emplace_back(defaultTimestamp, defaultBunchCrossing, defaultChargeSum, defaultChargeSumSize);
  clusters.emplace_back(defaultTimestamp, defaultBunchCrossing - 1, defaultChargeSum, defaultChargeSumSize);
  BOOST_CHECK_THROW(assertNotMixingClusters<ChargeSumMode>(clusters), std::invalid_argument);
}
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
