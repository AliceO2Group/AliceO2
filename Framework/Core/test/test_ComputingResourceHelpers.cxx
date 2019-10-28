// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework ComputingResourceHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "../src/ComputingResourceHelpers.h"
#include <string>
#include <vector>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestResourceParsing)
{
  auto test1 = "foo:16:1000:22000:23000";
  auto test2 = "foo:16:1000:22000:23000,bar:8:500:22000:23000";

  auto resources = ComputingResourceHelpers::parseResources(test1);
  BOOST_REQUIRE_EQUAL(resources.size(), 1);
  BOOST_CHECK_EQUAL(resources[0].cpu, 16);
  BOOST_CHECK_EQUAL(resources[0].memory, 1000000000);
  BOOST_CHECK_EQUAL(resources[0].hostname, "foo");
  BOOST_CHECK_EQUAL(resources[0].startPort, 22000);
  BOOST_CHECK_EQUAL(resources[0].lastPort, 23000);

  resources = ComputingResourceHelpers::parseResources(test2);
  BOOST_REQUIRE_EQUAL(resources.size(), 2);
  BOOST_CHECK_EQUAL(resources[0].cpu, 16);
  BOOST_CHECK_EQUAL(resources[0].memory, 1000000000);
  BOOST_CHECK_EQUAL(resources[0].hostname, "foo");
  BOOST_CHECK_EQUAL(resources[0].startPort, 22000);
  BOOST_CHECK_EQUAL(resources[0].lastPort, 23000);
  BOOST_CHECK_EQUAL(resources[1].cpu, 8);
  BOOST_CHECK_EQUAL(resources[1].memory, 500000000);
  BOOST_CHECK_EQUAL(resources[1].hostname, "bar");
  BOOST_CHECK_EQUAL(resources[1].startPort, 22000);
  BOOST_CHECK_EQUAL(resources[1].lastPort, 23000);
}
