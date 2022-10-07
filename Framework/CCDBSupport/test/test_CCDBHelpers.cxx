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

#define BOOST_TEST_MODULE Test Framework CCDBHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "../src/CCDBHelpers.h"

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestSorting)
{
  auto result = CCDBHelpers::parseRemappings("");
  BOOST_CHECK_EQUAL(result.error, ""); // not an error

  result = CCDBHelpers::parseRemappings("https");
  BOOST_CHECK_EQUAL(result.error, "URL should start with either http:// or https:// or file://");

  result = CCDBHelpers::parseRemappings("https://alice.cern.ch:8000");
  BOOST_CHECK_EQUAL(result.error, "Expecting at least one target path, missing `='?");

  result = CCDBHelpers::parseRemappings("https://alice.cern.ch:8000=");
  BOOST_CHECK_EQUAL(result.error, "Empty target");

  result = CCDBHelpers::parseRemappings("https://alice.cern.ch:8000=/foo/bar,");
  BOOST_CHECK_EQUAL(result.error, "Empty target");

  result = CCDBHelpers::parseRemappings("https://alice.cern.ch:8000=/foo/bar,/foo/bar;");
  BOOST_CHECK_EQUAL(result.error, "URL should start with either http:// or https:// or file://");

  result = CCDBHelpers::parseRemappings("https://alice.cern.ch:8000=/foo/bar,/foo/barbar;file://user/test=/foo/barr");
  BOOST_CHECK_EQUAL(result.error, "");
  BOOST_CHECK_EQUAL(result.remappings.size(), 3);
  BOOST_CHECK_EQUAL(result.remappings["/foo/bar"], "https://alice.cern.ch:8000");
  BOOST_CHECK_EQUAL(result.remappings["/foo/barbar"], "https://alice.cern.ch:8000");
  BOOST_CHECK_EQUAL(result.remappings["/foo/barr"], "file://user/test");

  result = CCDBHelpers::parseRemappings("https://alice.cern.ch:8000=/foo/bar;file://user/test=/foo/bar");
  BOOST_CHECK_EQUAL(result.remappings.size(), 1);
  BOOST_CHECK_EQUAL(result.error, "Path /foo/bar requested more than once.");
}
