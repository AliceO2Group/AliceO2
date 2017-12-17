// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework LogParsingHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/LogParsingHelpers.h"
#include <boost/test/unit_test.hpp>
#include <regex>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestParseTokenLevel) {
  BOOST_CHECK(LogParsingHelpers::parseTokenLevel("[10:10:10][ERROR] Some message") == LogParsingHelpers::LogLevel::Error);
  BOOST_CHECK(LogParsingHelpers::parseTokenLevel("[10:10:10][WARN] Some message") == LogParsingHelpers::LogLevel::Warning);
  BOOST_CHECK(LogParsingHelpers::parseTokenLevel("[10:10:10][INFO] Some message") == LogParsingHelpers::LogLevel::Info);
  BOOST_CHECK(LogParsingHelpers::parseTokenLevel("[10:10:10][DEBUG] Some message") == LogParsingHelpers::LogLevel::Debug);
  BOOST_CHECK(LogParsingHelpers::parseTokenLevel("[10:10:10][BLAH] Some message") == LogParsingHelpers::LogLevel::Unknown);
  BOOST_CHECK(LogParsingHelpers::parseTokenLevel("[1010:10][BLAH] Some message") == LogParsingHelpers::LogLevel::Unknown);
  // This fails because we require at least one space after the tagging
  BOOST_CHECK(LogParsingHelpers::parseTokenLevel("[10:10:10][ERROR]") == LogParsingHelpers::LogLevel::Unknown);
}
