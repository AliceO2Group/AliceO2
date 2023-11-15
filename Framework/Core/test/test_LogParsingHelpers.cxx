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

#include "Framework/LogParsingHelpers.h"
#include <catch_amalgamated.hpp>
#include <regex>

using namespace o2::framework;

TEST_CASE("TestParseTokenLevel")
{
  REQUIRE(LogParsingHelpers::parseTokenLevel("[10:10:10][ERROR] Some message") == LogParsingHelpers::LogLevel::Error);
  REQUIRE(LogParsingHelpers::parseTokenLevel("[10:10:10][WARN] Some message") == LogParsingHelpers::LogLevel::Warning);
  REQUIRE(LogParsingHelpers::parseTokenLevel("[10:10:10][INFO] Some message") == LogParsingHelpers::LogLevel::Info);
  // Log level STATE is interpreted as INFO
  REQUIRE(LogParsingHelpers::parseTokenLevel("[10:10:10][STATE] Some message") == LogParsingHelpers::LogLevel::Info);
  REQUIRE(LogParsingHelpers::parseTokenLevel("[10:10:10][DEBUG] Some message") == LogParsingHelpers::LogLevel::Debug);
  REQUIRE(LogParsingHelpers::parseTokenLevel("[10:10:10][BLAH] Some message") == LogParsingHelpers::LogLevel::Unknown);
  REQUIRE(LogParsingHelpers::parseTokenLevel("[1010:10][BLAH] Some message") == LogParsingHelpers::LogLevel::Unknown);
  // This fails because we require at least one space after the tagging
  REQUIRE(LogParsingHelpers::parseTokenLevel("[10:10:10][ERROR]") == LogParsingHelpers::LogLevel::Unknown);
}
