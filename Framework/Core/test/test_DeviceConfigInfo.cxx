// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework DeviceConfigInfo
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/DeviceConfigInfo.h"
#include "Framework/DeviceInfo.h"
#include <boost/test/unit_test.hpp>
#include <string_view>

BOOST_AUTO_TEST_CASE(TestDeviceMetricsInfo)
{
  using namespace o2::framework;
  std::string configString;
  bool result;
  ParsedConfigMatch match;
  DeviceInfo info;

  // Something which does not match
  configString = "foo[NOCONFIG] foo=bar 1789372894\n";
  std::string_view config1{configString.data() + 3, configString.size() - 4};
  result = DeviceConfigHelper::parseConfig(config1, match);
  BOOST_REQUIRE_EQUAL(result, false);

  // Something which does not match
  configString = "foo[XX:XX:XX][INFO] [CONFIG] foobar 1789372894\n";
  std::string_view config2{configString.data() + 3, configString.size() - 4};
  result = DeviceConfigHelper::parseConfig(config2, match);
  BOOST_REQUIRE_EQUAL(result, false);

  // Something which does not match
  configString = "foo[XX:XX:XX][INFO] [CONFIG] foo=bar1789372894\n";
  std::string_view config3{configString.data() + 3, configString.size() - 4};
  result = DeviceConfigHelper::parseConfig(config3, match);
  BOOST_REQUIRE_EQUAL(result, false);

  // Something which does not match
  configString = "foo[XX:XX:XX][INFO] [CONFIG] foo=bar\n";
  std::string_view config4{configString.data() + 3, configString.size() - 4};
  result = DeviceConfigHelper::parseConfig(config4, match);
  BOOST_REQUIRE_EQUAL(result, false);

  // Something which does not match
  configString = "foo[XX:XX:XX][INFO] [CONFIG] foo=bar 1789372894t\n";
  std::string_view config5{configString.data() + 3, configString.size() - 4};
  result = DeviceConfigHelper::parseConfig(config5, match);
  BOOST_REQUIRE_EQUAL(result, false);

  // Parse a simple configuration bit
  configString = "foo[XX:XX:XX][INFO] [CONFIG] foo=bar 1789372894\n";
  std::string_view config6{configString.data() + 3, configString.size() - 4};
  result = DeviceConfigHelper::parseConfig(config6, match);
  BOOST_REQUIRE_EQUAL(result, false);

  // Parse a simple configuration bit
  configString = "foo[XX:XX:XX][INFO] [CONFIG] foo=bar 1789372894 prov\n";
  std::string_view config{configString.data() + 3, configString.size() - 4};
  BOOST_REQUIRE_EQUAL(config, std::string("[XX:XX:XX][INFO] [CONFIG] foo=bar 1789372894 prov"));
  result = DeviceConfigHelper::parseConfig(config, match);
  BOOST_REQUIRE_EQUAL(result, true);
  BOOST_CHECK(strncmp(match.beginKey, "foo", 3) == 0);
  BOOST_CHECK_EQUAL(match.timestamp, 1789372894);
  BOOST_CHECK(strncmp(match.beginValue, "bar", 3) == 0);
  BOOST_CHECK(strncmp(match.beginProvenance, "prov", 4) == 0);

  // Process a given config entry
  result = DeviceConfigHelper::processConfig(match, info);
  BOOST_CHECK_EQUAL(result, true);
  BOOST_CHECK_EQUAL(info.currentConfig.get<std::string>("foo"), "bar");
  BOOST_CHECK_EQUAL(info.currentProvenance.get<std::string>("foo"), "prov");
}
