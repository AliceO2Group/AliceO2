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

#include "Framework/DeviceConfigInfo.h"
#include "Framework/DeviceInfo.h"
#include "Framework/Variant.h"
#include <catch_amalgamated.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <string_view>
#include <regex>

template <typename T>
std::string arrayPrinter(boost::property_tree::ptree const& tree)
{
  std::stringstream ss;
  int size = tree.size();
  int count = 0;
  ss << o2::framework::variant_array_symbol<T>::symbol << "[";
  for (auto& element : tree) {
    ss << element.second.get_value<T>();
    if (count < size - 1) {
      ss << ",";
    }
    ++count;
  }
  ss << "]";
  return ss.str();
}

TEST_CASE("TestDeviceConfigInfo")
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
  REQUIRE(result == false);

  // Something which does not match
  configString = "foo[CONFIG] foobar 1789372894\n";
  std::string_view config2{configString.data() + 3, configString.size() - 4};
  result = DeviceConfigHelper::parseConfig(config2, match);
  REQUIRE(result == false);

  // Something which does not match
  configString = "foo[CONFIG] foo=bar1789372894\n";
  std::string_view config3{configString.data() + 3, configString.size() - 4};
  result = DeviceConfigHelper::parseConfig(config3, match);
  REQUIRE(result == false);

  // Something which does not match
  configString = "foo[CONFIG] foo=bar\n";
  std::string_view config4{configString.data() + 3, configString.size() - 4};
  result = DeviceConfigHelper::parseConfig(config4, match);
  REQUIRE(result == false);

  // Something which does not match
  configString = "foo[CONFIG] foo=bar 1789372894t\n";
  std::string_view config5{configString.data() + 3, configString.size() - 4};
  result = DeviceConfigHelper::parseConfig(config5, match);
  REQUIRE(result == false);

  // Parse a simple configuration bit
  configString = "foo[CONFIG] foo=bar 1789372894\n";
  std::string_view config6{configString.data() + 3, configString.size() - 4};
  result = DeviceConfigHelper::parseConfig(config6, match);
  REQUIRE(result == false);

  // Parse a simple configuration bit
  configString = "foo[CONFIG] foo=bar 1789372894 prov\n";
  std::string_view config{configString.data() + 3, configString.size() - 4};
  REQUIRE(config == std::string("[CONFIG] foo=bar 1789372894 prov"));
  result = DeviceConfigHelper::parseConfig(config, match);
  REQUIRE(result == true);
  REQUIRE(strncmp(match.beginKey, "foo", 3) == 0);
  REQUIRE(match.timestamp == 1789372894);
  REQUIRE(strncmp(match.beginValue, "bar", 3) == 0);
  REQUIRE(strncmp(match.beginProvenance, "prov", 4) == 0);

  // Parse a simple configuration bit with a space in the value
  configString = "foo[CONFIG] foo=bar and foo 1789372894 prov\n";
  std::string_view configWithSpace{configString.data() + 3, configString.size() - 4};
  REQUIRE(configWithSpace == std::string("[CONFIG] foo=bar and foo 1789372894 prov"));
  result = DeviceConfigHelper::parseConfig(configWithSpace, match);
  REQUIRE(result == true);
  REQUIRE(strncmp(match.beginKey, "foo", 3) == 0);
  REQUIRE(match.timestamp == 1789372894);
  REQUIRE(strncmp(match.beginValue, "bar and foo", 11) == 0);
  REQUIRE(strncmp(match.beginProvenance, "prov", 4) == 0);

  // Process a given config entry
  result = DeviceConfigHelper::processConfig(match, info);
  REQUIRE(result == true);
  REQUIRE(info.currentConfig.get<std::string>("foo") == "bar and foo");
  REQUIRE(info.currentProvenance.get<std::string>("foo") == "prov");

  // Parse an array
  configString = "foo[CONFIG] array={\"\":\"1\",\"\":\"2\",\"\":\"3\",\"\":\"4\",\"\":\"5\"} 1789372894 prov\n";
  std::string_view configa{configString.data() + 3, configString.size() - 4};
  result = DeviceConfigHelper::parseConfig(configa, match);
  auto valueString = std::string(match.beginValue, match.endValue - match.beginValue);

  REQUIRE(result == true);
  REQUIRE(strncmp(match.beginKey, "array", 5) == 0);
  REQUIRE(match.timestamp == 1789372894);
  REQUIRE(strncmp(match.beginValue, "{\"\":\"1\",\"\":\"2\",\"\":\"3\",\"\":\"4\",\"\":\"5\"}", 35) == 0);
  REQUIRE(strncmp(match.beginProvenance, "prov", 4) == 0);

  // Process a given config entry
  result = DeviceConfigHelper::processConfig(match, info);
  REQUIRE(result == true);
  REQUIRE(info.currentProvenance.get<std::string>("array") == "prov");
  REQUIRE(arrayPrinter<int>(info.currentConfig.get_child("array")) == "i[1,2,3,4,5]");
}
