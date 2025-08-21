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

#include <catch_amalgamated.hpp>
#include "../src/CCDBFetcherHelper.h"

using namespace o2::framework;

TEST_CASE("TestSorting")
{
  auto result = CCDBFetcherHelper::parseRemappings("");
  CHECK(result.error == ""); // not an error

  result = CCDBFetcherHelper::parseRemappings("https");
  CHECK(result.error == "URL should start with either http:// or https:// or file://");

  result = CCDBFetcherHelper::parseRemappings("https://alice.cern.ch:8000");
  CHECK(result.error == "Expecting at least one target path, missing `='?");

  result = CCDBFetcherHelper::parseRemappings("https://alice.cern.ch:8000=");
  CHECK(result.error == "Empty target");

  result = CCDBFetcherHelper::parseRemappings("https://alice.cern.ch:8000=/foo/bar,");
  CHECK(result.error == "Empty target");

  result = CCDBFetcherHelper::parseRemappings("https://alice.cern.ch:8000=/foo/bar,/foo/bar;");
  CHECK(result.error == "URL should start with either http:// or https:// or file://");

  result = CCDBFetcherHelper::parseRemappings("https://alice.cern.ch:8000=/foo/bar,/foo/barbar;file://user/test=/foo/barr");
  CHECK(result.error == "");
  CHECK(result.remappings.size() == 3);
  CHECK(result.remappings["/foo/bar"] == "https://alice.cern.ch:8000");
  CHECK(result.remappings["/foo/barbar"] == "https://alice.cern.ch:8000");
  CHECK(result.remappings["/foo/barr"] == "file://user/test");

  result = CCDBFetcherHelper::parseRemappings("https://alice.cern.ch:8000=/foo/bar;file://user/test=/foo/bar");
  CHECK(result.remappings.size() == 1);
  CHECK(result.error == "Path /foo/bar requested more than once.");
}
