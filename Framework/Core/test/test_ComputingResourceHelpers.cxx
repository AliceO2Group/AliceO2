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

#include "../src/ComputingResourceHelpers.h"
#include <string>
#include <vector>

using namespace o2::framework;

TEST_CASE("TestResourceParsing")
{
  auto test1 = "foo:16:1000:22000:23000";
  auto test2 = "foo:16:1000:22000:23000,bar:8:500:22000:23000";

  auto resources = ComputingResourceHelpers::parseResources(test1);
  REQUIRE(resources.size() == 1);
  REQUIRE(resources[0].cpu == 16);
  REQUIRE(resources[0].memory == 1000000000);
  REQUIRE(resources[0].hostname == "foo");
  REQUIRE(resources[0].startPort == 22000);
  REQUIRE(resources[0].lastPort == 23000);

  resources = ComputingResourceHelpers::parseResources(test2);
  REQUIRE(resources.size() == 2);
  REQUIRE(resources[0].cpu == 16);
  REQUIRE(resources[0].memory == 1000000000);
  REQUIRE(resources[0].hostname == "foo");
  REQUIRE(resources[0].startPort == 22000);
  REQUIRE(resources[0].lastPort == 23000);
  REQUIRE(resources[1].cpu == 8);
  REQUIRE(resources[1].memory == 500000000);
  REQUIRE(resources[1].hostname == "bar");
  REQUIRE(resources[1].startPort == 22000);
  REQUIRE(resources[1].lastPort == 23000);
}
