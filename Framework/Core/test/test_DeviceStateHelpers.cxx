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
#include "../src/DeviceStateHelpers.h"
#include "Framework/DeviceState.h"

TEST_CASE("TestDeviceStateHelpers")
{
  using namespace o2::framework;
  REQUIRE(DeviceStateHelpers::parseTracingFlags("WS_COMMUNICATION") == DeviceState::LoopReason::WS_COMMUNICATION);
  REQUIRE(DeviceStateHelpers::parseTracingFlags("WS_CONNECTED|WS_COMMUNICATION") == (DeviceState::LoopReason::WS_CONNECTED | DeviceState::LoopReason::WS_COMMUNICATION));
  REQUIRE(DeviceStateHelpers::parseTracingFlags("ABOBODABO") == DeviceState::LoopReason::UNKNOWN);
}
