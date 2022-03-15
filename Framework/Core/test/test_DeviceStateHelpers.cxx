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
#define BOOST_TEST_MODULE Test Framework DeviceSpec
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "../src/DeviceStateHelpers.h"
#include "Framework/DeviceState.h"

BOOST_AUTO_TEST_CASE(TestDeviceStateHelpers)
{
  using namespace o2::framework;
  BOOST_CHECK(DeviceStateHelpers::parseTracingFlags("WS_COMMUNICATION") == DeviceState::LoopReason::WS_COMMUNICATION);
  BOOST_CHECK(DeviceStateHelpers::parseTracingFlags("WS_CONNECTED|WS_COMMUNICATION") == (DeviceState::LoopReason::WS_CONNECTED | DeviceState::LoopReason::WS_COMMUNICATION));
  BOOST_CHECK(DeviceStateHelpers::parseTracingFlags("ABOBODABO") == DeviceState::LoopReason::UNKNOWN);
}
