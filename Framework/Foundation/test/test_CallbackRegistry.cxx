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
#include "Framework/CallbackRegistry.h"
#include <iostream>

using namespace o2::framework;

TEST_CASE("TestCallbackregistry")
{
  enum class StepId { Void,
                      Int };

  using VoidCallback = std::function<void()>;
  using IntCallback = std::function<void(int)>;

  using Callbacks = CallbackRegistry<StepId,                                           //
                                     RegistryPair<StepId, StepId::Void, VoidCallback>, //
                                     RegistryPair<StepId, StepId::Int, IntCallback>>;  //
  Callbacks callbacks;
  REQUIRE(callbacks.size == 2);

  bool voidWasCalled = false;
  auto voidcb = [&]() {
    voidWasCalled = true;
  };
  callbacks.set<StepId::Void>(voidcb);
  int intWasCalled = -1;
  auto intcb = [&](int val) {
    intWasCalled = val;
  };
  callbacks.set<StepId::Int>(intcb);

  callbacks.call<StepId::Void>();
  REQUIRE(voidWasCalled);
  callbacks.call<StepId::Int>(5);
  REQUIRE(intWasCalled == 5);

  SECTION("test resetting")
  {
    auto intcb2 = [&](int val) {
      intWasCalled = 4;
    };
    callbacks.set<StepId::Int>(intcb2);
    callbacks.call<StepId::Int>(6);
    REQUIRE(intWasCalled == 4);
  }

  SECTION("test mutable")
  {
    int invoked = 0;
    callbacks.set<StepId::Int>([&invoked](int val) { invoked = val; });
    callbacks.call<StepId::Int>(6);
    REQUIRE(invoked == 6);
  }
}
