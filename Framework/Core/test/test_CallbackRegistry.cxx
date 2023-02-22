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
  enum class StepId { StateChange,
                      Exit };

  using ExitCallback = std::function<void()>;
  using StateChangeCallback = std::function<void(int)>;

  using Callbacks = CallbackRegistry<StepId,                                                          //
                                     RegistryPair<StepId, StepId::Exit, ExitCallback>,                //
                                     RegistryPair<StepId, StepId::StateChange, StateChangeCallback>>; //
  Callbacks callbacks;
  REQUIRE(callbacks.size == 2);

  bool exitcbWasCalled = false;
  auto exitcb = [&]() {
    exitcbWasCalled = true;
    std::cout << "Exit callback executed" << std::endl;
  };
  callbacks.set(StepId::Exit, exitcb);
  int statechangecbWasCalled = -1;
  auto statechangecb = [&](int val) {
    statechangecbWasCalled = val;
    std::cout << "StateChange callback executed with argument " << val << std::endl;
  };
  callbacks.set(StepId::StateChange, statechangecb);

  callbacks(StepId::Exit);
  REQUIRE(exitcbWasCalled);
  callbacks(StepId::StateChange, 5);
  REQUIRE(statechangecbWasCalled == 5);
}
