// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework CallbackRegistry
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/CallbackRegistry.h"
#include <iostream>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestCallbackregistry)
{
  enum class StepId { StateChange,
                      Exit };

  using ExitCallback = std::function<void()>;
  using StateChangeCallback = std::function<void(int)>;

  using Callbacks = CallbackRegistry<StepId,                                                          //
                                     RegistryPair<StepId, StepId::Exit, ExitCallback>,                //
                                     RegistryPair<StepId, StepId::StateChange, StateChangeCallback>>; //
  Callbacks callbacks;
  BOOST_REQUIRE(callbacks.size == 2);

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
  BOOST_CHECK(exitcbWasCalled);
  callbacks(StepId::StateChange, 5);
  BOOST_CHECK(statechangecbWasCalled == 5);
}
