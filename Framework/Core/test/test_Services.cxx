// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework Services
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/ServiceRegistry.h"
#include "Framework/CallbackService.h"
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <memory>

BOOST_AUTO_TEST_CASE(TestServiceRegistry)
{
  using namespace o2::framework;
  struct InterfaceA {
    virtual bool method() = 0;
  };

  struct ConcreteA : InterfaceA {
    bool method() final { return true; }
  };

  struct InterfaceB {
    virtual bool method() = 0;
  };

  struct ConcreteB : InterfaceB {
    bool method() final { return false; }
  };

  struct InterfaceC {
    virtual bool method() const = 0;
  };

  struct ConcreteC : InterfaceC {
    bool method() const final { return false; }
  };

  ServiceRegistry registry;
  ConcreteA serviceA;
  ConcreteB serviceB;
  ConcreteC const serviceC;
  registry.registerService<InterfaceA>(&serviceA);
  registry.registerService<InterfaceB>(&serviceB);
  registry.registerService<InterfaceC>(&serviceC);
  BOOST_CHECK(registry.get<InterfaceA>().method() == true);
  BOOST_CHECK(registry.get<InterfaceB>().method() == false);
  BOOST_CHECK(registry.get<InterfaceC const>().method() == false);
  BOOST_CHECK_THROW(registry.get<InterfaceA const>(), std::runtime_error);
  BOOST_CHECK_THROW(registry.get<InterfaceC>(), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(TestCallbackService)
{
  using namespace o2::framework;
  ServiceRegistry registry;
  auto service = std::make_unique<CallbackService>();
  registry.registerService<CallbackService>(service.get());

  // the callback simply sets the captured variable to indicated that it was called
  bool cbCalled = false;
  auto cb = [&]() { cbCalled = true; };
  registry.get<CallbackService>().set(CallbackService::Id::Stop, cb);

  // check to set with the wrong type
  BOOST_CHECK_THROW(registry.get<CallbackService>().set(CallbackService::Id::Stop, [](int) {}), std::runtime_error);

  // execute and check
  registry.get<CallbackService>()(CallbackService::Id::Stop);
  BOOST_CHECK(cbCalled);
}
