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
#define BOOST_TEST_MODULE Test Framework Services
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/ServiceHandle.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/CallbackService.h"
#include "Framework/CommonServices.h"
#include <Framework/DeviceState.h>
#include <boost/test/unit_test.hpp>
#include <fairmq/ProgOptions.h>
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
  registry.registerService(ServiceRegistryHelpers::handleForService<InterfaceA>(&serviceA));
  registry.registerService(ServiceRegistryHelpers::handleForService<InterfaceB>(&serviceB));
  registry.registerService(ServiceRegistryHelpers::handleForService<InterfaceC const>(&serviceC));
  BOOST_CHECK(registry.get<InterfaceA>().method() == true);
  BOOST_CHECK(registry.get<InterfaceB>().method() == false);
  BOOST_CHECK(registry.get<InterfaceC const>().method() == false);
  BOOST_CHECK(registry.active<InterfaceA>() == true);
  BOOST_CHECK(registry.active<InterfaceB>() == true);
  BOOST_CHECK(registry.active<InterfaceC>() == false);
  BOOST_CHECK_THROW(registry.get<InterfaceA const>(), RuntimeErrorRef);
  BOOST_CHECK_THROW(registry.get<InterfaceC>(), RuntimeErrorRef);
}

BOOST_AUTO_TEST_CASE(TestCallbackService)
{
  using namespace o2::framework;
  ServiceRegistry registry;
  auto service = std::make_unique<CallbackService>();
  registry.registerService(ServiceRegistryHelpers::handleForService<CallbackService>(service.get()));

  // the callback simply sets the captured variable to indicated that it was called
  bool cbCalled = false;
  auto cb = [&]() { cbCalled = true; };
  registry.get<CallbackService>().set(CallbackService::Id::Stop, cb);

  // check to set with the wrong type
  BOOST_CHECK_THROW(registry.get<CallbackService>().set(CallbackService::Id::Stop, [](int) {}), RuntimeErrorRef);

  // execute and check
  registry.get<CallbackService>()(CallbackService::Id::Stop);
  BOOST_CHECK(cbCalled);
}

struct DummyService {
  int threadId;
};

BOOST_AUTO_TEST_CASE(TestSerialServices)
{
  using namespace o2::framework;
  ServiceRegistry registry;

  DummyService t0{0};
  /// We register it pretending to be on thread 0
  registry.registerService(TypeIdHelpers::uniqueId<DummyService>(), &t0, ServiceKind::Serial, 0);

  auto tt0 = reinterpret_cast<DummyService*>(registry.get(TypeIdHelpers::uniqueId<DummyService>(), 0, ServiceKind::Serial));
  auto tt1 = reinterpret_cast<DummyService*>(registry.get(TypeIdHelpers::uniqueId<DummyService>(), 1, ServiceKind::Serial));
  auto tt2 = reinterpret_cast<DummyService*>(registry.get(TypeIdHelpers::uniqueId<DummyService>(), 2, ServiceKind::Serial));
  BOOST_CHECK_EQUAL(tt0->threadId, 0);
  BOOST_CHECK_EQUAL(tt1->threadId, 0);
  BOOST_CHECK_EQUAL(tt2->threadId, 0);
}

BOOST_AUTO_TEST_CASE(TestGlobalServices)
{
  using namespace o2::framework;
  ServiceRegistry registry;

  DummyService t0{0};
  /// We register it pretending to be on thread 0
  registry.registerService(TypeIdHelpers::uniqueId<DummyService>(), &t0, ServiceKind::Global, 0);

  auto tt0 = reinterpret_cast<DummyService*>(registry.get(TypeIdHelpers::uniqueId<DummyService>(), 0, ServiceKind::Serial));
  auto tt1 = reinterpret_cast<DummyService*>(registry.get(TypeIdHelpers::uniqueId<DummyService>(), 1, ServiceKind::Serial));
  auto tt2 = reinterpret_cast<DummyService*>(registry.get(TypeIdHelpers::uniqueId<DummyService>(), 2, ServiceKind::Serial));
  BOOST_CHECK_EQUAL(tt0->threadId, 0);
  BOOST_CHECK_EQUAL(tt1->threadId, 0);
  BOOST_CHECK_EQUAL(tt2->threadId, 0);
}

BOOST_AUTO_TEST_CASE(TestGlobalServices02)
{
  using namespace o2::framework;
  ServiceRegistry registry;

  DummyService t0{1};
  /// We register it pretending to be on thread 0
  registry.registerService(TypeIdHelpers::uniqueId<DummyService>(), &t0, ServiceKind::Global, 1);

  auto tt0 = reinterpret_cast<DummyService*>(registry.get(TypeIdHelpers::uniqueId<DummyService>(), 0, ServiceKind::Global));
  auto tt1 = reinterpret_cast<DummyService*>(registry.get(TypeIdHelpers::uniqueId<DummyService>(), 1, ServiceKind::Global));
  auto tt2 = reinterpret_cast<DummyService*>(registry.get(TypeIdHelpers::uniqueId<DummyService>(), 2, ServiceKind::Global));
  BOOST_CHECK_EQUAL(tt0->threadId, 1);
  BOOST_CHECK_EQUAL(tt1->threadId, 1);
  BOOST_CHECK_EQUAL(tt2->threadId, 1);
}

BOOST_AUTO_TEST_CASE(TestStreamServices)
{
  using namespace o2::framework;
  ServiceRegistry registry;

  DummyService t0{0};
  DummyService t1{1};
  DummyService t2{2};
  /// We register it pretending to be on thread 0
  registry.registerService(TypeIdHelpers::uniqueId<DummyService>(), &t0, ServiceKind::Stream, 0);
  registry.registerService(TypeIdHelpers::uniqueId<DummyService>(), &t1, ServiceKind::Stream, 1);
  registry.registerService(TypeIdHelpers::uniqueId<DummyService>(), &t2, ServiceKind::Stream, 2);

  auto tt0 = reinterpret_cast<DummyService*>(registry.get(TypeIdHelpers::uniqueId<DummyService>(), 0, ServiceKind::Stream));
  auto tt1 = reinterpret_cast<DummyService*>(registry.get(TypeIdHelpers::uniqueId<DummyService>(), 1, ServiceKind::Stream));
  auto tt2 = reinterpret_cast<DummyService*>(registry.get(TypeIdHelpers::uniqueId<DummyService>(), 2, ServiceKind::Stream));
  BOOST_CHECK_EQUAL(tt0->threadId, 0);
  BOOST_CHECK_EQUAL(tt1->threadId, 1);
  BOOST_CHECK_EQUAL(tt2->threadId, 2);
}

BOOST_AUTO_TEST_CASE(TestServiceRegistryCtor)
{
  using namespace o2::framework;
  ServiceRegistry registry;
  registry = ServiceRegistry();
}

BOOST_AUTO_TEST_CASE(TestServiceDeclaration)
{
  using namespace o2::framework;
  ServiceRegistry registry;
  DeviceState state;
  fair::mq::ProgOptions options;
  options.SetProperty("monitoring-backend", "no-op://");
  options.SetProperty("infologger-mode", "no-op://");
  options.SetProperty("infologger-severity", "info");
  options.SetProperty("configuration", "command-line");

  registry.declareService(CommonServices::callbacksSpec(), state, options);
  BOOST_CHECK(registry.active<CallbackService>() == true);
  BOOST_CHECK(registry.active<DummyService>() == false);
}

BOOST_AUTO_TEST_CASE(TestServiceOverride)
{
  using namespace o2::framework;
  auto overrides = ServiceSpecHelpers::parseOverrides("foo:enable,bar:disable");
  BOOST_CHECK(overrides.size() == 2);
  BOOST_CHECK(overrides[0].name == "foo");
  BOOST_CHECK(overrides[0].active == true);
  BOOST_CHECK(overrides[1].name == "bar");
  BOOST_CHECK(overrides[1].active == false);

  auto overrides2 = ServiceSpecHelpers::parseOverrides("foo:enable");
  BOOST_CHECK(overrides2.size() == 1);
  BOOST_CHECK(overrides[0].name == "foo");
  BOOST_CHECK(overrides[0].active == true);

  BOOST_CHECK_THROW(ServiceSpecHelpers::parseOverrides("foo:enabledisabl"), std::runtime_error);
  BOOST_CHECK_THROW(ServiceSpecHelpers::parseOverrides("foo"), std::runtime_error);
  BOOST_CHECK_THROW(ServiceSpecHelpers::parseOverrides("foo:"), std::runtime_error);
  BOOST_CHECK_THROW(ServiceSpecHelpers::parseOverrides("foo:a,"), std::runtime_error);
  BOOST_CHECK_THROW(ServiceSpecHelpers::parseOverrides("foo:,"), std::runtime_error);
  BOOST_CHECK(ServiceSpecHelpers::parseOverrides("").size() == 0);
  BOOST_CHECK(ServiceSpecHelpers::parseOverrides(nullptr).size() == 0);

  auto overrides3 = ServiceSpecHelpers::parseOverrides("foo:disable,bar:enable,baz:enable");
  ServiceSpecs originalServices{
    {.name = "foo", .active = true},
    {.name = "bar", .active = false},
  };
  BOOST_CHECK(overrides3.size() == 3);
  auto services = ServiceSpecHelpers::filterDisabled(originalServices, overrides3);
  BOOST_CHECK(services.size() == 1);
  BOOST_CHECK_EQUAL(services[0].name, "bar");
  BOOST_CHECK_EQUAL(services[0].active, true);
}
