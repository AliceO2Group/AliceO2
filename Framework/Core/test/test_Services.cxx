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
    [[nodiscard]] virtual bool method() const = 0;
  };

  struct ConcreteC : InterfaceC {
    [[nodiscard]] bool method() const final { return false; }
  };

  ServiceRegistry registry;
  ServiceRegistryRef ref{registry};
  ConcreteA serviceA;
  ConcreteB serviceB;
  ConcreteC const serviceC;
  ref.registerService(ServiceRegistryHelpers::handleForService<InterfaceA>(&serviceA));
  ref.registerService(ServiceRegistryHelpers::handleForService<InterfaceB>(&serviceB));
  ref.registerService(ServiceRegistryHelpers::handleForService<InterfaceC const>(&serviceC));
  BOOST_CHECK(registry.get<InterfaceA>(ServiceRegistry::globalDeviceSalt()).method() == true);
  BOOST_CHECK(registry.get<InterfaceB>(ServiceRegistry::globalDeviceSalt()).method() == false);
  BOOST_CHECK(registry.get<InterfaceC const>(ServiceRegistry::globalDeviceSalt()).method() == false);
  BOOST_CHECK(registry.active<InterfaceA>(ServiceRegistry::globalDeviceSalt()) == true);
  BOOST_CHECK(registry.active<InterfaceB>(ServiceRegistry::globalDeviceSalt()) == true);
  BOOST_CHECK(registry.active<InterfaceC>(ServiceRegistry::globalDeviceSalt()) == false);
  BOOST_CHECK_THROW(registry.get<InterfaceA const>(ServiceRegistry::globalDeviceSalt()), RuntimeErrorRef);
  BOOST_CHECK_THROW(registry.get<InterfaceC>(ServiceRegistry::globalDeviceSalt()), RuntimeErrorRef);
}

BOOST_AUTO_TEST_CASE(TestCallbackService)
{
  using namespace o2::framework;
  ServiceRegistry registry;
  ServiceRegistryRef ref{registry};
  auto service = std::make_unique<CallbackService>();
  ref.registerService(ServiceRegistryHelpers::handleForService<CallbackService>(service.get()));

  // the callback simply sets the captured variable to indicated that it was called
  bool cbCalled = false;
  auto cb = [&]() { cbCalled = true; };
  registry.get<CallbackService>(ServiceRegistry::globalDeviceSalt()).set(CallbackService::Id::Stop, cb);

  // check to set with the wrong type
  BOOST_CHECK_THROW(registry.get<CallbackService>(ServiceRegistry::globalDeviceSalt()).set(CallbackService::Id::Stop, [](int) {}), RuntimeErrorRef);

  // execute and check
  registry.get<CallbackService>(ServiceRegistry::globalDeviceSalt())(CallbackService::Id::Stop);
  BOOST_CHECK(cbCalled);
}

struct DummyService {
  int threadId;
};

namespace o2::framework
{
static ServiceRegistry::Salt salt_0 =  ServiceRegistry::Salt{ServiceRegistry::Context{0,0}}; 
static ServiceRegistry::Salt salt_1 =  ServiceRegistry::Salt{ServiceRegistry::Context{1,0}};
static ServiceRegistry::Salt salt_2 =  ServiceRegistry::Salt{ServiceRegistry::Context{2,0}};
}

BOOST_AUTO_TEST_CASE(TestSerialServices)
{
  using namespace o2::framework;
  ServiceRegistry registry;

  DummyService t0{0};
  /// We register it pretending to be on thread 0
  registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, &t0, ServiceKind::Serial, salt_0);

  auto tt0 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_0, ServiceKind::Serial));
  auto tt1 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_1, ServiceKind::Serial));
  auto tt2 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_2, ServiceKind::Serial));
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
  registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, &t0, ServiceKind::Global, salt_0);

  auto tt0 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_0, ServiceKind::Serial));
  auto tt1 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_1, ServiceKind::Serial));
  auto tt2 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_2, ServiceKind::Serial));
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
  registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, &t0, ServiceKind::Global, salt_1);

  auto tt0 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_0, ServiceKind::Global));
  auto tt1 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_1, ServiceKind::Global));
  auto tt2 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_2, ServiceKind::Global));
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
  registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, &t0, ServiceKind::Stream, ServiceRegistry::Salt{ServiceRegistry::Context{0,0}});
  registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, &t1, ServiceKind::Stream, ServiceRegistry::Salt{ServiceRegistry::Context{1,0}});
  registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, &t2, ServiceKind::Stream, ServiceRegistry::Salt{ServiceRegistry::Context{2,0}});

  auto tt0 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_0, ServiceKind::Stream));
  auto tt1 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_1, ServiceKind::Stream));
  auto tt2 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_2, ServiceKind::Stream));
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
  BOOST_CHECK(registry.active<CallbackService>(ServiceRegistry::globalDeviceSalt()) == true);
  BOOST_CHECK(registry.active<DummyService>(ServiceRegistry::globalDeviceSalt()) == false);
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
