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
#include <boost/test/tools/old/interface.hpp>

#include "Framework/ServiceHandle.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/CallbackService.h"
#include "Framework/CommonServices.h"
#include <Framework/DeviceState.h>
#include <catch_amalgamated.hpp>
#include <fairmq/ProgOptions.h>
#include <memory>

TEST_CASE("TestServiceRegistry")
{
  o2::framework::clean_all_runtime_errors();
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
  REQUIRE(registry.get<InterfaceA>(ServiceRegistry::globalDeviceSalt()).method() == true);
  REQUIRE(registry.get<InterfaceB>(ServiceRegistry::globalDeviceSalt()).method() == false);
  REQUIRE(registry.get<InterfaceC const>(ServiceRegistry::globalDeviceSalt()).method() == false);
  REQUIRE(registry.active<InterfaceA>(ServiceRegistry::globalDeviceSalt()) == true);
  REQUIRE(registry.active<InterfaceB>(ServiceRegistry::globalDeviceSalt()) == true);
  REQUIRE(registry.active<InterfaceC>(ServiceRegistry::globalDeviceSalt()) == false);
  REQUIRE_THROWS_AS(registry.get<InterfaceA const>(ServiceRegistry::globalDeviceSalt()), RuntimeErrorRef);
  REQUIRE_THROWS_AS(registry.get<InterfaceC>(ServiceRegistry::globalDeviceSalt()), RuntimeErrorRef);
}

TEST_CASE("TestCallbackService")
{
  using namespace o2::framework;
  ServiceRegistry registry;
  ServiceRegistryRef ref{registry};
  auto service = std::make_unique<CallbackService>();
  ref.registerService(ServiceRegistryHelpers::handleForService<CallbackService>(service.get()));

  // the callback simply sets the captured variable to indicated that it was called
  bool cbCalled = false;
  auto cb = [&]() { cbCalled = true; };
  registry.get<CallbackService>(ServiceRegistry::globalDeviceSalt()).set<CallbackService::Id::Stop>(cb);

  // execute and check
  registry.get<CallbackService>(ServiceRegistry::globalDeviceSalt()).call<CallbackService::Id::Stop>();
  REQUIRE(cbCalled);
}

struct DummyService {
  int threadId;
};

namespace o2::framework
{
static ServiceRegistry::Salt salt_0 = ServiceRegistry::Salt{0, 0};
static ServiceRegistry::Salt salt_1 = ServiceRegistry::Salt{1, 0};
static ServiceRegistry::Salt salt_2 = ServiceRegistry::Salt{2, 0};
static ServiceRegistry::Salt salt_3 = ServiceRegistry::Salt{3, 0};
static ServiceRegistry::Salt salt_1_1 = ServiceRegistry::Salt{1, 1};
}

TEST_CASE("TestSerialServices")
{
  using namespace o2::framework;
  ServiceRegistry registry;

  DummyService t0{0};
  /// We register it pretending to be on thread 0
  registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, &t0, ServiceKind::Serial, salt_0);

  auto tt0 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_0, ServiceKind::Serial));
  auto tt1 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_1, ServiceKind::Serial));
  auto tt2 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_2, ServiceKind::Serial));
  REQUIRE(tt0->threadId == 0);
  REQUIRE(tt1->threadId == 0);
  REQUIRE(tt2->threadId == 0);
}

TEST_CASE("TestGlobalServices")
{
  using namespace o2::framework;
  ServiceRegistry registry;

  DummyService t0{0};
  /// We register it pretending to be on thread 0
  registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, &t0, ServiceKind::Global, salt_0);

  auto tt0 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_0, ServiceKind::Serial));
  auto tt1 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_1, ServiceKind::Serial));
  auto tt2 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_2, ServiceKind::Serial));
  REQUIRE(tt0->threadId == 0);
  REQUIRE(tt1->threadId == 0);
  REQUIRE(tt2->threadId == 0);
}

TEST_CASE("TestGlobalServices02")
{
  using namespace o2::framework;
  ServiceRegistry registry;

  DeviceState state;
  fair::mq::ProgOptions options;
  ServiceSpec spec{.name = "dummy-service",
                   .uniqueId = CommonServices::simpleServiceId<DummyService>(),
                   .init = [](ServiceRegistryRef, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
                     // this is needed to check we register it only once
                     static int i = 1;
                     return ServiceHandle{TypeIdHelpers::uniqueId<DummyService>(), new DummyService{i++}};
                   },
                   .configure = CommonServices::noConfiguration(),
                   .kind = ServiceKind::Global};

  // If the service was not declared, we should not be able to register it from a stream context.
  REQUIRE_THROWS_AS(registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, nullptr, ServiceKind::Global, salt_1), RuntimeErrorRef);
  // Declare the service
  registry.declareService(spec, state, options, ServiceRegistry::globalDeviceSalt());

  /// We register it pretending to be on thread 0
  try {
    registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, nullptr, ServiceKind::Global, salt_1);
  } catch (RuntimeErrorRef e) {
    INFO(error_from_ref(e).what);
    REQUIRE(false);
  }

  auto tt0 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_0, ServiceKind::Global));
  auto tt1 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_1, ServiceKind::Global));
  auto tt2 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_2, ServiceKind::Global));
  REQUIRE(tt0->threadId == 1);
  REQUIRE(tt1->threadId == 1);
  REQUIRE(tt2->threadId == 1);
}

TEST_CASE("TestStreamServices")
{
  using namespace o2::framework;
  ServiceRegistry registry;

  DummyService t1{1};
  DummyService t2{2};
  DummyService t3{3};
  DummyService t2_d1{2};

  ServiceSpec spec{.name = "dummy-service",
                   .uniqueId = CommonServices::simpleServiceId<DummyService>(),
                   .init = CommonServices::simpleServiceInit<DummyService, DummyService>(),
                   .configure = CommonServices::noConfiguration(),
                   .kind = ServiceKind::Stream};

  DeviceState state;
  fair::mq::ProgOptions options;
  // This will raise an exception because we have not declared the service yet.
  REQUIRE_THROWS_AS(registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, nullptr, ServiceKind::Stream, ServiceRegistry::Salt{1, 0}), RuntimeErrorRef);
  registry.declareService(spec, state, options, ServiceRegistry::globalDeviceSalt());

  /// We register it pretending to be on thread 0
  registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, &t1, ServiceKind::Stream, ServiceRegistry::Salt{1, 0}, "dummy-service1", ServiceRegistry::SpecIndex{0});
  registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, &t2, ServiceKind::Stream, ServiceRegistry::Salt{2, 0}, "dummy-service2", ServiceRegistry::SpecIndex{0});
  registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, &t3, ServiceKind::Stream, ServiceRegistry::Salt{3, 0}, "dummy-service3", ServiceRegistry::SpecIndex{0});

  auto tt1 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_1, ServiceKind::Stream));
  auto tt2 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_2, ServiceKind::Stream));
  auto tt3 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_3, ServiceKind::Stream));
  REQUIRE(tt1->threadId == 1);
  REQUIRE(tt2->threadId == 2);
  REQUIRE(tt3->threadId == 3);
  // Check that Context{1,1} throws, because we registerd it for a different data processor.
  REQUIRE_THROWS_AS(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_1_1, ServiceKind::Stream), RuntimeErrorRef);

  // Check that Context{0,0} throws.
  REQUIRE_THROWS_AS(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_0, ServiceKind::Stream), RuntimeErrorRef);
  registry.registerService({TypeIdHelpers::uniqueId<DummyService>()}, &t2_d1, ServiceKind::Stream, ServiceRegistry::Salt{3, 1}, "dummy-service", ServiceRegistry::SpecIndex{0});

  auto tt2_dp1 = reinterpret_cast<DummyService*>(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, ServiceRegistry::Salt{3, 1}, ServiceKind::Stream));
  REQUIRE(tt2_dp1->threadId == 2);

  REQUIRE_THROWS_AS(registry.get({TypeIdHelpers::uniqueId<DummyService>()}, salt_1_1, ServiceKind::Stream), RuntimeErrorRef);
}

TEST_CASE("TestServiceRegistryCtor")
{
  using namespace o2::framework;
  ServiceRegistry registry;
  registry = ServiceRegistry();
}

TEST_CASE("TestServiceDeclaration")
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
  REQUIRE(registry.active<CallbackService>(ServiceRegistry::globalDeviceSalt()) == true);
  REQUIRE(registry.active<DummyService>(ServiceRegistry::globalDeviceSalt()) == false);
}

TEST_CASE("TestServiceOverride")
{
  using namespace o2::framework;
  auto overrides = ServiceSpecHelpers::parseOverrides("foo:enable,bar:disable");
  REQUIRE(overrides.size() == 2);
  REQUIRE(overrides[0].name == "foo");
  REQUIRE(overrides[0].active == true);
  REQUIRE(overrides[1].name == "bar");
  REQUIRE(overrides[1].active == false);

  auto overrides2 = ServiceSpecHelpers::parseOverrides("foo:enable");
  REQUIRE(overrides2.size() == 1);
  REQUIRE(overrides[0].name == "foo");
  REQUIRE(overrides[0].active == true);

  REQUIRE_THROWS_AS(ServiceSpecHelpers::parseOverrides("foo:enabledisabl"), std::runtime_error);
  REQUIRE_THROWS_AS(ServiceSpecHelpers::parseOverrides("foo"), std::runtime_error);
  REQUIRE_THROWS_AS(ServiceSpecHelpers::parseOverrides("foo:"), std::runtime_error);
  REQUIRE_THROWS_AS(ServiceSpecHelpers::parseOverrides("foo:a,"), std::runtime_error);
  REQUIRE_THROWS_AS(ServiceSpecHelpers::parseOverrides("foo:,"), std::runtime_error);
  REQUIRE(ServiceSpecHelpers::parseOverrides("").size() == 0);
  REQUIRE(ServiceSpecHelpers::parseOverrides(nullptr).size() == 0);

  auto overrides3 = ServiceSpecHelpers::parseOverrides("foo:disable,bar:enable,baz:enable");
  ServiceSpecs originalServices{
    {.name = "foo", .active = true},
    {.name = "bar", .active = false},
  };
  REQUIRE(overrides3.size() == 3);
  auto services = ServiceSpecHelpers::filterDisabled(originalServices, overrides3);
  REQUIRE(services.size() == 1);
  REQUIRE(services[0].name == "bar");
  REQUIRE(services[0].active == true);
}
