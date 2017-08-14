// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework Services 
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/ServiceRegistry.h"
#include <boost/test/unit_test.hpp>
#include <iostream>


BOOST_AUTO_TEST_CASE(TestServiceRegistry) {
  using namespace o2::framework;
  struct InterfaceA {
    virtual bool method() = 0;
  };

  struct ConcreteA : InterfaceA {
    bool method() final {return true; }
  };

  struct InterfaceB {
    virtual bool method() = 0;
  };

  struct ConcreteB : InterfaceB {
    bool method() final {return false; }
  };

  struct InterfaceC {
    virtual bool method() = 0;
  };

  ServiceRegistry registry;
  auto service = std::make_unique<ConcreteA>;
  ConcreteA serviceA;
  ConcreteB serviceB;
  registry.registerService<InterfaceA>(&serviceA);
  registry.registerService<InterfaceB>(&serviceB);
  BOOST_CHECK(registry.get<InterfaceA>().method() == true);
  BOOST_CHECK(registry.get<InterfaceB>().method() == false);
  BOOST_CHECK_THROW(registry.get<InterfaceC>().method(), std::runtime_error);
}
