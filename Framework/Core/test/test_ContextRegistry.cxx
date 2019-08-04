// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework ContextRegistry
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/ContextRegistry.h"
#include "Framework/ArrowContext.h"
#include "Framework/StringContext.h"
#include "Framework/RawBufferContext.h"
#include "Framework/RootObjectContext.h"
#include "Framework/MessageContext.h"
#include <TObject.h>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestContextRegistry)
{
  FairMQDeviceProxy proxy(nullptr);
  ArrowContext c0(proxy);
  StringContext c1(proxy);
  RawBufferContext c2(proxy);
  RootObjectContext c3(proxy);
  MessageContext c4(proxy);
  ContextRegistry registry({&c0, &c1, &c2, &c3, &c4});

  BOOST_REQUIRE_EQUAL(&c0, registry.get<ArrowContext>());
  BOOST_REQUIRE_EQUAL(&c1, registry.get<StringContext>());
  BOOST_REQUIRE_EQUAL(&c2, registry.get<RawBufferContext>());
  BOOST_REQUIRE_EQUAL(&c3, registry.get<RootObjectContext>());
  BOOST_REQUIRE_EQUAL(&c4, registry.get<MessageContext>());
}
