// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework ExternalFairMQDeviceProxy
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/ExternalFairMQDeviceProxy.h"
#include <string>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(ExternalFairMQDeviceProxy) {
  InjectorFunction f;
  DataProcessorSpec spec = specifyExternalFairMQDeviceProxy("testSource",
                            {}, "type=sub,method=connect,address=tcp://localhost:10000,rateLogging=1", f);
  BOOST_CHECK_EQUAL(spec.name, "testSource");
  BOOST_CHECK_EQUAL(spec.inputs.size(), 0);
  BOOST_REQUIRE_EQUAL(spec.options.size(), 1);
  BOOST_CHECK_EQUAL(spec.options[0].name, "channel-config");
  BOOST_CHECK_EQUAL(spec.options[0].defaultValue.get<const char *>(), std::string("name=testSource,type=sub,method=connect,address=tcp://localhost:10000,rateLogging=1"));
}
