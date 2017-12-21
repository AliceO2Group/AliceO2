// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework InputRecord
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/InputRecord.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"
#include <fairmq/FairMQMessage.h>
#include <fairmq/FairMQTransportFactory.h>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;

bool any_exception( std::exception const& ex ) { return true; }

BOOST_AUTO_TEST_CASE(TestInputRecord) {
  // Create the routes we want for the InputRecord
  InputSpec spec1;
  spec1.binding = "x";
  spec1.description = "CLUSTERS";
  spec1.origin = "TPC";
  spec1.subSpec = 0;
  spec1.lifetime = InputSpec::Timeframe;

  InputSpec spec2;
  spec2.binding = "y";
  spec2.description = "CLUSTERS";
  spec2.origin = "ITS";
  spec2.subSpec = 0;
  spec2.lifetime = InputSpec::Timeframe;

  auto createRoute = [](const char *source, InputSpec &spec) {
    InputRoute route;
    route.sourceChannel = source;
    route.matcher = spec;
    return route;
  };

  std::vector<InputRoute> schema = {
    createRoute("x_source", spec1),
    createRoute("y_source", spec2)
  };
  // First of all we test if an empty registry behaves as expected, raising a
  // bunch of exceptions.
  InputRecord emptyRegistry(schema, {});

  BOOST_CHECK_EXCEPTION(emptyRegistry.get("x"), std::exception, any_exception);
  BOOST_CHECK_EXCEPTION(emptyRegistry.get("y"), std::exception, any_exception);
  BOOST_CHECK_EXCEPTION(emptyRegistry.getByPos(0), std::exception, any_exception);
  BOOST_CHECK_EXCEPTION(emptyRegistry.getByPos(1), std::exception, any_exception);
  // Then we actually check with a real set of inputs.

  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");
  std::vector<FairMQMessagePtr> inputs;

  auto createMessage = [&transport, &inputs] (DataHeader &dh, int value) {
    DataProcessingHeader dph{0,1};
    Stack stack{dh, dph};
    FairMQMessagePtr header = transport->CreateMessage(stack.size());
    FairMQMessagePtr payload = transport->CreateMessage(sizeof(int));
    memcpy(header->GetData(), stack.data(), stack.size());
    memcpy(payload->GetData(), &value, sizeof(int));
    inputs.emplace_back(std::move(header));
    inputs.emplace_back(std::move(payload));
  };
  DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  DataHeader dh2;
  dh2.dataDescription = "CLUSTERS";
  dh2.dataOrigin = "ITS";
  dh2.subSpecification = 0;
  createMessage(dh1, 1);
  createMessage(dh2, 2);
  InputRecord registry(schema, inputs);

  // Checking we can get the whole ref by name
  BOOST_CHECK_NO_THROW(registry.get("x"));
  BOOST_CHECK_NO_THROW(registry.get("y"));
  auto ref00 = registry.get("x");
  auto ref10 = registry.get("y");
  BOOST_CHECK_EXCEPTION(registry.get("z"), std::exception, any_exception);

  // Or we can get it positionally
  BOOST_CHECK_NO_THROW(registry.get("x"));
  auto ref01 = registry.getByPos(0);
  auto ref11= registry.getByPos(1);
  BOOST_CHECK_EXCEPTION(registry.getByPos(3), std::exception, any_exception);

  // This should be exactly the same pointers
  BOOST_CHECK_EQUAL(ref00.header, ref01.header);
  BOOST_CHECK_EQUAL(ref00.payload, ref01.payload);
  BOOST_CHECK_EQUAL(ref10.header, ref11.header);
  BOOST_CHECK_EQUAL(ref10.payload, ref11.payload);

  // This by default is a shortcut for 
  //
  // *static_cast<int const *>(registry.get("x").payload);
  //
  BOOST_CHECK_EQUAL(registry.get<int>("x"),1);
  BOOST_CHECK_EQUAL(registry.get<int>("y"),2);
  // A few more time just to make sure we are not stateful..
  BOOST_CHECK_EQUAL(registry.get<int>("x"),1);
  BOOST_CHECK_EQUAL(registry.get<int>("x"),1);
}
