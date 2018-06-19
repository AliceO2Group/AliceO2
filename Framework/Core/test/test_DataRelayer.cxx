// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework DataRelayer
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Headers/DataHeader.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DataRelayer.h"
#include "Framework/DataProcessingHeader.h"
#include <Monitoring/Monitoring.h>
#include <fairmq/FairMQTransportFactory.h>
#include <cstring>

using Monitoring = o2::monitoring::Monitoring;
using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;


// A simple test where an input is provided
// and the subsequent InputRecord is immediately requested.
BOOST_AUTO_TEST_CASE(TestNoWait) {
  Monitoring metrics;
  InputSpec spec;
  spec.binding = "clusters";
  spec.description = "CLUSTERS";
  spec.origin = "TPC";
  spec.subSpec = 0;
  spec.lifetime = Lifetime::Timeframe;

  InputRoute route;
  route.sourceChannel = "Fake";
  route.matcher = spec;
  route.timeslice = 0;

  std::vector<InputRoute> inputs = {
    route
  };
  std::vector<ForwardRoute> forwards;

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, forwards, metrics);
  relayer.setPipelineLength(4);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;

  DataProcessingHeader dph{0,1};
  Stack stack{dh, dph};
  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");
  FairMQMessagePtr header = transport->CreateMessage(stack.size());
  FairMQMessagePtr payload = transport->CreateMessage(1000);
  memcpy(header->GetData(), stack.data(), stack.size());
  relayer.relay(std::move(header),std::move(payload));
  auto ready = relayer.getReadyToProcess();
  BOOST_REQUIRE_EQUAL(ready.size(), 1);
  BOOST_CHECK_EQUAL(ready[0].cacheLineIdx, 0);
  BOOST_CHECK_EQUAL(ready[0].op, CompletionPolicy::CompletionOp::Consume);
  auto result = relayer.getInputsForTimeslice(ready[0].cacheLineIdx);
  // One for the header, one for the payload
  BOOST_REQUIRE_EQUAL(result.size(),2);
}

// This test a more complicated set of inputs, and verifies that data is
// correctly relayed before being processed.
BOOST_AUTO_TEST_CASE(TestRelay) {
  Monitoring metrics;
  InputSpec spec1;
  spec1.binding = "clusters";
  spec1.description = "CLUSTERS";
  spec1.origin = "TPC";
  spec1.subSpec = 0;
  spec1.lifetime = Lifetime::Timeframe;

  InputSpec spec2;
  spec2.binding = "clusters_its";
  spec2.description = "CLUSTERS";
  spec2.origin = "ITS";
  spec2.subSpec = 0;
  spec2.lifetime = Lifetime::Timeframe;

  InputRoute route1;
  route1.sourceChannel = "Fake";
  route1.matcher = spec1;
  route1.timeslice = 0;

  InputRoute route2;
  route2.sourceChannel = "Fake";
  route2.matcher = spec2;
  route2.timeslice = 0;

  std::vector<InputRoute> inputs = { route1, route2 };
  std::vector<ForwardRoute> forwards;

  auto policy = CompletionPolicyHelpers::consumeWhenAll();
  DataRelayer relayer(policy, inputs, forwards, metrics);
  relayer.setPipelineLength(4);

  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");

  auto createMessage = [&transport,&relayer] (DataHeader &dh) {
    DataProcessingHeader dph{0,1};
    Stack stack{dh, dph};
    FairMQMessagePtr header = transport->CreateMessage(stack.size());
    FairMQMessagePtr payload = transport->CreateMessage(1000);
    memcpy(header->GetData(), stack.data(), stack.size());
    relayer.relay(std::move(header),std::move(payload));
  };

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;

  // Let's create the second O2 Message:
  DataHeader dh2;
  dh2.dataDescription = "CLUSTERS";
  dh2.dataOrigin = "ITS";
  dh2.subSpecification = 0;

  createMessage(dh1);
  auto ready = relayer.getReadyToProcess();
  BOOST_REQUIRE_EQUAL(ready.size(), 0);

  createMessage(dh2);
  ready = relayer.getReadyToProcess();
  BOOST_REQUIRE_EQUAL(ready.size(), 1);
  BOOST_CHECK_EQUAL(ready[0].cacheLineIdx, 0);
  BOOST_CHECK_EQUAL(ready[0].op, CompletionPolicy::CompletionOp::Consume);

  auto result = relayer.getInputsForTimeslice(ready[0].cacheLineIdx);
  // One for the header, one for the payload, for two inputs.
  BOOST_REQUIRE_EQUAL(result.size(),4);
}

// This tests a simple cache pruning, where a single input is shifted out of
// the cache.
BOOST_AUTO_TEST_CASE(TestCache) {
  Monitoring metrics;
  InputSpec spec;
  spec.binding = "clusters";
  spec.description = "CLUSTERS";
  spec.origin = "TPC";
  spec.subSpec = 0;
  spec.lifetime = Lifetime::Timeframe;

  InputRoute route;
  route.sourceChannel = "Fake";
  route.matcher = spec;
  route.timeslice = 0;

  std::vector<InputRoute> inputs = {
    route
  };
  std::vector<ForwardRoute> forwards;

  auto policy = CompletionPolicyHelpers::consumeWhenAll();
  DataRelayer relayer(policy, inputs, forwards, metrics);
  // Only two messages to fill the cache.
  relayer.setPipelineLength(2);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;

  DataProcessingHeader dph{0,1};
  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");
  auto createMessage = [&transport, &relayer, &dh](const DataProcessingHeader &h)
  {
    Stack stack{dh, h};
    FairMQMessagePtr header = transport->CreateMessage(stack.size());
    FairMQMessagePtr payload = transport->CreateMessage(1000);
    memcpy(header->GetData(), stack.data(), stack.size());
    relayer.relay(std::move(header),std::move(payload));
    assert(header.get() == nullptr);
    assert(payload.get() == nullptr);
  };

  // This fills the cache, and then empties it.
  createMessage(DataProcessingHeader{0,1});
  createMessage(DataProcessingHeader{1,1});
  auto ready = relayer.getReadyToProcess();
  BOOST_REQUIRE_EQUAL(ready.size(), 2);
  BOOST_CHECK_EQUAL(ready[0].cacheLineIdx, 0);
  BOOST_CHECK_EQUAL(ready[1].cacheLineIdx, 1);
  BOOST_CHECK_EQUAL(ready[0].op, CompletionPolicy::CompletionOp::Consume);
  BOOST_CHECK_EQUAL(ready[1].op, CompletionPolicy::CompletionOp::Consume);
  for (size_t i = 0; i < ready.size(); ++i) {
    auto result = relayer.getInputsForTimeslice(ready[i].cacheLineIdx);
  }

  // This fills the cache and makes 2 obsolete.
  createMessage(DataProcessingHeader{2,1});
  createMessage(DataProcessingHeader{3,1});
  createMessage(DataProcessingHeader{4,1});
  ready = relayer.getReadyToProcess();
  BOOST_REQUIRE_EQUAL(ready.size(), 2);

  auto result1 = relayer.getInputsForTimeslice(ready[0].cacheLineIdx);
  auto result2 = relayer.getInputsForTimeslice(ready[1].cacheLineIdx);
  // One for the header, one for the payload
  BOOST_REQUIRE_EQUAL(result1.size(),2);
  BOOST_REQUIRE_EQUAL(result2.size(),2);
}

// This the any policy. Even when there are two inputs, given the any policy
// it will run immediately.
BOOST_AUTO_TEST_CASE(TestPolicies) {
  Monitoring metrics;
  InputSpec spec1;
  spec1.binding = "clusters";
  spec1.description = "CLUSTERS";
  spec1.origin = "TPC";
  spec1.subSpec = 0;
  spec1.lifetime = Lifetime::Timeframe;

  InputSpec spec2;
  spec2.binding = "tracks";
  spec2.description = "TRACKS";
  spec2.origin = "TPC";
  spec2.subSpec = 0;
  spec2.lifetime = Lifetime::Timeframe;

  InputRoute route1;
  route1.sourceChannel = "Fake";
  route1.matcher = spec1;
  route1.timeslice = 0;

  InputRoute route2;
  route2.sourceChannel = "Fake2";
  route2.matcher = spec2;
  route2.timeslice = 0;

  std::vector<InputRoute> inputs = {
    route1,
    route2
  };
  std::vector<ForwardRoute> forwards;

  auto policy = CompletionPolicyHelpers::processWhenAny();
  DataRelayer relayer(policy, inputs, forwards, metrics);
  // Only two messages to fill the cache.
  relayer.setPipelineLength(2);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;

  DataHeader dh2;
  dh2.dataDescription = "TRACKS";
  dh2.dataOrigin = "TPC";
  dh2.subSpecification = 0;

  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");
  auto createMessage = [&transport, &relayer](DataHeader const&dh, DataProcessingHeader const& h)
  {
    Stack stack{dh, h};
    FairMQMessagePtr header = transport->CreateMessage(stack.size());
    FairMQMessagePtr payload = transport->CreateMessage(1000);
    memcpy(header->GetData(), stack.data(), stack.size());
    return relayer.relay(std::move(header),std::move(payload));
  };

  // This fills the cache, and then empties it.
  auto actions1 = createMessage(dh1, DataProcessingHeader{0,1});
  auto ready1 = relayer.getReadyToProcess();
  BOOST_REQUIRE_EQUAL(ready1.size(), 1);
  BOOST_CHECK_EQUAL(ready1[0].cacheLineIdx, 0);
  BOOST_CHECK_EQUAL(ready1[0].op, CompletionPolicy::CompletionOp::Process);

  auto actions2 = createMessage(dh1, DataProcessingHeader{1,1});
  auto ready2 = relayer.getReadyToProcess();
  BOOST_REQUIRE_EQUAL(ready2.size(), 1);
  BOOST_CHECK_EQUAL(ready2[0].cacheLineIdx, 1);
  BOOST_CHECK_EQUAL(ready2[0].op, CompletionPolicy::CompletionOp::Process);

  auto actions3 = createMessage(dh2, DataProcessingHeader{1,1});
  auto ready3 = relayer.getReadyToProcess();
  BOOST_REQUIRE_EQUAL(ready3.size(), 1);
  BOOST_CHECK_EQUAL(ready3[0].cacheLineIdx, 1);
  BOOST_CHECK_EQUAL(ready3[0].op, CompletionPolicy::CompletionOp::Consume);
}
