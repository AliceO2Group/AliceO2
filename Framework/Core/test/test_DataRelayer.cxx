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
#include "Headers/Stack.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DataRelayer.h"
#include "../src/DataRelayerHelpers.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/WorkflowSpec.h"
#include <Monitoring/Monitoring.h>
#include <fairmq/FairMQTransportFactory.h>
#include <cstring>

using Monitoring = o2::monitoring::Monitoring;
using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;
using RecordAction = o2::framework::DataRelayer::RecordAction;

// A simple test where an input is provided
// and the subsequent InputRecord is immediately requested.
BOOST_AUTO_TEST_CASE(TestNoWait)
{
  Monitoring metrics;
  InputSpec spec{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec, 0, "Fake", 0}};

  std::vector<ForwardRoute> forwards;
  TimesliceIndex index;

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, metrics, index);
  relayer.setPipelineLength(4);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;
  dh.splitPayloadIndex = 0;
  dh.splitPayloadParts = 1;

  DataProcessingHeader dph{0, 1};
  Stack stack{dh, dph};
  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");
  FairMQMessagePtr header = transport->CreateMessage(stack.size());
  FairMQMessagePtr payload = transport->CreateMessage(1000);
  memcpy(header->GetData(), stack.data(), stack.size());
  relayer.relay(header, payload);
  std::vector<RecordAction> ready;
  relayer.getReadyToProcess(ready);
  BOOST_REQUIRE_EQUAL(ready.size(), 1);
  BOOST_CHECK_EQUAL(ready[0].slot.index, 0);
  BOOST_CHECK_EQUAL(ready[0].op, CompletionPolicy::CompletionOp::Consume);
  BOOST_CHECK_EQUAL(header.get(), nullptr);
  BOOST_CHECK_EQUAL(payload.get(), nullptr);
  auto result = relayer.getInputsForTimeslice(ready[0].slot);
  // one MessageSet with one PartRef with header and payload
  BOOST_REQUIRE_EQUAL(result.size(), 1);
  BOOST_REQUIRE_EQUAL(result.at(0).size(), 1);
}

//
BOOST_AUTO_TEST_CASE(TestNoWaitMatcher)
{
  Monitoring metrics;
  auto specs = o2::framework::select("clusters:TPC/CLUSTERS");

  std::vector<InputRoute> inputs = {
    InputRoute{specs[0], 0, "Fake", 0}};

  std::vector<ForwardRoute> forwards;
  TimesliceIndex index;

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, metrics, index);
  relayer.setPipelineLength(4);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;
  dh.splitPayloadIndex = 0;
  dh.splitPayloadParts = 1;

  DataProcessingHeader dph{0, 1};
  Stack stack{dh, dph};
  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");
  FairMQMessagePtr header = transport->CreateMessage(stack.size());
  FairMQMessagePtr payload = transport->CreateMessage(1000);
  memcpy(header->GetData(), stack.data(), stack.size());
  relayer.relay(header, payload);
  std::vector<RecordAction> ready;
  relayer.getReadyToProcess(ready);
  BOOST_REQUIRE_EQUAL(ready.size(), 1);
  BOOST_CHECK_EQUAL(ready[0].slot.index, 0);
  BOOST_CHECK_EQUAL(ready[0].op, CompletionPolicy::CompletionOp::Consume);
  BOOST_CHECK_EQUAL(header.get(), nullptr);
  BOOST_CHECK_EQUAL(payload.get(), nullptr);
  auto result = relayer.getInputsForTimeslice(ready[0].slot);
  // one MessageSet with one PartRef with header and payload
  BOOST_REQUIRE_EQUAL(result.size(), 1);
  BOOST_REQUIRE_EQUAL(result.at(0).size(), 1);
}

// This test a more complicated set of inputs, and verifies that data is
// correctly relayed before being processed.
BOOST_AUTO_TEST_CASE(TestRelay)
{
  Monitoring metrics;
  InputSpec spec1{
    "clusters",
    "TPC",
    "CLUSTERS",
  };
  InputSpec spec2{
    "clusters_its",
    "ITS",
    "CLUSTERS",
  };

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
    InputRoute{spec2, 1, "Fake2", 0}};

  std::vector<ForwardRoute> forwards;

  TimesliceIndex index;

  auto policy = CompletionPolicyHelpers::consumeWhenAll();
  DataRelayer relayer(policy, inputs, metrics, index);
  relayer.setPipelineLength(4);

  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");

  auto createMessage = [&transport, &relayer](DataHeader& dh, size_t time) {
    DataProcessingHeader dph{time, 1};
    Stack stack{dh, dph};
    FairMQMessagePtr header = transport->CreateMessage(stack.size());
    FairMQMessagePtr payload = transport->CreateMessage(1000);
    memcpy(header->GetData(), stack.data(), stack.size());
    relayer.relay(header, payload);
    BOOST_CHECK_EQUAL(header.get(), nullptr);
    BOOST_CHECK_EQUAL(payload.get(), nullptr);
  };

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.splitPayloadIndex = 0;
  dh1.splitPayloadParts = 1;

  // Let's create the second O2 Message:
  DataHeader dh2;
  dh2.dataDescription = "CLUSTERS";
  dh2.dataOrigin = "ITS";
  dh2.subSpecification = 0;
  dh2.splitPayloadIndex = 0;
  dh2.splitPayloadParts = 1;

  createMessage(dh1, 0);
  std::vector<RecordAction> ready;
  relayer.getReadyToProcess(ready);
  BOOST_REQUIRE_EQUAL(ready.size(), 0);

  createMessage(dh2, 0);
  ready.clear();
  relayer.getReadyToProcess(ready);
  BOOST_REQUIRE_EQUAL(ready.size(), 1);
  BOOST_CHECK_EQUAL(ready[0].slot.index, 0);
  BOOST_CHECK_EQUAL(ready[0].op, CompletionPolicy::CompletionOp::Consume);

  auto result = relayer.getInputsForTimeslice(ready[0].slot);
  // two MessageSets, each with one PartRef
  BOOST_REQUIRE_EQUAL(result.size(), 2);
  BOOST_REQUIRE_EQUAL(result.at(0).size(), 1);
  BOOST_REQUIRE_EQUAL(result.at(1).size(), 1);
}

// This test a more complicated set of inputs, and verifies that data is
// correctly relayed before being processed.
BOOST_AUTO_TEST_CASE(TestRelayBug)
{
  Monitoring metrics;
  InputSpec spec1{
    "clusters",
    "TPC",
    "CLUSTERS",
  };
  InputSpec spec2{
    "clusters_its",
    "ITS",
    "CLUSTERS",
  };

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
    InputRoute{spec2, 1, "Fake2", 0}};

  std::vector<ForwardRoute> forwards;

  TimesliceIndex index;

  auto policy = CompletionPolicyHelpers::consumeWhenAll();
  DataRelayer relayer(policy, inputs, metrics, index);
  relayer.setPipelineLength(3);

  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");

  auto createMessage = [&transport, &relayer](DataHeader& dh, size_t time) {
    DataProcessingHeader dph{time, 1};
    Stack stack{dh, dph};
    FairMQMessagePtr header = transport->CreateMessage(stack.size());
    FairMQMessagePtr payload = transport->CreateMessage(1000);
    memcpy(header->GetData(), stack.data(), stack.size());
    relayer.relay(header, payload);
    BOOST_CHECK_EQUAL(header.get(), nullptr);
    BOOST_CHECK_EQUAL(payload.get(), nullptr);
  };

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.splitPayloadIndex = 0;
  dh1.splitPayloadParts = 1;

  // Let's create the second O2 Message:
  DataHeader dh2;
  dh2.dataDescription = "CLUSTERS";
  dh2.dataOrigin = "ITS";
  dh2.subSpecification = 0;
  dh2.splitPayloadIndex = 0;
  dh2.splitPayloadParts = 1;

  // Let's create the second O2 Message:
  DataHeader dh3;
  dh3.dataDescription = "CLUSTERS";
  dh3.dataOrigin = "FOO";
  dh3.subSpecification = 0;
  dh3.splitPayloadIndex = 0;
  dh3.splitPayloadParts = 1;

  /// Reproduce the bug reported by Matthias in https://github.com/AliceO2Group/AliceO2/pull/1483
  createMessage(dh1, 0);
  std::vector<RecordAction> ready;
  relayer.getReadyToProcess(ready);
  BOOST_REQUIRE_EQUAL(ready.size(), 0);
  createMessage(dh1, 1);
  ready.clear();
  relayer.getReadyToProcess(ready);
  BOOST_REQUIRE_EQUAL(ready.size(), 0);
  createMessage(dh2, 0);
  ready.clear();
  relayer.getReadyToProcess(ready);
  BOOST_REQUIRE_EQUAL(ready.size(), 1);
  BOOST_CHECK_EQUAL(ready[0].slot.index, 0);
  BOOST_CHECK_EQUAL(ready[0].op, CompletionPolicy::CompletionOp::Consume);
  auto result = relayer.getInputsForTimeslice(ready[0].slot);
  createMessage(dh2, 1);
  ready.clear();
  relayer.getReadyToProcess(ready);
  BOOST_REQUIRE_EQUAL(ready.size(), 1);
  BOOST_CHECK_EQUAL(ready[0].slot.index, 1);
  BOOST_CHECK_EQUAL(ready[0].op, CompletionPolicy::CompletionOp::Consume);
  result = relayer.getInputsForTimeslice(ready[0].slot);
}

// This tests a simple cache pruning, where a single input is shifted out of
// the cache.
BOOST_AUTO_TEST_CASE(TestCache)
{
  Monitoring metrics;
  InputSpec spec{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec, 0, "Fake", 0}};
  std::vector<ForwardRoute> forwards;

  auto policy = CompletionPolicyHelpers::consumeWhenAll();
  TimesliceIndex index;
  DataRelayer relayer(policy, inputs, metrics, index);
  // Only two messages to fill the cache.
  relayer.setPipelineLength(2);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;
  dh.splitPayloadIndex = 0;
  dh.splitPayloadParts = 1;

  DataProcessingHeader dph{0, 1};
  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");
  auto createMessage = [&transport, &relayer, &dh](const DataProcessingHeader& h) {
    Stack stack{dh, h};
    FairMQMessagePtr header = transport->CreateMessage(stack.size());
    FairMQMessagePtr payload = transport->CreateMessage(1000);
    memcpy(header->GetData(), stack.data(), stack.size());
    auto res = relayer.relay(header, payload);
    BOOST_REQUIRE(res != DataRelayer::RelayChoice::WillRelay || header.get() == nullptr);
    BOOST_REQUIRE(res != DataRelayer::RelayChoice::WillRelay || payload.get() == nullptr);
    BOOST_REQUIRE(res != DataRelayer::RelayChoice::Backpressured || header.get() != nullptr);
    BOOST_REQUIRE(res != DataRelayer::RelayChoice::Backpressured || payload.get() != nullptr);
  };

  // This fills the cache, and then empties it.
  createMessage(DataProcessingHeader{0, 1});
  createMessage(DataProcessingHeader{1, 1});
  std::vector<RecordAction> ready;
  relayer.getReadyToProcess(ready);
  BOOST_REQUIRE_EQUAL(ready.size(), 2);
  BOOST_CHECK_EQUAL(ready[0].slot.index, 0);
  BOOST_CHECK_EQUAL(ready[1].slot.index, 1);
  BOOST_CHECK_EQUAL(ready[0].op, CompletionPolicy::CompletionOp::Consume);
  BOOST_CHECK_EQUAL(ready[1].op, CompletionPolicy::CompletionOp::Consume);
  for (size_t i = 0; i < ready.size(); ++i) {
    auto result = relayer.getInputsForTimeslice(ready[i].slot);
  }

  // This fills the cache and makes 2 obsolete.
  createMessage(DataProcessingHeader{2, 1});
  createMessage(DataProcessingHeader{3, 1});
  createMessage(DataProcessingHeader{4, 1});
  ready.clear();
  relayer.getReadyToProcess(ready);
  BOOST_REQUIRE_EQUAL(ready.size(), 2);

  auto result1 = relayer.getInputsForTimeslice(ready[0].slot);
  auto result2 = relayer.getInputsForTimeslice(ready[1].slot);
  // One for the header, one for the payload
  BOOST_REQUIRE_EQUAL(result1.size(), 1);
  BOOST_REQUIRE_EQUAL(result2.size(), 1);
}

// This the any policy. Even when there are two inputs, given the any policy
// it will run immediately.
BOOST_AUTO_TEST_CASE(TestPolicies)
{
  Monitoring metrics;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};
  InputSpec spec2{"tracks", "TPC", "TRACKS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
    InputRoute{spec2, 1, "Fake2", 0},
  };

  std::vector<ForwardRoute> forwards;
  TimesliceIndex index;

  auto policy = CompletionPolicyHelpers::processWhenAny();
  DataRelayer relayer(policy, inputs, metrics, index);
  // Only two messages to fill the cache.
  relayer.setPipelineLength(2);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.splitPayloadIndex = 0;
  dh1.splitPayloadParts = 1;

  DataHeader dh2;
  dh2.dataDescription = "TRACKS";
  dh2.dataOrigin = "TPC";
  dh2.subSpecification = 0;
  dh2.splitPayloadIndex = 0;
  dh2.splitPayloadParts = 1;

  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");
  auto createMessage = [&transport, &relayer](DataHeader const& dh, DataProcessingHeader const& h) {
    Stack stack{dh, h};
    FairMQMessagePtr header = transport->CreateMessage(stack.size());
    FairMQMessagePtr payload = transport->CreateMessage(1000);
    memcpy(header->GetData(), stack.data(), stack.size());
    return relayer.relay(header, payload);
  };

  // This fills the cache, and then empties it.
  auto actions1 = createMessage(dh1, DataProcessingHeader{0, 1});
  std::vector<RecordAction> ready1;
  relayer.getReadyToProcess(ready1);
  BOOST_REQUIRE_EQUAL(ready1.size(), 1);
  BOOST_CHECK_EQUAL(ready1[0].slot.index, 0);
  BOOST_CHECK_EQUAL(ready1[0].op, CompletionPolicy::CompletionOp::Process);

  auto actions2 = createMessage(dh1, DataProcessingHeader{1, 1});
  std::vector<RecordAction> ready2;
  relayer.getReadyToProcess(ready2);
  BOOST_REQUIRE_EQUAL(ready2.size(), 1);
  BOOST_CHECK_EQUAL(ready2[0].slot.index, 1);
  BOOST_CHECK_EQUAL(ready2[0].op, CompletionPolicy::CompletionOp::Process);

  auto actions3 = createMessage(dh2, DataProcessingHeader{1, 1});
  std::vector<RecordAction> ready3;
  relayer.getReadyToProcess(ready3);
  BOOST_REQUIRE_EQUAL(ready3.size(), 1);
  BOOST_CHECK_EQUAL(ready3[0].slot.index, 1);
  BOOST_CHECK_EQUAL(ready3[0].op, CompletionPolicy::CompletionOp::Consume);
}

/// Test that the clear method actually works.
BOOST_AUTO_TEST_CASE(TestClear)
{
  Monitoring metrics;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};
  InputSpec spec2{"tracks", "TPC", "TRACKS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
    InputRoute{spec2, 1, "Fake2", 0},
  };

  std::vector<ForwardRoute> forwards;
  TimesliceIndex index;

  auto policy = CompletionPolicyHelpers::processWhenAny();
  DataRelayer relayer(policy, inputs, metrics, index);
  // Only two messages to fill the cache.
  relayer.setPipelineLength(3);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.splitPayloadIndex = 0;
  dh1.splitPayloadParts = 1;

  DataHeader dh2;
  dh2.dataDescription = "TRACKS";
  dh2.dataOrigin = "TPC";
  dh2.subSpecification = 0;
  dh2.splitPayloadIndex = 0;
  dh2.splitPayloadParts = 1;

  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");
  auto createMessage = [&transport, &relayer](DataHeader const& dh, DataProcessingHeader const& h) {
    Stack stack{dh, h};
    FairMQMessagePtr header = transport->CreateMessage(stack.size());
    FairMQMessagePtr payload = transport->CreateMessage(1000);
    memcpy(header->GetData(), stack.data(), stack.size());
    return relayer.relay(header, payload);
  };

  // This fills the cache, and then empties it.
  auto actions1 = createMessage(dh1, DataProcessingHeader{0, 1});
  auto actions2 = createMessage(dh1, DataProcessingHeader{1, 1});
  auto actions3 = createMessage(dh2, DataProcessingHeader{1, 1});
  relayer.clear();
  std::vector<RecordAction> ready;
  relayer.getReadyToProcess(ready);
  BOOST_REQUIRE_EQUAL(ready.size(), 0);
}

/// Test that the clear method actually works.
BOOST_AUTO_TEST_CASE(TestTooMany)
{
  Monitoring metrics;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};
  InputSpec spec2{"tracks", "TPC", "TRACKS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
    InputRoute{spec2, 1, "Fake2", 0},
  };

  std::vector<ForwardRoute> forwards;
  TimesliceIndex index;

  auto policy = CompletionPolicyHelpers::processWhenAny();
  DataRelayer relayer(policy, inputs, metrics, index);
  // Only two messages to fill the cache.
  relayer.setPipelineLength(1);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.splitPayloadIndex = 0;
  dh1.splitPayloadParts = 1;

  DataHeader dh2;
  dh2.dataDescription = "TRACKS";
  dh2.dataOrigin = "TPC";
  dh2.subSpecification = 0;
  dh2.splitPayloadIndex = 0;
  dh2.splitPayloadParts = 1;

  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");

  Stack s1{dh1, DataProcessingHeader{0, 1}};
  FairMQMessagePtr header = transport->CreateMessage(s1.size());
  FairMQMessagePtr payload = transport->CreateMessage(1000);
  memcpy(header->GetData(), s1.data(), s1.size());
  relayer.relay(header, payload);
  BOOST_CHECK_EQUAL(header.get(), nullptr);
  BOOST_CHECK_EQUAL(payload.get(), nullptr);
  // This fills the cache, and then waits.
  Stack s2{dh1, DataProcessingHeader{1, 1}};
  FairMQMessagePtr header2 = transport->CreateMessage(s2.size());
  FairMQMessagePtr payload2 = transport->CreateMessage(1000);
  memcpy(header2->GetData(), s2.data(), s2.size());
  auto action = relayer.relay(header2, payload2);
  BOOST_CHECK_EQUAL(action, DataRelayer::Backpressured);
  BOOST_CHECK_NE(header2.get(), nullptr);
  BOOST_CHECK_NE(payload2.get(), nullptr);
}

BOOST_AUTO_TEST_CASE(SplitParts)
{
  Monitoring metrics;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};
  InputSpec spec2{"its", "ITS", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
    InputRoute{spec2, 0, "Fake2", 0},
  };

  std::vector<ForwardRoute> forwards;
  TimesliceIndex index;

  auto policy = CompletionPolicyHelpers::processWhenAny();
  DataRelayer relayer(policy, inputs, metrics, index);
  // Only two messages to fill the cache.
  relayer.setPipelineLength(1);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.splitPayloadIndex = 0;
  dh1.splitPayloadParts = 1;

  DataHeader dh2;
  dh2.dataDescription = "TRACKS";
  dh2.dataOrigin = "TPC";
  dh2.subSpecification = 0;
  dh2.splitPayloadIndex = 0;
  dh2.splitPayloadParts = 1;

  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");

  Stack s1{dh1, DataProcessingHeader{0, 1}};
  FairMQMessagePtr header = transport->CreateMessage(s1.size());
  FairMQMessagePtr payload = transport->CreateMessage(1000);
  memcpy(header->GetData(), s1.data(), s1.size());
  relayer.relay(header, payload);
  BOOST_CHECK_EQUAL(header.get(), nullptr);
  BOOST_CHECK_EQUAL(payload.get(), nullptr);
  // This fills the cache, and then waits.
  Stack s2{dh1, DataProcessingHeader{1, 1}};
  FairMQMessagePtr header2 = transport->CreateMessage(s2.size());
  FairMQMessagePtr payload2 = transport->CreateMessage(1000);
  memcpy(header2->GetData(), s2.data(), s2.size());
  auto action = relayer.relay(header2, payload2);
  BOOST_CHECK_EQUAL(action, DataRelayer::Backpressured);
  BOOST_CHECK_NE(header2.get(), nullptr);
  BOOST_CHECK_NE(payload2.get(), nullptr);
}
