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

#include <catch_amalgamated.hpp>
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include "MemoryResources/MemoryResources.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DataRelayer.h"
#include "../src/DataRelayerHelpers.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/WorkflowSpec.h"
#include <Monitoring/Monitoring.h>
#include <fairmq/TransportFactory.h>
#include <array>
#include <vector>

using Monitoring = o2::monitoring::Monitoring;
using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;
using RecordAction = o2::framework::DataRelayer::RecordAction;

// A simple test where an input is provided
// and the subsequent InputRecord is immediately requested.
TEST_CASE("TestNoWait")
{
  Monitoring metrics;
  InputSpec spec{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec, 0, "Fake", 0}};

  std::vector<ForwardRoute> forwards;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

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
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  std::array<fair::mq::MessagePtr, 2> messages;
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  messages[0] = o2::pmr::getMessage(Stack{channelAlloc, dh, dph});
  messages[1] = transport->CreateMessage(1000);
  fair::mq::MessagePtr& header = messages[0];
  fair::mq::MessagePtr& payload = messages[1];
  relayer.relay(header->GetData(), messages.data(), messages.size());
  std::vector<RecordAction> ready;
  relayer.getReadyToProcess(ready);
  REQUIRE(ready.size() == 1);
  REQUIRE(ready[0].slot.index == 0);
  REQUIRE(ready[0].op == CompletionPolicy::CompletionOp::Consume);
  REQUIRE(header.get() == nullptr);
  REQUIRE(payload.get() == nullptr);
  auto result = relayer.consumeAllInputsForTimeslice(ready[0].slot);
  // one MessageSet with one PartRef with header and payload
  REQUIRE(result.size() == 1);
  REQUIRE(result.at(0).size() == 1);
}

//
TEST_CASE("TestNoWaitMatcher")
{
  Monitoring metrics;
  auto specs = o2::framework::select("clusters:TPC/CLUSTERS");

  std::vector<InputRoute> inputs = {
    InputRoute{specs[0], 0, "Fake", 0}};

  std::vector<ForwardRoute> forwards;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

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
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  std::array<fair::mq::MessagePtr, 2> messages;
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  messages[0] = o2::pmr::getMessage(Stack{channelAlloc, dh, dph});
  messages[1] = transport->CreateMessage(1000);
  fair::mq::MessagePtr& header = messages[0];
  fair::mq::MessagePtr& payload = messages[1];
  relayer.relay(header->GetData(), messages.data(), messages.size());
  std::vector<RecordAction> ready;
  relayer.getReadyToProcess(ready);
  REQUIRE(ready.size() == 1);
  REQUIRE(ready[0].slot.index == 0);
  REQUIRE(ready[0].op == CompletionPolicy::CompletionOp::Consume);
  REQUIRE(header.get() == nullptr);
  REQUIRE(payload.get() == nullptr);
  auto result = relayer.consumeAllInputsForTimeslice(ready[0].slot);
  // one MessageSet with one PartRef with header and payload
  REQUIRE(result.size() == 1);
  REQUIRE(result.at(0).size() == 1);
}

// This test a more complicated set of inputs, and verifies that data is
// correctly relayed before being processed.
TEST_CASE("TestRelay")
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

  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

  auto policy = CompletionPolicyHelpers::consumeWhenAll();
  DataRelayer relayer(policy, inputs, metrics, index);
  relayer.setPipelineLength(4);

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());

  auto createMessage = [&transport, &channelAlloc, &relayer](DataHeader& dh, size_t time) {
    std::array<fair::mq::MessagePtr, 2> messages;
    messages[0] = o2::pmr::getMessage(Stack{channelAlloc, dh, DataProcessingHeader{time, 1}});
    messages[1] = transport->CreateMessage(1000);
    fair::mq::MessagePtr& header = messages[0];
    fair::mq::MessagePtr& payload = messages[1];
    relayer.relay(header->GetData(), messages.data(), messages.size());
    REQUIRE(header.get() == nullptr);
    REQUIRE(payload.get() == nullptr);
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
  REQUIRE(ready.size() == 0);

  createMessage(dh2, 0);
  ready.clear();
  relayer.getReadyToProcess(ready);
  REQUIRE(ready.size() == 1);
  REQUIRE(ready[0].slot.index == 0);
  REQUIRE(ready[0].op == CompletionPolicy::CompletionOp::Consume);

  auto result = relayer.consumeAllInputsForTimeslice(ready[0].slot);
  // two MessageSets, each with one PartRef
  REQUIRE(result.size() == 2);
  REQUIRE(result.at(0).size() == 1);
  REQUIRE(result.at(1).size() == 1);
}

// This test a more complicated set of inputs, and verifies that data is
// correctly relayed before being processed.
TEST_CASE("TestRelayBug")
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

  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

  auto policy = CompletionPolicyHelpers::consumeWhenAll();
  DataRelayer relayer(policy, inputs, metrics, index);
  relayer.setPipelineLength(3);

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());

  auto createMessage = [&transport, &channelAlloc, &relayer](DataHeader& dh, size_t time) {
    std::array<fair::mq::MessagePtr, 2> messages;
    messages[0] = o2::pmr::getMessage(Stack{channelAlloc, dh, DataProcessingHeader{time, 1}});
    messages[1] = transport->CreateMessage(1000);
    fair::mq::MessagePtr& header = messages[0];
    fair::mq::MessagePtr& payload = messages[1];
    relayer.relay(header->GetData(), messages.data(), messages.size());
    REQUIRE(header.get() == nullptr);
    REQUIRE(payload.get() == nullptr);
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
  REQUIRE(ready.size() == 0);
  createMessage(dh1, 1);
  ready.clear();
  relayer.getReadyToProcess(ready);
  REQUIRE(ready.size() == 0);
  createMessage(dh2, 0);
  ready.clear();
  relayer.getReadyToProcess(ready);
  REQUIRE(ready.size() == 1);
  REQUIRE(ready[0].slot.index == 0);
  REQUIRE(ready[0].op == CompletionPolicy::CompletionOp::Consume);
  auto result = relayer.consumeAllInputsForTimeslice(ready[0].slot);
  createMessage(dh2, 1);
  ready.clear();
  relayer.getReadyToProcess(ready);
  REQUIRE(ready.size() == 1);
  REQUIRE(ready[0].slot.index == 1);
  REQUIRE(ready[0].op == CompletionPolicy::CompletionOp::Consume);
  result = relayer.consumeAllInputsForTimeslice(ready[0].slot);
}

// This tests a simple cache pruning, where a single input is shifted out of
// the cache.
TEST_CASE("TestCache")
{
  Monitoring metrics;
  InputSpec spec{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec, 0, "Fake", 0}};
  std::vector<ForwardRoute> forwards;

  auto policy = CompletionPolicyHelpers::consumeWhenAll();
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};
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
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto createMessage = [&transport, &channelAlloc, &relayer, &dh](auto const& h) {
    std::array<fair::mq::MessagePtr, 2> messages;
    messages[0] = o2::pmr::getMessage(Stack{channelAlloc, dh, h});
    messages[1] = transport->CreateMessage(1000);
    fair::mq::MessagePtr& header = messages[0];
    fair::mq::MessagePtr& payload = messages[1];
    auto res = relayer.relay(header->GetData(), messages.data(), messages.size());
    REQUIRE((res.type != DataRelayer::RelayChoice::Type::WillRelay || header.get() == nullptr));
    REQUIRE((res.type != DataRelayer::RelayChoice::Type::WillRelay || payload.get() == nullptr));
    REQUIRE((res.type != DataRelayer::RelayChoice::Type::Backpressured || header.get() != nullptr));
    REQUIRE((res.type != DataRelayer::RelayChoice::Type::Backpressured || payload.get() != nullptr));
  };

  // This fills the cache, and then empties it.
  createMessage(DataProcessingHeader{0, 1});
  createMessage(DataProcessingHeader{1, 1});
  std::vector<RecordAction> ready;
  relayer.getReadyToProcess(ready);
  REQUIRE(ready.size() == 2);
  REQUIRE(ready[0].slot.index == 1);
  REQUIRE(ready[1].slot.index == 0);
  REQUIRE(ready[0].op == CompletionPolicy::CompletionOp::Consume);
  REQUIRE(ready[1].op == CompletionPolicy::CompletionOp::Consume);
  for (size_t i = 0; i < ready.size(); ++i) {
    auto result = relayer.consumeAllInputsForTimeslice(ready[i].slot);
  }

  // This fills the cache and makes 2 obsolete.
  createMessage(DataProcessingHeader{2, 1});
  createMessage(DataProcessingHeader{3, 1});
  createMessage(DataProcessingHeader{4, 1});
  ready.clear();
  relayer.getReadyToProcess(ready);
  REQUIRE(ready.size() == 2);

  auto result1 = relayer.consumeAllInputsForTimeslice(ready[0].slot);
  auto result2 = relayer.consumeAllInputsForTimeslice(ready[1].slot);
  // One for the header, one for the payload
  REQUIRE(result1.size() == 1);
  REQUIRE(result2.size() == 1);
}

// This the any policy. Even when there are two inputs, given the any policy
// it will run immediately.
TEST_CASE("TestPolicies")
{
  Monitoring metrics;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};
  InputSpec spec2{"tracks", "TPC", "TRACKS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
    InputRoute{spec2, 1, "Fake2", 0},
  };

  std::vector<ForwardRoute> forwards;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

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

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto createMessage = [&transport, &channelAlloc, &relayer](auto const& dh, auto const& h) {
    std::array<fair::mq::MessagePtr, 2> messages;
    messages[0] = o2::pmr::getMessage(Stack{channelAlloc, dh, h});
    messages[1] = transport->CreateMessage(1000);
    fair::mq::MessagePtr& header = messages[0];
    return relayer.relay(header->GetData(), messages.data(), messages.size());
  };

  // This fills the cache, and then empties it.
  createMessage(dh1, DataProcessingHeader{0, 1});
  std::vector<RecordAction> ready1;
  relayer.getReadyToProcess(ready1);
  REQUIRE(ready1.size() == 1);
  REQUIRE(ready1[0].slot.index == 0);
  REQUIRE(ready1[0].op == CompletionPolicy::CompletionOp::Process);

  createMessage(dh1, DataProcessingHeader{1, 1});
  std::vector<RecordAction> ready2;
  relayer.getReadyToProcess(ready2);
  REQUIRE(ready2.size() == 1);
  REQUIRE(ready2[0].slot.index == 1);
  REQUIRE(ready2[0].op == CompletionPolicy::CompletionOp::Process);

  createMessage(dh2, DataProcessingHeader{1, 1});
  std::vector<RecordAction> ready3;
  relayer.getReadyToProcess(ready3);
  REQUIRE(ready3.size() == 1);
  REQUIRE(ready3[0].slot.index == 1);
  REQUIRE(ready3[0].op == CompletionPolicy::CompletionOp::Consume);
}

/// Test that the clear method actually works.
TEST_CASE("TestClear")
{
  Monitoring metrics;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};
  InputSpec spec2{"tracks", "TPC", "TRACKS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
    InputRoute{spec2, 1, "Fake2", 0},
  };

  std::vector<ForwardRoute> forwards;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

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

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  auto createMessage = [&transport, &channelAlloc, &relayer](auto const& dh, auto const& h) {
    std::array<fair::mq::MessagePtr, 2> messages;
    messages[0] = o2::pmr::getMessage(Stack{channelAlloc, dh, h});
    messages[1] = transport->CreateMessage(1000);
    fair::mq::MessagePtr& header = messages[0];
    return relayer.relay(header->GetData(), messages.data(), messages.size());
  };

  // This fills the cache, and then empties it.
  createMessage(dh1, DataProcessingHeader{0, 1});
  createMessage(dh1, DataProcessingHeader{1, 1});
  createMessage(dh2, DataProcessingHeader{1, 1});
  relayer.clear();
  std::vector<RecordAction> ready;
  relayer.getReadyToProcess(ready);
  REQUIRE(ready.size() == 0);
}

/// Test that the clear method actually works.
TEST_CASE("TestTooMany")
{
  Monitoring metrics;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};
  InputSpec spec2{"tracks", "TPC", "TRACKS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
    InputRoute{spec2, 1, "Fake2", 0},
  };

  std::vector<ForwardRoute> forwards;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

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

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());

  std::array<fair::mq::MessagePtr, 4> messages;
  messages[0] = o2::pmr::getMessage(Stack{channelAlloc, dh1, DataProcessingHeader{0, 1}});
  messages[1] = transport->CreateMessage(1000);
  fair::mq::MessagePtr& header = messages[0];
  fair::mq::MessagePtr& payload = messages[1];
  relayer.relay(header->GetData(), &messages[0], 2);
  REQUIRE(header.get() == nullptr);
  REQUIRE(payload.get() == nullptr);
  // This fills the cache, and then waits.
  messages[2] = o2::pmr::getMessage(Stack{channelAlloc, dh1, DataProcessingHeader{1, 1}});
  messages[3] = transport->CreateMessage(1000);
  fair::mq::MessagePtr& header2 = messages[2];
  fair::mq::MessagePtr& payload2 = messages[3];
  auto action = relayer.relay(header2->GetData(), &messages[2], 2);
  REQUIRE(action.type == DataRelayer::RelayChoice::Type::Backpressured);
  REQUIRE(header2.get() != nullptr);
  REQUIRE(payload2.get() != nullptr);
}

TEST_CASE("SplitParts")
{
  Monitoring metrics;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};
  InputSpec spec2{"its", "ITS", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
    InputRoute{spec2, 0, "Fake2", 0},
  };

  std::vector<ForwardRoute> forwards;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

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

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());

  std::array<fair::mq::MessagePtr, 6> messages;
  messages[0] = o2::pmr::getMessage(Stack{channelAlloc, dh1, DataProcessingHeader{0, 1}});
  messages[1] = transport->CreateMessage(1000);
  fair::mq::MessagePtr& header = messages[0];
  fair::mq::MessagePtr& payload = messages[1];
  relayer.relay(header->GetData(), &messages[0], 2);
  REQUIRE(header.get() == nullptr);
  REQUIRE(payload.get() == nullptr);
  // This fills the cache, and then waits.
  messages[2] = o2::pmr::getMessage(Stack{channelAlloc, dh1, DataProcessingHeader{1, 1}});
  messages[3] = transport->CreateMessage(1000);
  fair::mq::MessagePtr& header2 = messages[2];
  fair::mq::MessagePtr& payload2 = messages[3];
  auto action = relayer.relay(header2->GetData(), &messages[2], 2);
  REQUIRE(action.type == DataRelayer::RelayChoice::Type::Backpressured);
  CHECK(action.timeslice.value == 1);
  REQUIRE(header2.get() != nullptr);
  REQUIRE(payload2.get() != nullptr);
  // This fills the cache, and then waits.
  messages[4] = o2::pmr::getMessage(Stack{channelAlloc, dh1, DataProcessingHeader{1, 1}});
  messages[5] = transport->CreateMessage(1000);
  auto action2 = relayer.relay(header2->GetData(), &messages[4], 2);
  REQUIRE(action2.type == DataRelayer::RelayChoice::Type::Backpressured);
  CHECK(action2.timeslice.value == 1);
  REQUIRE(header2.get() != nullptr);
  REQUIRE(payload2.get() != nullptr);
}

TEST_CASE("SplitPayloadPairs")
{
  Monitoring metrics;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
  };

  std::vector<ForwardRoute> forwards;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, metrics, index);
  relayer.setPipelineLength(4);

  DataHeader dh{"CLUSTERS", "TPC", 0};

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
  size_t timeslice = 0;

  const int nSplitParts = 100;
  std::vector<std::unique_ptr<fair::mq::Message>> splitParts;
  splitParts.reserve(2 * nSplitParts);

  for (size_t i = 0; i < nSplitParts; ++i) {
    dh.splitPayloadIndex = i;
    dh.splitPayloadParts = nSplitParts;

    fair::mq::MessagePtr header = o2::pmr::getMessage(Stack{channelAlloc, dh, DataProcessingHeader{timeslice, 1}});
    fair::mq::MessagePtr payload = transport->CreateMessage(100);

    splitParts.emplace_back(std::move(header));
    splitParts.emplace_back(std::move(payload));
  }
  REQUIRE(splitParts.size() == 2 * nSplitParts);

  relayer.relay(splitParts[0]->GetData(), splitParts.data(), splitParts.size());
  std::vector<RecordAction> ready;
  relayer.getReadyToProcess(ready);
  REQUIRE(ready.size() == 1);
  REQUIRE(ready[0].op == CompletionPolicy::CompletionOp::Consume);
  auto messageSet = relayer.consumeAllInputsForTimeslice(ready[0].slot);
  // we have one input route and thus one message set containing pairs for all
  // payloads
  REQUIRE(messageSet.size() == 1);
  REQUIRE(messageSet[0].size() == nSplitParts);
  REQUIRE(messageSet[0].getNumberOfPayloads(0) == 1);
}

TEST_CASE("SplitPayloadSequence")
{
  Monitoring metrics;
  InputSpec spec1{"clusters", "TST", "COUNTER"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
  };

  std::vector<ForwardRoute> forwards;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, metrics, index);
  relayer.setPipelineLength(4);

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  size_t timeslice = 0;

  std::vector<size_t> sequenceSize;
  size_t nTotalPayloads = 0;

  auto createSequence = [&nTotalPayloads, &timeslice, &sequenceSize, &transport, &relayer](size_t nPayloads) -> void {
    auto channelAlloc = o2::pmr::getTransportAllocator(transport.get());
    std::vector<std::unique_ptr<fair::mq::Message>> messages;
    messages.reserve(nPayloads + 1);
    DataHeader dh{"COUNTER", "TST", 0};

    // one header with index set to the number of split parts indicates sequence
    // of payloads without additional headers
    dh.splitPayloadIndex = nPayloads;
    dh.splitPayloadParts = nPayloads;
    fair::mq::MessagePtr header = o2::pmr::getMessage(Stack{channelAlloc, dh, DataProcessingHeader{timeslice, 1}});
    messages.emplace_back(std::move(header));

    for (size_t i = 0; i < nPayloads; ++i) {
      messages.emplace_back(transport->CreateMessage(100));
      *(reinterpret_cast<size_t*>(messages.back()->GetData())) = nTotalPayloads;
      ++nTotalPayloads;
    }
    REQUIRE(messages.size() == nPayloads + 1);
    relayer.relay(messages[0]->GetData(), messages.data(), messages.size(), nPayloads);
    sequenceSize.emplace_back(nPayloads);
  };
  createSequence(100);
  createSequence(1);
  createSequence(42);

  std::vector<RecordAction> ready;
  relayer.getReadyToProcess(ready);
  REQUIRE(ready.size() == 1);
  REQUIRE(ready[0].op == CompletionPolicy::CompletionOp::Consume);
  auto messageSet = relayer.consumeAllInputsForTimeslice(ready[0].slot);
  // we have one input route
  REQUIRE(messageSet.size() == 1);
  // one message set containing number of added sequences of messages
  REQUIRE(messageSet[0].size() == sequenceSize.size());
  size_t counter = 0;
  for (auto seqid = 0; seqid < sequenceSize.size(); ++seqid) {
    REQUIRE(messageSet[0].getNumberOfPayloads(seqid) == sequenceSize[seqid]);
    for (auto pi = 0; pi < messageSet[0].getNumberOfPayloads(seqid); ++pi) {
      REQUIRE(messageSet[0].payload(seqid, pi));
      auto const* data = messageSet[0].payload(seqid, pi)->GetData();
      REQUIRE(*(reinterpret_cast<size_t const*>(data)) == counter);
      ++counter;
    }
  }
}
