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
#include <benchmark/benchmark.h>

#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DataRelayer.h"
#include "Framework/DataProcessingHeader.h"
#include "../src/DataRelayerHelpers.h"
#include <Monitoring/Monitoring.h>
#include <fairmq/FairMQTransportFactory.h>
#include <cstring>
#include <array>
#include <vector>

using Monitoring = o2::monitoring::Monitoring;
using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;
using RecordAction = o2::framework::DataRelayer::RecordAction;

// A simple test where an input is provided
// and the subsequent InputRecord is immediately requested.
static void BM_RelayMessageCreation(benchmark::State& state)
{
  Monitoring metrics;
  InputSpec spec{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec, 0, "Fake", 0}};

  std::vector<ForwardRoute> forwards;
  TimesliceIndex index{1};

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, metrics, index);
  relayer.setPipelineLength(4);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;

  DataProcessingHeader dph{0, 1};
  Stack stack{dh, dph};
  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");

  for (auto _ : state) {
    // FIXME: Understand why pausing the timer makes it slower..
    //state.PauseTiming();
    FairMQMessagePtr header = transport->CreateMessage(stack.size());
    FairMQMessagePtr payload = transport->CreateMessage(1000);
    memcpy(header->GetData(), stack.data(), stack.size());
    //state.ResumeTiming();
  }
  // One for the header, one for the payload
}

BENCHMARK(BM_RelayMessageCreation);

// A simple test where an input is provided
// and the subsequent InputRecord is immediately requested.
static void BM_RelaySingleSlot(benchmark::State& state)
{
  Monitoring metrics;
  InputSpec spec{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec, 0, "Fake", 0}};

  std::vector<ForwardRoute> forwards;
  TimesliceIndex index{1};

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, metrics, index);
  relayer.setPipelineLength(4);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;

  DataProcessingHeader dph{0, 1};
  Stack stack{dh, dph};
  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");

  for (auto _ : state) {
    // FIXME: Understand why pausing the timer makes it slower..
    //state.PauseTiming();
    std::array<FairMQMessagePtr, 2> messages;
    messages[0] = transport->CreateMessage(stack.size());
    messages[1] = transport->CreateMessage(1000);
    FairMQMessagePtr& header = messages[0];
    FairMQMessagePtr& payload = messages[1];
    memcpy(header->GetData(), stack.data(), stack.size());
    //state.ResumeTiming();

    relayer.relay(header->GetData(), messages.data(), messages.size());
    std::vector<RecordAction> ready;
    relayer.getReadyToProcess(ready);
    assert(ready.size() == 1);
    assert(ready[0].slot.index == 0);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);
    auto result = relayer.consumeAllInputsForTimeslice(ready[0].slot);
    assert(result.size() == 1);
    assert(result.at(0).size() == 1);
  }
  // One for the header, one for the payload
}

BENCHMARK(BM_RelaySingleSlot);

// This one will simulate a single input.
static void BM_RelayMultipleSlots(benchmark::State& state)
{
  Monitoring metrics;
  InputSpec spec{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec, 0, "Fake", 0}};

  std::vector<ForwardRoute> forwards;
  TimesliceIndex index{1};

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, metrics, index);
  relayer.setPipelineLength(4);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;

  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");
  size_t timeslice = 0;

  for (auto _ : state) {
    // FIXME: Understand why pausing the timer makes it slower..
    //state.PauseTiming();

    DataProcessingHeader dph{timeslice++, 1};
    Stack stack{dh, dph};
    std::array<FairMQMessagePtr, 2> messages;
    messages[0] = transport->CreateMessage(stack.size());
    messages[1] = transport->CreateMessage(1000);
    FairMQMessagePtr& header = messages[0];
    FairMQMessagePtr& payload = messages[1];

    memcpy(header->GetData(), stack.data(), stack.size());
    //state.ResumeTiming();

    relayer.relay(header->GetData(), messages.data(), messages.size());
    std::vector<RecordAction> ready;
    relayer.getReadyToProcess(ready);
    assert(ready.size() == 1);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);
    auto result = relayer.consumeAllInputsForTimeslice(ready[0].slot);
    assert(result.size() == 1);
    assert(result.at(0).size() == 1);
  }
  // One for the header, one for the payload
}

BENCHMARK(BM_RelayMultipleSlots);

/// In this case we have a record with two entries
static void BM_RelayMultipleRoutes(benchmark::State& state)
{
  Monitoring metrics;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};
  InputSpec spec2{"tracks", "TPC", "TRACKS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
    InputRoute{spec2, 1, "Fake2", 0}};

  std::vector<ForwardRoute> forwards;
  TimesliceIndex index{1};

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, metrics, index);
  relayer.setPipelineLength(4);

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
  size_t timeslice = 0;

  for (auto _ : state) {
    // FIXME: Understand why pausing the timer makes it slower..
    //state.PauseTiming();

    DataProcessingHeader dph1{timeslice, 1};
    Stack stack1{dh1, dph1};

    std::array<FairMQMessagePtr, 4> messages;
    messages[0] = transport->CreateMessage(stack1.size());
    messages[1] = transport->CreateMessage(1000);
    FairMQMessagePtr& header1 = messages[0];
    FairMQMessagePtr& payload1 = messages[1];

    memcpy(header1->GetData(), stack1.data(), stack1.size());

    DataProcessingHeader dph2{timeslice, 1};
    Stack stack2{dh2, dph2};

    messages[2] = transport->CreateMessage(stack2.size());
    messages[3] = transport->CreateMessage(1000);
    FairMQMessagePtr& header2 = messages[2];
    FairMQMessagePtr& payload2 = messages[3];

    memcpy(header2->GetData(), stack2.data(), stack2.size());
    //state.ResumeTiming();

    relayer.relay(header1->GetData(), &messages[0], 2);
    std::vector<RecordAction> ready;
    relayer.getReadyToProcess(ready);
    assert(ready.size() == 1);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);

    relayer.relay(header2->GetData(), &messages[2], 2);
    ready.clear();
    relayer.getReadyToProcess(ready);
    assert(ready.size() == 1);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);
    auto result = relayer.consumeAllInputsForTimeslice(ready[0].slot);
    assert(result.size() == 2);
    assert(result.at(0).size() == 1);
    assert(result.at(1).size() == 1);
  }
  // One for the header, one for the payload
}

BENCHMARK(BM_RelayMultipleRoutes);

/// In this case we have a record with two entries
static void BM_RelaySplitParts(benchmark::State& state)
{
  Monitoring metrics;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
  };

  std::vector<ForwardRoute> forwards;
  TimesliceIndex index{1};

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, metrics, index);
  relayer.setPipelineLength(4);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;

  auto transport = FairMQTransportFactory::CreateTransportFactory("zeromq");
  size_t timeslice = 0;

  for (auto _ : state) {
    // FIXME: Understand why pausing the timer makes it slower..
    state.PauseTiming();
    const int nSplitParts = 10;
    std::vector<std::unique_ptr<FairMQMessage>> splitParts;
    splitParts.reserve(nSplitParts);

    for (size_t i = 0; i < nSplitParts; ++i) {
      DataProcessingHeader dph1{timeslice, 1};
      dh1.splitPayloadIndex = i;
      dh1.splitPayloadParts = nSplitParts;
      Stack stack1{dh1, dph1};

      FairMQMessagePtr header1 = transport->CreateMessage(stack1.size());
      FairMQMessagePtr payload1 = transport->CreateMessage(100);

      memcpy(header1->GetData(), stack1.data(), stack1.size());
      splitParts.emplace_back(std::move(header1));
      splitParts.emplace_back(std::move(payload1));
    }
    state.ResumeTiming();

    relayer.relay(splitParts[0]->GetData(), splitParts.data(), splitParts.size());
    std::vector<RecordAction> ready;
    relayer.getReadyToProcess(ready);
    assert(ready.size() == 1);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);
  }
  // One for the header, one for the payload
}

BENCHMARK(BM_RelaySplitParts);

BENCHMARK_MAIN();
