// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <Monitoring/Monitoring.h>
#include <fairmq/FairMQTransportFactory.h>
#include <cstring>

using Monitoring = o2::monitoring::Monitoring;
using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;

// A simple test where an input is provided
// and the subsequent InputRecord is immediately requested.
static void BM_RelayMessageCreation(benchmark::State& state)
{
  Monitoring metrics;
  InputSpec spec{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec, 0, "Fake", 0}};

  std::vector<ForwardRoute> forwards;
  TimesliceIndex index;

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, forwards, metrics, index);
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
  TimesliceIndex index;

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, forwards, metrics, index);
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

    relayer.relay(std::move(header), std::move(payload));
    auto ready = relayer.getReadyToProcess();
    assert(ready.size() == 1);
    assert(ready[0].slot.index == 0);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);
    auto result = relayer.getInputsForTimeslice(ready[0].slot);
    assert(result.size() == 2);
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
  TimesliceIndex index;

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, forwards, metrics, index);
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
    FairMQMessagePtr header = transport->CreateMessage(stack.size());
    FairMQMessagePtr payload = transport->CreateMessage(1000);

    memcpy(header->GetData(), stack.data(), stack.size());
    //state.ResumeTiming();

    relayer.relay(std::move(header), std::move(payload));
    auto ready = relayer.getReadyToProcess();
    assert(ready.size() == 1);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);
    auto result = relayer.getInputsForTimeslice(ready[0].slot);
    assert(result.size() == 2);
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
  TimesliceIndex index;

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, forwards, metrics, index);
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

    FairMQMessagePtr header1 = transport->CreateMessage(stack1.size());
    FairMQMessagePtr payload1 = transport->CreateMessage(1000);

    memcpy(header1->GetData(), stack1.data(), stack1.size());

    DataProcessingHeader dph2{timeslice, 1};
    Stack stack2{dh2, dph2};

    FairMQMessagePtr header2 = transport->CreateMessage(stack2.size());
    FairMQMessagePtr payload2 = transport->CreateMessage(1000);

    memcpy(header2->GetData(), stack2.data(), stack2.size());
    //state.ResumeTiming();

    relayer.relay(std::move(header1), std::move(payload1));
    auto ready = relayer.getReadyToProcess();
    assert(ready.size() == 1);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);

    relayer.relay(std::move(header2), std::move(payload2));
    ready = relayer.getReadyToProcess();
    assert(ready.size() == 1);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);
    auto result = relayer.getInputsForTimeslice(ready[0].slot);
    assert(result.size() == 4);
  }
  // One for the header, one for the payload
}

BENCHMARK(BM_RelayMultipleRoutes);

BENCHMARK_MAIN();
