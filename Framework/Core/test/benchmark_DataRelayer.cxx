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
#include "Framework/DataModelViews.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataProcessingStates.h"
#include "Framework/DataProcessingStats.h"
#include "Framework/DeviceState.h"
#include "Framework/DriverConfig.h"
#include "Framework/ServiceRegistryHelpers.h"
#include "Framework/TimingHelpers.h"
#include <Monitoring/Monitoring.h>
#include <fairmq/TransportFactory.h>
#include <cstring>
#include <vector>
#include <uv.h>

using Monitoring = o2::monitoring::Monitoring;
using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;
using RecordAction = o2::framework::DataRelayer::RecordAction;

struct BenchmarkServices {
  Monitoring monitoring;
  const DriverConfig driverConfig{.batch = false};
  DataProcessingStates states{
    TimingHelpers::defaultRealtimeBaseConfigurator(0, uv_default_loop()),
    TimingHelpers::defaultCPUTimeConfigurator(uv_default_loop())};
  DataProcessingStats stats{
    TimingHelpers::defaultRealtimeBaseConfigurator(0, uv_default_loop()),
    TimingHelpers::defaultCPUTimeConfigurator(uv_default_loop()),
    {}};
  DeviceState deviceState;
  ServiceRegistry registry;

  ServiceRegistryRef ref()
  {
    using MetricSpec = DataProcessingStats::MetricSpec;
    int quickUpdateInterval = 1;
    std::vector<MetricSpec> specs{
      MetricSpec{.name = "malformed_inputs", .metricId = static_cast<short>(ProcessingStatsId::MALFORMED_INPUTS), .minPublishInterval = quickUpdateInterval},
      MetricSpec{.name = "dropped_computations", .metricId = static_cast<short>(ProcessingStatsId::DROPPED_COMPUTATIONS), .minPublishInterval = quickUpdateInterval},
      MetricSpec{.name = "dropped_incoming_messages", .metricId = static_cast<short>(ProcessingStatsId::DROPPED_INCOMING_MESSAGES), .minPublishInterval = quickUpdateInterval},
      MetricSpec{.name = "relayed_messages", .metricId = static_cast<short>(ProcessingStatsId::RELAYED_MESSAGES), .minPublishInterval = quickUpdateInterval}};
    for (auto& spec : specs) {
      stats.registerMetric(spec);
    }

    ServiceRegistryRef r{registry};
    r.registerService(ServiceRegistryHelpers::handleForService<Monitoring>(&monitoring));
    r.registerService(ServiceRegistryHelpers::handleForService<DataProcessingStates>(&states));
    r.registerService(ServiceRegistryHelpers::handleForService<DataProcessingStats>(&stats));
    r.registerService(ServiceRegistryHelpers::handleForService<DriverConfig const>(&driverConfig));
    r.registerService(ServiceRegistryHelpers::handleForService<DeviceState>(&deviceState));
    return r;
  }
};

// a simple benchmark of the contribution of the pure message creation
// this was important when the benchmarks below included the message
// creation inside the benchmark loop, its somewhat obsolete now but
// we keep it for reference
static void BM_RelayMessageCreation(benchmark::State& state)
{
  DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;

  DataProcessingHeader dph{0, 1};
  Stack stack{dh, dph};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");

  for (auto _ : state) {
    fair::mq::MessagePtr header = transport->CreateMessage(stack.size());
    fair::mq::MessagePtr payload = transport->CreateMessage(1000);
    memcpy(header->GetData(), stack.data(), stack.size());
  }
}

BENCHMARK(BM_RelayMessageCreation);

// A simple test where an input is provided
// and the subsequent InputRecord is immediately requested.
static void BM_RelaySingleSlot(benchmark::State& state)
{
  BenchmarkServices services;
  InputSpec spec{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec, 0, "Fake", 0}};

  std::vector<ForwardRoute> forwards;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};
  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, index, services.ref(), -1);
  relayer.setPipelineLength(4);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;

  DataProcessingHeader dph{0, 1};
  Stack stack{dh, dph};
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  // we are creating the inflight messages once outside the benchmark
  // loop and make sure that they are moved back to the original vector
  // when processed by the relayer
  std::vector<fair::mq::MessagePtr> inflightMessages;
  inflightMessages.emplace_back(transport->CreateMessage(stack.size()));
  inflightMessages.emplace_back(transport->CreateMessage(1000));
  memcpy(inflightMessages[0]->GetData(), stack.data(), stack.size());

  DataRelayer::InputInfo fakeInfo{0, inflightMessages.size(), DataRelayer::InputType::Data, {ChannelIndex::INVALID}};
  for (auto _ : state) {
    relayer.relay(inflightMessages[0]->GetData(), inflightMessages.data(), fakeInfo, inflightMessages.size());
    std::vector<RecordAction> ready;
    relayer.getReadyToProcess(ready);
    assert(ready.size() == 1);
    assert(ready[0].slot.index == 0);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);
    auto result = relayer.consumeAllInputsForTimeslice(ready[0].slot);
    assert(result.size() == 1);
    assert((result.at(0) | count_parts{}) == 1);
    inflightMessages = std::move(result[0]);
  }
}

BENCHMARK(BM_RelaySingleSlot);

// This one will simulate a single input.
static void BM_RelayMultipleSlots(benchmark::State& state)
{
  BenchmarkServices services;
  InputSpec spec{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec, 0, "Fake", 0}};

  std::vector<ForwardRoute> forwards;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, index, services.ref(), -1);
  relayer.setPipelineLength(4);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  size_t timeslice = 0;

  DataProcessingHeader dph{timeslice, 1};
  Stack placeholder{dh, dph};

  // we are creating the inflight messages once outside the benchmark
  // loop and make sure that they are moved back to the original vector
  // when processed by the relayer
  std::vector<fair::mq::MessagePtr> inflightMessages;
  inflightMessages.emplace_back(transport->CreateMessage(placeholder.size()));
  inflightMessages.emplace_back(transport->CreateMessage(1000));

  for (auto _ : state) {
    Stack stack{dh, DataProcessingHeader{timeslice++, 1}};
    memcpy(inflightMessages[0]->GetData(), stack.data(), stack.size());

    DataRelayer::InputInfo fakeInfo{0, inflightMessages.size(), DataRelayer::InputType::Data, {ChannelIndex::INVALID}};
    relayer.relay(inflightMessages[0]->GetData(), inflightMessages.data(), fakeInfo, inflightMessages.size());
    std::vector<RecordAction> ready;
    relayer.getReadyToProcess(ready);
    assert(ready.size() == 1);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);
    auto result = relayer.consumeAllInputsForTimeslice(ready[0].slot);
    assert(result.size() == 1);
    assert((result.at(0) | count_parts{}) == 1);
    inflightMessages = std::move(result[0]);
  }
}

BENCHMARK(BM_RelayMultipleSlots);

/// In this case we have a record with two entries
static void BM_RelayMultipleRoutes(benchmark::State& state)
{
  BenchmarkServices services;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};
  InputSpec spec2{"tracks", "TPC", "TRACKS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
    InputRoute{spec2, 1, "Fake2", 0}};

  std::vector<ForwardRoute> forwards;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, index, services.ref(), -1);
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

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  size_t timeslice = 0;

  DataProcessingHeader dph1{timeslice, 1};
  Stack stack1{dh1, dph1};

  std::vector<fair::mq::MessagePtr> inflightMessages;
  inflightMessages.emplace_back(transport->CreateMessage(stack1.size()));
  inflightMessages.emplace_back(transport->CreateMessage(1000));

  memcpy(inflightMessages[0]->GetData(), stack1.data(), stack1.size());

  DataProcessingHeader dph2{timeslice, 1};
  Stack stack2{dh2, dph2};

  inflightMessages.emplace_back(transport->CreateMessage(stack2.size()));
  inflightMessages.emplace_back(transport->CreateMessage(1000));

  memcpy(inflightMessages[2]->GetData(), stack2.data(), stack2.size());

  for (auto _ : state) {
    DataRelayer::InputInfo fakeInfo{0, inflightMessages.size(), DataRelayer::InputType::Data, {ChannelIndex::INVALID}};
    relayer.relay(inflightMessages[0]->GetData(), &inflightMessages[0], fakeInfo, 2);
    std::vector<RecordAction> ready;
    relayer.getReadyToProcess(ready);
    assert(ready.size() == 1);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);

    DataRelayer::InputInfo fakeInfo2{0, inflightMessages.size(), DataRelayer::InputType::Data, {ChannelIndex::INVALID}};
    relayer.relay(inflightMessages[2]->GetData(), &inflightMessages[2], fakeInfo2, 2);
    ready.clear();
    relayer.getReadyToProcess(ready);
    assert(ready.size() == 1);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);
    auto result = relayer.consumeAllInputsForTimeslice(ready[0].slot);
    assert(result.size() == 2);
    assert((result.at(0) | count_parts{}) == 1);
    assert((result.at(1) | count_parts{}) == 1);
    inflightMessages = std::move(result[0]);
    inflightMessages.emplace_back(std::move(result[1][0]));
    inflightMessages.emplace_back(std::move(result[1][1]));
  }
}

BENCHMARK(BM_RelayMultipleRoutes);

/// In this case we have a record with two entries
static void BM_RelaySplitParts(benchmark::State& state)
{
  BenchmarkServices services;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
  };

  std::vector<ForwardRoute> forwards;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, index, services.ref(), -1);
  relayer.setPipelineLength(4);

  // Let's create a dummy O2 Message with two headers in the stack:
  // - DataHeader matching the one provided in the input
  DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;
  dh.payloadSize = 100;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  size_t timeslice = 0;
  const int nSplitParts = state.range(0);

  std::vector<std::unique_ptr<fair::mq::Message>> inflightMessages;
  inflightMessages.reserve(2 * nSplitParts);

  for (size_t i = 0; i < nSplitParts; ++i) {
    DataProcessingHeader dph{timeslice, 1};
    dh.splitPayloadIndex = i;
    dh.splitPayloadParts = nSplitParts;
    Stack stack{dh, dph};

    fair::mq::MessagePtr header = transport->CreateMessage(stack.size());
    fair::mq::MessagePtr payload = transport->CreateMessage(dh.payloadSize);

    memcpy(header->GetData(), stack.data(), stack.size());
    inflightMessages.emplace_back(std::move(header));
    inflightMessages.emplace_back(std::move(payload));
  }

  DataRelayer::InputInfo fakeInfo{0, inflightMessages.size(), DataRelayer::InputType::Data, {ChannelIndex::INVALID}};
  for (auto _ : state) {
    relayer.relay(inflightMessages[0]->GetData(), inflightMessages.data(), fakeInfo, inflightMessages.size());
    std::vector<RecordAction> ready;
    relayer.getReadyToProcess(ready);
    assert(ready.size() == 1);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);
    inflightMessages = std::move(relayer.consumeAllInputsForTimeslice(ready[0].slot)[0]);
  }
}

BENCHMARK(BM_RelaySplitParts)->Arg(10)->Arg(100)->Arg(1000);

static void BM_RelayMultiplePayloads(benchmark::State& state)
{
  BenchmarkServices services;
  InputSpec spec1{"clusters", "TPC", "CLUSTERS"};

  std::vector<InputRoute> inputs = {
    InputRoute{spec1, 0, "Fake1", 0},
  };

  std::vector<ForwardRoute> forwards;
  std::vector<InputChannelInfo> infos{1};
  TimesliceIndex index{1, infos};

  auto policy = CompletionPolicyHelpers::consumeWhenAny();
  DataRelayer relayer(policy, inputs, index, services.ref(), -1);
  relayer.setPipelineLength(4);

  // DataHeader matching the one provided in the input
  DataHeader dh;
  dh.dataDescription = "CLUSTERS";
  dh.dataOrigin = "TPC";
  dh.subSpecification = 0;
  dh.payloadSize = 100;

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  size_t timeslice = 0;
  const int nPayloads = state.range(0);
  std::vector<std::unique_ptr<fair::mq::Message>> inflightMessages;
  inflightMessages.reserve(nPayloads + 1);

  DataProcessingHeader dph{timeslice, 1};
  dh.splitPayloadIndex = nPayloads;
  dh.splitPayloadParts = nPayloads;
  Stack stack{dh, dph};
  fair::mq::MessagePtr header = transport->CreateMessage(stack.size());
  memcpy(header->GetData(), stack.data(), stack.size());
  inflightMessages.emplace_back(std::move(header));
  for (size_t i = 0; i < nPayloads; ++i) {
    inflightMessages.emplace_back(transport->CreateMessage(dh.payloadSize));
  }

  DataRelayer::InputInfo fakeInfo{0, inflightMessages.size(), DataRelayer::InputType::Data, {ChannelIndex::INVALID}};
  for (auto _ : state) {
    relayer.relay(inflightMessages[0]->GetData(), inflightMessages.data(), fakeInfo, inflightMessages.size(), nPayloads);
    std::vector<RecordAction> ready;
    relayer.getReadyToProcess(ready);
    assert(ready.size() == 1);
    assert(ready[0].op == CompletionPolicy::CompletionOp::Consume);
    inflightMessages = std::move(relayer.consumeAllInputsForTimeslice(ready[0].slot)[0]);
  }
}

BENCHMARK(BM_RelayMultiplePayloads)->Arg(10)->Arg(100)->Arg(1000);

BENCHMARK_MAIN();
