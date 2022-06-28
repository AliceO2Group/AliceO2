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

#include "Framework/AlgorithmSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include "MemoryResources/MemoryResources.h"
#include <fairmq/Device.h>
#include <memory>
#include <random>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;

// we need to specify customizations before including Framework/runDataProcessing
// customize consumer to process immediately what comes in
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // we customize the pipeline processors to consume data as it comes
  using CompletionPolicy = o2::framework::CompletionPolicy;
  using CompletionPolicyHelpers = o2::framework::CompletionPolicyHelpers;
  policies.push_back(CompletionPolicyHelpers::defineByName("consumer", CompletionPolicy::CompletionOp::Consume));
  policies.push_back(CompletionPolicyHelpers::defineByName("spectator", CompletionPolicy::CompletionOp::Consume));
}

#include "Framework/runDataProcessing.h"

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(fatal) << R"(Test condition ")" #condition R"(" failed)"; \
  }

namespace test
{
// a header with the information expected in the payload
// will be sent on the header stack
struct SequenceDesc : public o2::header::BaseHeader {
  //static data for this header type/version
  static constexpr uint32_t sVersion{1};
  static constexpr o2::header::HeaderType sHeaderType{o2::header::String2<uint64_t>("SequDesc")};
  static constexpr o2::header::SerializationMethod sSerializationMethod{o2::header::gSerializationMethodNone};

  size_t iteration = 0;
  size_t nPayloads = 0;
  size_t initialValue = 0;

  constexpr SequenceDesc(size_t i, size_t n, size_t v)
    : BaseHeader(sizeof(SequenceDesc), sHeaderType, sSerializationMethod, sVersion), iteration(i), nPayloads(n), initialValue(v)
  {
  }
};

} // namespace test

std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const& config)
{
  struct Attributes {
    using EngineT = std::mt19937;
    using DistributionT = std::uniform_int_distribution<>;
    size_t nRolls = 2;
    EngineT gen;
    DistributionT distrib;
    size_t iteration = 0;
    std::string channelName;
  };

  std::random_device rd;
  auto attributes = std::make_shared<Attributes>();
  attributes->nRolls = 4;
  attributes->gen = std::mt19937(rd());
  attributes->distrib = std::uniform_int_distribution<>{1, 20};

  std::vector<DataProcessorSpec> workflow;
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // a producer process steered by a timer
  //
  // the compute callback of the producer
  // Producing three types of output:
  // 1. via default DPL Allocator
  // 2. multiple payloads in split-payloads format (header-payload pairs)
  // 3. multiple payload sequence with one header
  auto producerCallback = [attributes](InputRecord& inputs, DataAllocator& outputs, ControlService& control, RawDeviceService& rds) {
    auto& counter = attributes->iteration;
    auto& channelName = attributes->channelName;
    auto& nRolls = attributes->nRolls;
    outputs.make<int>(OutputRef{"allocator", 0}) = counter;

    if (channelName.empty()) {
      OutputSpec const query{"TST", "SEQUENCE", 0};
      auto outputRoutes = rds.spec().outputs;
      for (auto& route : outputRoutes) {
        if (DataSpecUtils::match(route.matcher, query)) {
          channelName = route.channel;
          break;
        }
      }
      ASSERT_ERROR(channelName.length() > 0);
    }
    fair::mq::Device& device = *(rds.device());
    auto transport = device.GetChannel(channelName, 0).Transport();
    auto channelAlloc = o2::pmr::getTransportAllocator(transport);

    auto const* dph = DataRefUtils::getHeader<DataProcessingHeader*>(inputs.get("timer"));
    test::SequenceDesc sd{counter, 0, 0};

    fair::mq::Parts messages;
    auto createSequence = [&dph, &sd, &attributes, &transport, &channelAlloc, &messages](size_t nPayloads, DataHeader dh) -> void {
      // one header with index set to the number of split parts indicates sequence
      // of payloads without additional headers
      dh.payloadSize = sizeof(size_t);
      dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;
      dh.splitPayloadIndex = nPayloads;
      dh.splitPayloadParts = nPayloads;
      sd.nPayloads = nPayloads;
      sd.initialValue = attributes->distrib(attributes->gen);
      fair::mq::MessagePtr header = o2::pmr::getMessage(Stack{channelAlloc, dh, *dph, sd});
      messages.AddPart(std::move(header));

      for (size_t i = 0; i < nPayloads; ++i) {
        fair::mq::MessagePtr payload = transport->CreateMessage(dh.payloadSize);
        *(reinterpret_cast<size_t*>(payload->GetData())) = sd.initialValue + i;
        messages.AddPart(std::move(payload));
      }
    };

    auto createPairs = [&dph, &transport, &channelAlloc, &messages](size_t nPayloads, DataHeader dh) -> void {
      // one header with index set to the number of split parts indicates sequence
      // of payloads without additional headers
      dh.payloadSize = sizeof(size_t);
      dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;
      dh.splitPayloadIndex = 0;
      dh.splitPayloadParts = nPayloads;
      for (size_t i = 0; i < nPayloads; ++i) {
        dh.splitPayloadIndex = i;
        fair::mq::MessagePtr header = o2::pmr::getMessage(Stack{channelAlloc, dh, *dph});
        messages.AddPart(std::move(header));
        fair::mq::MessagePtr payload = transport->CreateMessage(dh.payloadSize);
        *(reinterpret_cast<size_t*>(payload->GetData())) = i;
        messages.AddPart(std::move(payload));
      }
    };

    createSequence(attributes->distrib(attributes->gen), DataHeader{"SEQUENCE", "TST", 0});
    createPairs(counter + 1, DataHeader{"PAIR", "TST", 0});

    // using utility from ExternalFairMQDeviceProxy
    sendOnChannel(device, messages, channelName, (size_t)-1);

    if (++(counter) >= nRolls) {
      // send the end of stream signal, this is transferred by the proxies
      // and allows to properly terminate downstream devices
      control.endOfStream();
      control.readyToQuit(QuitRequest::Me);
    }
  };

  workflow.emplace_back(DataProcessorSpec{"producer",
                                          {InputSpec{"timer", "TST", "TIMER", 0, Lifetime::Timer}},
                                          {OutputSpec{{"pair"}, "TST", "PAIR", 0, Lifetime::Timeframe},
                                           OutputSpec{{"sequence"}, "TST", "SEQUENCE", 0, Lifetime::Timeframe},
                                           OutputSpec{{"allocator"}, "TST", "ALLOCATOR", 0, Lifetime::Timeframe}},
                                          AlgorithmSpec{adaptStateless(producerCallback)},
                                          {ConfigParamSpec{"period-timer", VariantType::Int, 100000, {"period of timer"}}}});

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // consumer utils used by two processes
  //
  using ConsumerCounters = std::map<std::string, int>;
  auto inputChecker = [](InputRecord& inputs, ConsumerCounters& counters) {
    size_t nSequencePayloads = 0;
    size_t expectedPayloads = 0;
    size_t iteration = 0;
    ConsumerCounters active;
    for (auto const& ref : InputRecordWalker(inputs)) {
      if (!inputs.isValid(ref.spec->binding)) {
        continue;
      }
      auto const* dh = DataRefUtils::getHeader<DataHeader*>(ref);
      ASSERT_ERROR(dh != nullptr)
      if (!dh) {
        continue;
      }
      active[ref.spec->binding] = 1;
      if (ref.spec->binding == "sequencein") {
        auto const* sd = DataRefUtils::getHeader<test::SequenceDesc*>(ref);
        ASSERT_ERROR(sd != nullptr);
        if (!sd) {
          continue;
        }
        iteration = sd->iteration;
        if (expectedPayloads == 0) {
          expectedPayloads = sd->nPayloads;
        } else {
          ASSERT_ERROR(expectedPayloads == sd->nPayloads);
        }
        ASSERT_ERROR(*reinterpret_cast<size_t const*>(ref.payload) == sd->initialValue + nSequencePayloads);
        ++nSequencePayloads;
      }
      //LOG(info) << "input " << ref.spec->binding << " has data {" << dh->dataOrigin.as<std::string>() << "/" << dh->dataDescription.as<std::string>() << "/" << dh->subSpecification << "}: " << *reinterpret_cast<size_t const*>(ref.payload);
    }
    for (auto const& [channel, count] : active) {
      ++counters[channel];
    }
  };

  auto createCounters = [](RawDeviceService& rds) -> std::shared_ptr<ConsumerCounters> {
    auto counters = std::make_shared<ConsumerCounters>();
    ConsumerCounters& c = *counters;
    for (auto const& channelSpec : rds.spec().inputChannels) {
      // we would need the input spec here, while in the device spec we have the attributes
      // of the FairMQ Channels
      //(*counters)[channelSpec.name] = 0;
    }
    return counters;
  };

  auto checkCounters = [nRolls = attributes->nRolls](std::shared_ptr<ConsumerCounters> const& counters) -> bool {
    bool sane = true;
    for (auto const& [channel, count] : *counters) {
      if (count != nRolls) {
        LOG(fatal) << "inconsistent event count on input '" << channel << "': " << count << ", expected " << nRolls;
        sane = false;
      }
    }
    return sane;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // the consumer process connects to the producer
  //
  auto consumerInit = [createCounters, checkCounters, inputChecker](RawDeviceService& rds, CallbackService& callbacks) {
    auto counters = createCounters(rds);
    callbacks.set(CallbackService::Id::Stop, [counters, checkCounters]() {
      ASSERT_ERROR(checkCounters(counters));
    });
    callbacks.set(CallbackService::Id::EndOfStream, [counters, checkCounters](EndOfStreamContext& context) {
      ASSERT_ERROR(checkCounters(counters));
      context.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    });

    auto processing = [inputChecker, counters](InputRecord& inputs) {
      inputChecker(inputs, *counters);
    };

    return adaptStateless(processing);
  };

  workflow.emplace_back(DataProcessorSpec{"consumer",
                                          {InputSpec{"pairin", "TST", "PAIR", 0, Lifetime::Timeframe},
                                           InputSpec{"sequencein", "TST", "SEQUENCE", 0, Lifetime::Timeframe},
                                           InputSpec{"dpldefault", "TST", "ALLOCATOR", 0, Lifetime::Timeframe}},
                                          {},
                                          AlgorithmSpec{adaptStateful(consumerInit)}});

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // spectator process which should get the forwarded data
  //
  workflow.emplace_back(DataProcessorSpec{"spectator",
                                          {InputSpec{"pairin", "TST", "PAIR", 0, Lifetime::Timeframe},
                                           InputSpec{"sequencein", "TST", "SEQUENCE", 0, Lifetime::Timeframe},
                                           InputSpec{"dpldefault", "TST", "ALLOCATOR", 0, Lifetime::Timeframe}},
                                          {},
                                          AlgorithmSpec{adaptStateful(consumerInit)}});

  return workflow;
}
