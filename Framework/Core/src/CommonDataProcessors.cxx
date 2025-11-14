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
#include "Framework/CommonDataProcessors.h"

#include "Framework/AlgorithmSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataProcessingStats.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/InitContext.h"
#include "Framework/InputSpec.h"
#include "Framework/RawDeviceService.h"
#include "Framework/TimesliceIndex.h"
#include "Framework/Variant.h"
#include "Framework/ChannelSpec.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/RuntimeError.h"
#include "Framework/RateLimiter.h"
#include "Framework/PluginManager.h"
#include "Framework/Signpost.h"
#include <Monitoring/Monitoring.h>

#include <fairmq/Device.h>
#include <uv.h>
#include <fstream>
#include <functional>
#include <memory>
#include <string>

using namespace o2::framework::data_matcher;

// Special log to track callbacks we know about
O2_DECLARE_DYNAMIC_LOG(callbacks);
O2_DECLARE_DYNAMIC_LOG(rate_limiting);
O2_DECLARE_DYNAMIC_LOG(quota);

namespace o2::framework
{

DataProcessorSpec
  CommonDataProcessors::getGlobalFileSink(std::vector<InputSpec> const& danglingOutputInputs,
                                          std::vector<InputSpec>& unmatched)
{
  auto writerFunction = [danglingOutputInputs](InitContext& ic) -> std::function<void(ProcessingContext&)> {
    auto filename = ic.options().get<std::string>("outfile");
    auto keepString = ic.options().get<std::string>("keep");

    if (filename.empty()) {
      throw runtime_error("output file missing");
    }

    bool hasOutputsToWrite = false;
    auto [variables, outputMatcher] = DataDescriptorQueryBuilder::buildFromKeepConfig(keepString);
    VariableContext context;
    for (auto& spec : danglingOutputInputs) {
      auto concrete = DataSpecUtils::asConcreteDataTypeMatcher(spec);
      if (outputMatcher->match(concrete, context)) {
        hasOutputsToWrite = true;
      }
    }
    if (hasOutputsToWrite == false) {
      return [](ProcessingContext&) mutable -> void {
        static bool once = false;
        if (!once) {
          LOG(debug) << "No dangling output to be dumped.";
          once = true;
        }
      };
    }
    auto output = std::make_shared<std::ofstream>(filename.c_str(), std::ios_base::binary);
    return [output, matcher = outputMatcher](ProcessingContext& pc) mutable -> void {
      VariableContext matchingContext;
      LOG(debug) << "processing data set with " << pc.inputs().size() << " entries";
      for (const auto& entry : pc.inputs()) {
        LOG(debug) << "  " << *(entry.spec);
        auto header = DataRefUtils::getHeader<header::DataHeader*>(entry);
        auto dataProcessingHeader = DataRefUtils::getHeader<DataProcessingHeader*>(entry);
        if (matcher->match(*header, matchingContext) == false) {
          continue;
        }
        output->write(reinterpret_cast<char const*>(header), sizeof(header::DataHeader));
        output->write(reinterpret_cast<char const*>(dataProcessingHeader), sizeof(DataProcessingHeader));
        output->write(entry.payload, o2::framework::DataRefUtils::getPayloadSize(entry));
        LOG(debug) << "wrote data, size " << o2::framework::DataRefUtils::getPayloadSize(entry);
      }
    };
  };

  std::vector<InputSpec> validBinaryInputs;
  auto onlyTimeframe = [](InputSpec const& input) {
    return (DataSpecUtils::partialMatch(input, o2::header::DataOrigin("TFN")) == false) &&
           input.lifetime == Lifetime::Timeframe;
  };

  auto noTimeframe = [](InputSpec const& input) {
    return (DataSpecUtils::partialMatch(input, o2::header::DataOrigin("TFN")) == true) ||
           input.lifetime != Lifetime::Timeframe;
  };

  std::copy_if(danglingOutputInputs.begin(), danglingOutputInputs.end(),
               std::back_inserter(validBinaryInputs), onlyTimeframe);
  std::copy_if(danglingOutputInputs.begin(), danglingOutputInputs.end(),
               std::back_inserter(unmatched), noTimeframe);

  DataProcessorSpec spec{
    "internal-dpl-injected-global-binary-file-sink",
    validBinaryInputs,
    Outputs{},
    AlgorithmSpec(writerFunction),
    {{"outfile", VariantType::String, "dpl-out.bin", {"Name of the output file"}},
     {"keep", VariantType::String, "", {"Comma separated list of ORIGIN/DESCRIPTION/SUBSPECIFICATION to save in outfile"}}}};

  return spec;
}

DataProcessorSpec CommonDataProcessors::getGlobalFairMQSink(std::vector<InputSpec> const& danglingOutputInputs)
{

  // we build the default channel configuration from the binding of the first input
  // in order to have more than one we would need to possibility to have support for
  // vectored options
  // use the OutputChannelSpec as a tool to create the default configuration for the out-of-band channel
  OutputChannelSpec externalChannelSpec;
  externalChannelSpec.name = "downstream";
  externalChannelSpec.type = ChannelType::Push;
  externalChannelSpec.method = ChannelMethod::Bind;
  externalChannelSpec.hostname = "localhost";
  externalChannelSpec.port = 0;
  externalChannelSpec.listeners = 0;
  // in principle, protocol and transport are two different things but fur simplicity
  // we use ipc when shared memory is selected and the normal tcp url whith zeromq,
  // this is for building the default configuration which can be simply changed from the
  // command line
  externalChannelSpec.protocol = ChannelProtocol::IPC;
  std::string defaultChannelConfig = formatExternalChannelConfiguration(externalChannelSpec);
  // at some point the formatting tool might add the transport as well so we have to check
  return specifyFairMQDeviceOutputProxy("internal-dpl-injected-output-proxy", danglingOutputInputs, defaultChannelConfig.c_str());
}

void retryMetricCallback(uv_async_t* async)
{
  O2_SIGNPOST_ID_FROM_POINTER(cid, callbacks, async);
  O2_SIGNPOST_EVENT_EMIT(callbacks, cid, "rate-limiting", "Attempting again propagating rate-limiting information.");

  // Check if this is a source device
  static size_t lastTimeslice = -1;
  auto* services = (ServiceRegistryRef*)async->data;
  auto& timesliceIndex = services->get<TimesliceIndex>();
  auto* device = services->get<RawDeviceService>().device();
  auto channel = device->GetChannels().find("metric-feedback");
  auto oldestPossingTimeslice = timesliceIndex.getOldestPossibleOutput().timeslice.value;
  if (channel == device->GetChannels().end()) {
    O2_SIGNPOST_EVENT_EMIT(callbacks, cid, "rate-limiting", "Could not find metric-feedback channel.");
    return;
  }
  fair::mq::MessagePtr payload(device->NewMessage());
  payload->Rebuild(&oldestPossingTimeslice, sizeof(int64_t), nullptr, nullptr);
  auto consumed = oldestPossingTimeslice;

  size_t start = uv_hrtime();
  int64_t result = channel->second[0].Send(payload, 100);
  size_t stop = uv_hrtime();
  // If the sending worked, we do not retry.
  if (result <= 0) {
    // Forcefully slow down in case FairMQ returns earlier than expected...
    int64_t ellapsed = (stop - start) / 1000000;
    if (ellapsed < 100) {
      O2_SIGNPOST_EVENT_EMIT(callbacks, cid, "rate-limiting",
                             "FairMQ returned %llu earlier than expected. Sleeping %llu ms more before, retrying.",
                             result, ellapsed);
      uv_sleep(100 - ellapsed);
    } else {
      O2_SIGNPOST_EVENT_EMIT(callbacks, cid, "rate-limiting",
                             "FairMQ returned %llu, unable to send last consumed timeslice to source for %llu ms, retrying.", result, ellapsed);
    }
    // If the sending did not work, we keep trying until it actually works.
    // This will schedule other tasks in the queue, so the processing of the
    // data will still happen.
    uv_async_send(async);
  } else {
    O2_SIGNPOST_EVENT_EMIT(callbacks, cid, "rate-limiting", "Send %llu bytes, Last timeslice now set to %zu.", result, consumed);
    lastTimeslice = consumed;
  }
}

DataProcessorSpec CommonDataProcessors::getDummySink(std::vector<InputSpec> const& danglingOutputInputs, std::string rateLimitingChannelConfig)
{
  return DataProcessorSpec{
    .name = "internal-dpl-injected-dummy-sink",
    .inputs = danglingOutputInputs,
    .algorithm = AlgorithmSpec{adaptStateful([](CallbackService& callbacks, DeviceState& deviceState, InitContext& ic) {
      static uv_async_t async;
      // The callback will only have access to the
      async.data = new ServiceRegistryRef{ic.services()};
      uv_async_init(deviceState.loop, &async, retryMetricCallback);
      auto domainInfoUpdated = [](ServiceRegistryRef services, size_t timeslice, ChannelIndex channelIndex) {
        LOGP(debug, "Domain info updated with timeslice {}", timeslice);
        retryMetricCallback(&async);
        auto& timesliceIndex = services.get<TimesliceIndex>();
        auto oldestPossingTimeslice = timesliceIndex.getOldestPossibleOutput().timeslice.value;
        auto& stats = services.get<DataProcessingStats>();
        stats.updateStats({(int)ProcessingStatsId::CONSUMED_TIMEFRAMES, DataProcessingStats::Op::Set, (int64_t)oldestPossingTimeslice});
        stats.updateStats({(int)ProcessingStatsId::TIMESLICE_NUMBER_DONE, DataProcessingStats::Op::Set, (int64_t)oldestPossingTimeslice});
        stats.processCommandQueue();
      };
      callbacks.set<CallbackService::Id::DomainInfoUpdated>(domainInfoUpdated);

      return adaptStateless([]() {
      });
    })},
    .options = !rateLimitingChannelConfig.empty() ? std::vector<ConfigParamSpec>{{"channel-config", VariantType::String, // raw input channel
                                                                                  rateLimitingChannelConfig,
                                                                                  {"Out-of-band channel config"}}}
                                                  : std::vector<ConfigParamSpec>(),
    .labels = {{"resilient"}}};
}

// For the cases were the driver is guaranteed to be there (e.g. in analysis) we can use a
// more sophisticated controller which can get offers for timeslices so that we can rate limit
// across multiple input devices and rate limit shared memory usage without race conditions
DataProcessorSpec CommonDataProcessors::getScheduledDummySink(std::vector<InputSpec> const& danglingOutputInputs)
{
  return DataProcessorSpec{
    .name = "internal-dpl-injected-dummy-sink",
    .inputs = danglingOutputInputs,
    .algorithm = AlgorithmSpec{adaptStateful([](CallbackService& callbacks, DeviceState& deviceState, InitContext& ic) {
      // We update the number of consumed timeframes based on the oldestPossingTimeslice
      // this information will be aggregated in the driver which will then decide wether or not a new offer for
      // a timeslice should be done and to which device
      auto domainInfoUpdated = [](ServiceRegistryRef services, size_t timeslice, ChannelIndex channelIndex) {
        LOGP(debug, "Domain info updated with timeslice {}", timeslice);
        auto& timesliceIndex = services.get<TimesliceIndex>();
        auto oldestPossingTimeslice = timesliceIndex.getOldestPossibleOutput().timeslice.value;
        auto& stats = services.get<DataProcessingStats>();
        O2_SIGNPOST_ID_GENERATE(sid, rate_limiting);
        O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "run", "Consumed timeframes (domain info updated) to be set to %zu.", oldestPossingTimeslice);
        stats.updateStats({(int)ProcessingStatsId::CONSUMED_TIMEFRAMES, DataProcessingStats::Op::Set, (int64_t)oldestPossingTimeslice});
        stats.updateStats({(int)ProcessingStatsId::TIMESLICE_NUMBER_DONE, DataProcessingStats::Op::Set, (int64_t)oldestPossingTimeslice});
        stats.processCommandQueue();
      };
      callbacks.set<CallbackService::Id::DomainInfoUpdated>(domainInfoUpdated);

      return adaptStateless([](DataProcessingStats& stats, TimesliceIndex& timesliceIndex) {
        O2_SIGNPOST_ID_GENERATE(sid, rate_limiting);
        auto oldestPossingTimeslice = timesliceIndex.getOldestPossibleOutput().timeslice.value;
        O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "run", "Consumed timeframes (processing) to be set to %zu.", oldestPossingTimeslice);
        stats.updateStats({(int)ProcessingStatsId::CONSUMED_TIMEFRAMES, DataProcessingStats::Op::Set, (int64_t)oldestPossingTimeslice});
        stats.updateStats({(int)ProcessingStatsId::TIMESLICE_NUMBER_DONE, DataProcessingStats::Op::Set, (int64_t)oldestPossingTimeslice});
        stats.processCommandQueue();
      });
    })},
    .labels = {{"resilient"}}};
}

AlgorithmSpec CommonDataProcessors::wrapWithRateLimiting(AlgorithmSpec spec)
{
  return PluginManager::wrapAlgorithm(spec, [](AlgorithmSpec::ProcessCallback& original, ProcessingContext& pcx) -> void {
    auto& raw = pcx.services().get<RawDeviceService>();
    static RateLimiter limiter;
    O2_SIGNPOST_ID_FROM_POINTER(sid, rate_limiting, &pcx);
    auto limit = std::stoi(raw.device()->fConfig->GetValue<std::string>("timeframes-rate-limit"));
    O2_SIGNPOST_EVENT_EMIT_DETAIL(rate_limiting, sid, "rate limiting callback",
                                  "Rate limiting to %d timeframes in flight", limit);
    limiter.check(pcx, limit, 2000);
    O2_SIGNPOST_EVENT_EMIT_DETAIL(rate_limiting, sid, "rate limiting callback",
                                  "Rate limiting passed. Invoking old callback.");
    original(pcx);
    O2_SIGNPOST_EVENT_EMIT_DETAIL(rate_limiting, sid, "rate limiting callback",
                                  "Rate limited callback done.");
  });
}

// The wrapped algorithm consumes 1 timeslice every time is invoked
AlgorithmSpec CommonDataProcessors::wrapWithTimesliceConsumption(AlgorithmSpec spec)
{
  return PluginManager::wrapAlgorithm(spec, [](AlgorithmSpec::ProcessCallback& original, ProcessingContext& pcx) -> void {
    original(pcx);

    auto disposeResources = [](int taskId,
                               std::array<ComputingQuotaOffer, 32>& offers,
                               ComputingQuotaStats& stats,
                               std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats&)> accountDisposed) {
      ComputingQuotaOffer disposed;
      disposed.sharedMemory = 0;
      // When invoked, we have processed one timeslice by construction.
      int64_t timeslicesProcessed = 1;
      for (auto& offer : offers) {
        if (offer.user != taskId) {
          continue;
        }
        int64_t toRemove = std::min((int64_t)timeslicesProcessed, offer.timeslices);
        offer.timeslices -= toRemove;
        timeslicesProcessed -= toRemove;
        disposed.timeslices += toRemove;
        if (timeslicesProcessed <= 0) {
          break;
        }
      }
      return accountDisposed(disposed, stats);
    };
    pcx.services().get<DeviceState>().offerConsumers.emplace_back(disposeResources);
  });
}

} // namespace o2::framework
