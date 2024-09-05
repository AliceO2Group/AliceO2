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
#include "Framework/DataOutputDirector.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataProcessingStats.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/TableBuilder.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/InitContext.h"
#include "Framework/InputSpec.h"
#include "Framework/Logger.h"
#include "Framework/OutputSpec.h"
#include "Framework/RawDeviceService.h"
#include "Framework/TimesliceIndex.h"
#include "Framework/Variant.h"
#include "../../../Algorithm/include/Algorithm/HeaderStack.h"
#include "Framework/OutputObjHeader.h"
#include "Framework/StringHelpers.h"
#include "Framework/ChannelSpec.h"
#include "Framework/ChannelSpecHelpers.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/RuntimeError.h"
#include "Framework/RateLimiter.h"
#include "Framework/PluginManager.h"
#include "Framework/DeviceSpec.h"
#include "WorkflowHelpers.h"
#include <Monitoring/Monitoring.h>

#include <fairmq/Device.h>
#include <chrono>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <thread>

using namespace o2::framework::data_matcher;

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
  static size_t lastTimeslice = -1;
  auto* services = (ServiceRegistryRef*)async->data;
  auto& timesliceIndex = services->get<TimesliceIndex>();
  auto* device = services->get<RawDeviceService>().device();
  auto channel = device->GetChannels().find("metric-feedback");
  auto oldestPossingTimeslice = timesliceIndex.getOldestPossibleOutput().timeslice.value;
  if (channel == device->GetChannels().end()) {
    return;
  }
  fair::mq::MessagePtr payload(device->NewMessage());
  payload->Rebuild(&oldestPossingTimeslice, sizeof(int64_t), nullptr, nullptr);
  auto consumed = oldestPossingTimeslice;

  int64_t result = channel->second[0].Send(payload, 100);
  // If the sending worked, we do not retry.
  if (result != 0) {
    // If the sending did not work, we keep trying until it actually works.
    // This will schedule other tasks in the queue, so the processing of the
    // data will still happen.
    uv_async_send(async);
  } else {
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

AlgorithmSpec CommonDataProcessors::wrapWithRateLimiting(AlgorithmSpec spec)
{
  return PluginManager::wrapAlgorithm(spec, [](AlgorithmSpec::ProcessCallback& original, ProcessingContext& pcx) -> void {
    auto& raw = pcx.services().get<RawDeviceService>();
    static RateLimiter limiter;
    auto limit = std::stoi(raw.device()->fConfig->GetValue<std::string>("timeframes-rate-limit"));
    LOG(detail) << "Rate limiting to " << limit << " timeframes in flight";
    limiter.check(pcx, limit, 2000);
    LOG(detail) << "Rate limiting passed. Invoking old callback";
    original(pcx);
    LOG(detail) << "Rate limited callback done";
  });
}

} // namespace o2::framework
