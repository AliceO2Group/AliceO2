// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/CommonMessageBackends.h"
#include "Framework/MessageContext.h"
#include "Framework/ArrowContext.h"
#include "Framework/StringContext.h"
#include "Framework/RawBufferContext.h"
#include "Framework/DataProcessor.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/Tracing.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceInfo.h"

#include <Monitoring/Monitoring.h>
#include <Headers/DataHeader.h>

#include <options/FairMQProgOptions.h>

#include <uv.h>
#include <boost/program_options/variables_map.hpp>
#include <csignal>

namespace o2::framework
{

struct EndOfStreamContext;
struct ProcessingContext;

namespace
{
template <typename T>
struct CommonMessageBackendsHelpers {
  static ServiceInit createCallback()
  {
    return [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) {
      auto& device = services.get<RawDeviceService>();
      return ServiceHandle{TypeIdHelpers::uniqueId<T>(), new T(FairMQDeviceProxy{device.device()})};
    };
  }

  static ServiceProcessingCallback sendCallback()
  {
    return [](ProcessingContext& ctx, void* service) {
      ZoneScopedN("send message callback");
      T* context = reinterpret_cast<T*>(service);
      auto& device = ctx.services().get<RawDeviceService>();
      DataProcessor::doSend(*device.device(), *context, ctx.services());
    };
  }

  static ServiceProcessingCallback clearContext()
  {
    return [](ProcessingContext&, void* service) {
      T* context = reinterpret_cast<T*>(service);
      context->clear();
    };
  }

  static ServiceEOSCallback clearContextEOS()
  {
    return [](EndOfStreamContext&, void* service) {
      T* context = reinterpret_cast<T*>(service);
      context->clear();
    };
  }

  static ServiceEOSCallback sendCallbackEOS()
  {
    return [](EndOfStreamContext& ctx, void* service) {
      T* context = reinterpret_cast<T*>(service);
      auto& device = ctx.services().get<RawDeviceService>();
      DataProcessor::doSend(*device.device(), *context, ctx.services());
    };
  }
};
} // namespace

enum struct RateLimitingState {
  UNKNOWN = 0,                   // No information received yet.
  STARTED = 1,                   // Information received, new timeframe not requested.
  CHANGED = 2,                   // Information received, new timeframe requested but not yet accounted.
  BELOW_LIMIT = 3,               // New metric received, we are below limit.
  NEXT_ITERATION_FROM_BELOW = 4, // Iteration when previously in BELOW_LIMIT.
  ABOVE_LIMIT = 5,               // New metric received, we are above limit.
};

struct RateLimitConfig {
  int64_t maxMemory = 0;
};

static int64_t memLimit = 0;

/// Service for common handling of rate limiting
o2::framework::ServiceSpec CommonMessageBackends::rateLimitingSpec()
{
  return ServiceSpec{"aod-rate-limiting",
                     [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) {
                       return ServiceHandle{TypeIdHelpers::uniqueId<RateLimitConfig>(), new RateLimitConfig{}};
                     },
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     [](ServiceRegistry& registry, boost::program_options::variables_map const& vm) {
                       if (!vm["aod-memory-rate-limit"].defaulted()) {
                         memLimit = std::stoll(vm["aod-memory-rate-limit"].as<std::string const>());
                         // registry.registerService(ServiceRegistryHelpers::handleForService<RateLimitConfig>(new RateLimitConfig{memLimit}));
                       }
                     },
                     nullptr,
                     nullptr,
                     nullptr,
                     ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonMessageBackends::arrowBackendSpec()
{
  using o2::monitoring::Metric;
  using o2::monitoring::Monitoring;
  using o2::monitoring::tags::Key;
  using o2::monitoring::tags::Value;

  return ServiceSpec{"arrow-backend",
                     CommonMessageBackendsHelpers<ArrowContext>::createCallback(),
                     CommonServices::noConfiguration(),
                     CommonMessageBackendsHelpers<ArrowContext>::clearContext(),
                     CommonMessageBackendsHelpers<ArrowContext>::sendCallback(),
                     nullptr,
                     nullptr,
                     CommonMessageBackendsHelpers<ArrowContext>::clearContextEOS(),
                     CommonMessageBackendsHelpers<ArrowContext>::sendCallbackEOS(),
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     [](ServiceRegistry& registry,
                        std::vector<DeviceMetricsInfo>& allDeviceMetrics,
                        std::vector<DeviceSpec>& specs,
                        std::vector<DeviceInfo>& infos,
                        DeviceMetricsInfo& driverMetrics,
                        size_t timestamp) {
                       int64_t totalBytesCreated = 0;
                       int64_t totalBytesDestroyed = 0;
                       int64_t totalMessagesCreated = 0;
                       int64_t totalMessagesDestroyed = 0;
                       static RateLimitingState currentState = RateLimitingState::UNKNOWN;
                       static auto stateMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "rate-limit-state");
                       static auto totalBytesCreatedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-bytes-created");
                       static auto totalBytesDestroyedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-bytes-destroyed");
                       static auto totalMessagesCreatedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-messages-created");
                       static auto totalMessagesDestroyedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-messages-destroyed");
                       static auto totalBytesDeltaMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "arrow-bytes-delta");
                       static auto totalSignalsMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "aod-reader-signals");
                       static auto remainingBytes = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "aod-remaining-bytes");

                       bool changed = false;
                       bool hasMetrics = false;
                       // Find  the last timestamp when we signaled.
                       size_t signalIndex = DeviceMetricsHelper::metricIdxByName("aod-reader-signals", driverMetrics);
                       size_t lastSignalTimestamp = 0;
                       if (signalIndex < driverMetrics.metrics.size()) {
                         MetricInfo info = driverMetrics.metrics.at(signalIndex);
                         lastSignalTimestamp = driverMetrics.timestamps[signalIndex][info.pos - 1];
                       }

                       size_t lastCreatedBytesTimestamp = 0;
                       size_t lastDestroyedBytesTimestamp = 0;
                       for (auto& deviceMetrics : allDeviceMetrics) {
                         {
                           size_t index = DeviceMetricsHelper::metricIdxByName("arrow-bytes-created", deviceMetrics);
                           if (index < deviceMetrics.metrics.size()) {
                             hasMetrics = true;
                             changed |= deviceMetrics.changed.at(index);
                             MetricInfo info = deviceMetrics.metrics.at(index);
                             auto& data = deviceMetrics.uint64Metrics.at(info.storeIdx);
                             totalBytesCreated += (int64_t)data.at((info.pos - 1) % data.size());
                             lastCreatedBytesTimestamp = deviceMetrics.timestamps[index][info.pos - 1];
                           }
                         }
                         {
                           size_t index = DeviceMetricsHelper::metricIdxByName("arrow-bytes-destroyed", deviceMetrics);
                           if (index < deviceMetrics.metrics.size()) {
                             hasMetrics = true;
                             changed |= deviceMetrics.changed.at(index);
                             MetricInfo info = deviceMetrics.metrics.at(index);
                             auto& data = deviceMetrics.uint64Metrics.at(info.storeIdx);
                             totalBytesDestroyed += (int64_t)data.at((info.pos - 1) % data.size());
                             lastDestroyedBytesTimestamp = deviceMetrics.timestamps[index][info.pos - 1];
                           }
                         }
                         {
                           size_t index = DeviceMetricsHelper::metricIdxByName("arrow-messages-created", deviceMetrics);
                           if (index < deviceMetrics.metrics.size()) {
                             MetricInfo info = deviceMetrics.metrics.at(index);
                             auto& data = deviceMetrics.uint64Metrics.at(info.storeIdx);
                             totalMessagesCreated += (int64_t)data.at((info.pos - 1) % data.size());
                           }
                         }
                         {
                           size_t index = DeviceMetricsHelper::metricIdxByName("arrow-messages-destroyed", deviceMetrics);
                           if (index < deviceMetrics.metrics.size()) {
                             MetricInfo info = deviceMetrics.metrics.at(index);
                             auto& data = deviceMetrics.uint64Metrics.at(info.storeIdx);
                             totalMessagesDestroyed += (int64_t)data.at((info.pos - 1) % data.size());
                           }
                         }
                       }
                       if (changed) {
                         totalBytesCreatedMetric(driverMetrics, totalBytesCreated, timestamp);
                         totalBytesDestroyedMetric(driverMetrics, totalBytesDestroyed, timestamp);
                         totalMessagesCreatedMetric(driverMetrics, totalMessagesCreated, timestamp);
                         totalMessagesDestroyedMetric(driverMetrics, totalMessagesDestroyed, timestamp);
                         totalBytesDeltaMetric(driverMetrics, totalBytesCreated - totalBytesDestroyed, timestamp);
                       }
                       bool done = false;
                       static int stateTransitions = 0;
                       while (!done) {
                         stateMetric(driverMetrics, (uint64_t)(currentState), stateTransitions++);
                         switch (currentState) {
                           case RateLimitingState::UNKNOWN: {
                             for (auto& deviceMetrics : allDeviceMetrics) {
                               size_t index = DeviceMetricsHelper::metricIdxByName("arrow-bytes-created", deviceMetrics);
                               if (index < deviceMetrics.metrics.size()) {
                                 currentState = RateLimitingState::STARTED;
                               }
                             }
                             done = true;
                           } break;
                           case RateLimitingState::STARTED: {
                             for (size_t di = 0; di < specs.size(); ++di) {
                               if (specs[di].name == "internal-dpl-aod-reader") {
                                 if (di < infos.size()) {
                                   kill(infos[di].pid, SIGUSR1);
                                   totalSignalsMetric(driverMetrics, 1, timestamp);
                                 }
                               }
                             }
                             changed = false;
                             currentState = RateLimitingState::CHANGED;
                           } break;
                           case RateLimitingState::CHANGED: {
                             remainingBytes(driverMetrics, totalBytesCreated <= (totalBytesDestroyed + memLimit) ? (totalBytesDestroyed + memLimit) - totalBytesCreated : 0, timestamp);
                             if (totalBytesCreated <= (totalBytesDestroyed + memLimit)) {
                               currentState = RateLimitingState::BELOW_LIMIT;
                             } else {
                               currentState = RateLimitingState::ABOVE_LIMIT;
                             }
                             changed = false;
                           } break;
                           case RateLimitingState::BELOW_LIMIT: {
                             for (size_t di = 0; di < specs.size(); ++di) {
                               if (specs[di].name == "internal-dpl-aod-reader") {
                                 if (di < infos.size()) {
                                   kill(infos[di].pid, SIGUSR1);
                                   totalSignalsMetric(driverMetrics, 1, timestamp);
                                 }
                               }
                             }
                             changed = false;
                             currentState = RateLimitingState::NEXT_ITERATION_FROM_BELOW;
                           } break;
                           case RateLimitingState::NEXT_ITERATION_FROM_BELOW: {
                             if (!changed) {
                               done = true;
                             } else {
                               currentState = RateLimitingState::CHANGED;
                             }
                           } break;
                           case RateLimitingState::ABOVE_LIMIT: {
                             if (!changed) {
                               done = true;
                             } else if (totalBytesCreated > (totalBytesDestroyed + memLimit / 3)) {
                               done = true;
                             } else {
                               currentState = RateLimitingState::CHANGED;
                             }
                           };
                         };
                       }
                     },
                     [](ProcessingContext& ctx, void* service) {
                       using DataHeader = o2::header::DataHeader;
                       ArrowContext* arrow = reinterpret_cast<ArrowContext*>(service);
                       auto totalBytes = 0;
                       auto totalMessages = 0;
                       for (auto& input : ctx.inputs()) {
                         if (input.header == nullptr) {
                           continue;
                         }
                         auto dh = o2::header::get<DataHeader*>(input.header);
                         if (dh->serialization != o2::header::gSerializationMethodArrow) {
                           continue;
                         }
                         auto dph = o2::header::get<DataProcessingHeader*>(input.header);
                         bool forwarded = false;
                         for (auto const& forward : ctx.services().get<DeviceSpec const>().forwards) {
                           if (DataSpecUtils::match(forward.matcher, dh->dataOrigin, dh->dataDescription, dh->subSpecification)) {
                             forwarded = true;
                             break;
                           }
                         }
                         if (forwarded) {
                           continue;
                         }
                         totalBytes += dh->payloadSize;
                         totalMessages += 1;
                       }
                       arrow->updateBytesDestroyed(totalBytes);
                       arrow->updateMessagesDestroyed(totalMessages);
                       auto& monitoring = ctx.services().get<Monitoring>();
                       monitoring.send(Metric{(uint64_t)arrow->bytesDestroyed(), "arrow-bytes-destroyed"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
                       monitoring.send(Metric{(uint64_t)arrow->messagesDestroyed(), "arrow-messages-destroyed"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
                       monitoring.flushBuffer();
                     },
                     ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonMessageBackends::fairMQBackendSpec()
{
  return ServiceSpec{"fairmq-backend",
                     [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
                       auto& device = services.get<RawDeviceService>();
                       auto context = new MessageContext(FairMQDeviceProxy{device.device()});
                       auto& spec = services.get<DeviceSpec const>();

                       auto dispatcher = [&device](FairMQParts&& parts, std::string const& channel, unsigned int index) {
                         DataProcessor::doSend(*device.device(), std::move(parts), channel.c_str(), index);
                       };

                       auto matcher = [policy = spec.dispatchPolicy](o2::header::DataHeader const& header) {
                         if (policy.triggerMatcher == nullptr) {
                           return true;
                         }
                         return policy.triggerMatcher(Output{header});
                       };

                       if (spec.dispatchPolicy.action == DispatchPolicy::DispatchOp::WhenReady) {
                         context->init(DispatchControl{dispatcher, matcher});
                       }
                       return ServiceHandle{TypeIdHelpers::uniqueId<MessageContext>(), context};
                     },
                     CommonServices::noConfiguration(),
                     CommonMessageBackendsHelpers<MessageContext>::clearContext(),
                     CommonMessageBackendsHelpers<MessageContext>::sendCallback(),
                     nullptr,
                     nullptr,
                     CommonMessageBackendsHelpers<MessageContext>::clearContextEOS(),
                     CommonMessageBackendsHelpers<MessageContext>::sendCallbackEOS(),
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonMessageBackends::stringBackendSpec()
{
  return ServiceSpec{"string-backend",
                     CommonMessageBackendsHelpers<StringContext>::createCallback(),
                     CommonServices::noConfiguration(),
                     CommonMessageBackendsHelpers<StringContext>::clearContext(),
                     CommonMessageBackendsHelpers<StringContext>::sendCallback(),
                     nullptr,
                     nullptr,
                     CommonMessageBackendsHelpers<StringContext>::clearContextEOS(),
                     CommonMessageBackendsHelpers<StringContext>::sendCallbackEOS(),
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonMessageBackends::rawBufferBackendSpec()
{
  return ServiceSpec{"raw-backend",
                     CommonMessageBackendsHelpers<RawBufferContext>::createCallback(),
                     CommonServices::noConfiguration(),
                     CommonMessageBackendsHelpers<RawBufferContext>::clearContext(),
                     CommonMessageBackendsHelpers<RawBufferContext>::sendCallback(),
                     nullptr,
                     nullptr,
                     CommonMessageBackendsHelpers<RawBufferContext>::clearContextEOS(),
                     CommonMessageBackendsHelpers<RawBufferContext>::sendCallbackEOS(),
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     nullptr,
                     ServiceKind::Serial};
}

} // namespace o2::framework
