// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "ArrowSupport.h"
#include "Framework/ArrowContext.h"
#include "Framework/DataProcessor.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/DeviceSpec.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/Tracing.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceMetricsHelper.h"
#include "Framework/DeviceInfo.h"

#include "CommonMessageBackendsHelpers.h"
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

enum struct RateLimitingState {
  UNKNOWN = 0,                   // No information received yet.
  STARTED = 1,                   // Information received, new timeframe not requested.
  CHANGED = 2,                   // Information received, new timeframe requested but not yet accounted.
  BELOW_LIMIT = 3,               // New metric received, we are below limit.
  NEXT_ITERATION_FROM_BELOW = 4, // Iteration when previously in BELOW_LIMIT.
  ABOVE_LIMIT = 5,               // New metric received, we are above limit.
  EMPTY = 6,                     //
};

struct RateLimitConfig {
  int64_t maxMemory = 0;
};

static int64_t memLimit = 0;

struct MetricIndices {
  size_t arrowBytesCreated = 0;
  size_t arrowBytesDestroyed = 0;
  size_t arrowMessagesCreated = 0;
  size_t arrowMessagesDestroyed = 0;
};

/// Service for common handling of rate limiting
o2::framework::ServiceSpec ArrowSupport::rateLimitingSpec()
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
                     nullptr,
                     nullptr,
                     ServiceKind::Serial};
}

std::vector<MetricIndices> createDefaultIndices(std::vector<DeviceMetricsInfo>& allDevicesMetrics)
{
  std::vector<MetricIndices> results;

  for (auto& info : allDevicesMetrics) {
    MetricIndices indices;
    indices.arrowBytesCreated = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-bytes-created");
    indices.arrowBytesDestroyed = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-bytes-destroyed");
    indices.arrowMessagesCreated = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-messages-created");
    indices.arrowMessagesDestroyed = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-messages-destroyed");
    results.push_back(indices);
  }
  return results;
}

o2::framework::ServiceSpec ArrowSupport::arrowBackendSpec()
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
                       static auto totalBytesDeltaMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "arrow-bytes-delta");
                       static auto totalSignalsMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "aod-reader-signals");
                       static auto skippedSignalsMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "aod-skipped-signals");
                       static auto remainingBytes = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "aod-remaining-bytes");

                       bool changed = false;
                       bool hasMetrics = false;
                       // Find  the last timestamp when we signaled.
                       static size_t signalIndex = DeviceMetricsHelper::metricIdxByName("aod-reader-signals", driverMetrics);
                       size_t lastSignalTimestamp = 0;
                       if (signalIndex < driverMetrics.metrics.size()) {
                         MetricInfo& info = driverMetrics.metrics.at(signalIndex);
                         if (info.filledMetrics) {
                           lastSignalTimestamp = driverMetrics.timestamps.at(signalIndex).at((info.pos - 1) % driverMetrics.timestamps.at(signalIndex).size());
                         }
                       }

                       size_t lastTimestamp = 0;
                       size_t firstTimestamp = -1;
                       size_t lastDecision = 0;
                       static std::vector<MetricIndices> allIndices = createDefaultIndices(allDeviceMetrics);
                       for (size_t mi = 0; mi < allDeviceMetrics.size(); ++mi) {
                         auto& deviceMetrics = allDeviceMetrics[mi];
                         auto& indices = allIndices[mi];
                         {
                           size_t index = indices.arrowBytesCreated;
                           if (index < deviceMetrics.metrics.size()) {
                             hasMetrics = true;
                             changed |= deviceMetrics.changed.at(index);
                             MetricInfo info = deviceMetrics.metrics.at(index);
                             auto& data = deviceMetrics.uint64Metrics.at(info.storeIdx);
                             totalBytesCreated += (int64_t)data.at((info.pos - 1) % data.size());
                             lastTimestamp = std::max(lastTimestamp, deviceMetrics.timestamps[index][info.pos - 1]);
                             firstTimestamp = std::min(lastTimestamp, firstTimestamp);
                           }
                         }
                         {
                           size_t index = indices.arrowBytesDestroyed;
                           if (index < deviceMetrics.metrics.size()) {
                             hasMetrics = true;
                             changed |= deviceMetrics.changed.at(index);
                             MetricInfo info = deviceMetrics.metrics.at(index);
                             auto& data = deviceMetrics.uint64Metrics.at(info.storeIdx);
                             totalBytesDestroyed += (int64_t)data.at((info.pos - 1) % data.size());
                             firstTimestamp = std::min(lastTimestamp, firstTimestamp);
                           }
                         }
                         {
                           size_t index = indices.arrowMessagesCreated;
                           if (index < deviceMetrics.metrics.size()) {
                             MetricInfo info = deviceMetrics.metrics.at(index);
                             auto& data = deviceMetrics.uint64Metrics.at(info.storeIdx);
                             totalMessagesCreated += (int64_t)data.at((info.pos - 1) % data.size());
                           }
                         }
                         {
                           size_t index = indices.arrowMessagesDestroyed;
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
                       static int signalsCount = 0;
                       static int skippedCount = 0;
                       static uint64_t now = 0;
                       static uint64_t lastSignal = 0;
                       now = uv_hrtime();
                       static RateLimitingState lastReportedState = RateLimitingState::UNKNOWN;
                       static uint64_t lastReportTime = 0;
                       while (!done) {
#ifndef NDEBUG
                         if (currentState != lastReportedState || now - lastReportTime > 1000000000LL) {
                           stateMetric(driverMetrics, (uint64_t)(currentState), timestamp);
                           lastReportedState = currentState;
                           lastReportTime = timestamp;
                         }
#endif
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
                                 if (di < infos.size() && (now - lastSignal > 10000000)) {
                                   kill(infos[di].pid, SIGUSR1);
                                   totalSignalsMetric(driverMetrics, signalsCount++, timestamp);
                                   lastSignal = now;
                                 } else {
                                   skippedSignalsMetric(driverMetrics, skippedCount++, timestamp);
                                 }
                               }
                             }
                             changed = false;
                             currentState = RateLimitingState::CHANGED;
                           } break;
                           case RateLimitingState::CHANGED: {
                             remainingBytes(driverMetrics, totalBytesCreated <= (totalBytesDestroyed + memLimit) ? (totalBytesDestroyed + memLimit) - totalBytesCreated : 0, timestamp);
                             if (totalBytesCreated <= totalBytesDestroyed) {
                               currentState = RateLimitingState::EMPTY;
                             } else if (totalBytesCreated <= (totalBytesDestroyed + memLimit)) {
                               currentState = RateLimitingState::BELOW_LIMIT;
                             } else {
                               currentState = RateLimitingState::ABOVE_LIMIT;
                             }
                             changed = false;
                           } break;
                           case RateLimitingState::EMPTY: {
                             for (size_t di = 0; di < specs.size(); ++di) {
                               if (specs[di].name == "internal-dpl-aod-reader") {
                                 if (di < infos.size()) {
                                   kill(infos[di].pid, SIGUSR1);
                                   totalSignalsMetric(driverMetrics, signalsCount++, timestamp);
                                   lastSignal = now;
                                 }
                               }
                             }
                             changed = false;
                             currentState = RateLimitingState::NEXT_ITERATION_FROM_BELOW;
                           }
                           case RateLimitingState::BELOW_LIMIT: {
                             for (size_t di = 0; di < specs.size(); ++di) {
                               if (specs[di].name == "internal-dpl-aod-reader") {
                                 if (di < infos.size() && (now - lastSignal > 10000000)) {
                                   kill(infos[di].pid, SIGUSR1);
                                   totalSignalsMetric(driverMetrics, signalsCount++, timestamp);
                                   lastSignal = now;
                                 } else {
                                   skippedSignalsMetric(driverMetrics, skippedCount++, timestamp);
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
                     nullptr,
                     nullptr,
                     ServiceKind::Serial};
}

} // namespace o2::framework
