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
#include "ArrowSupport.h"

#include "Framework/ArrowContext.h"
#include "Framework/ArrowTableSlicingCache.h"
#include "Framework/DataProcessor.h"
#include "Framework/CommonDataProcessors.h"
#include "Framework/DataProcessingStats.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/ConfigContext.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataSpecViews.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceMetricsHelper.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DevicesManager.h"
#include "Framework/DeviceConfig.h"
#include "Framework/PluginManager.h"
#include "Framework/ServiceMetricsInfo.h"
#include "WorkflowHelpers.h"
#include "Framework/WorkflowSpecNode.h"
#include "Framework/AnalysisSupportHelpers.h"
#include "Framework/ServiceRegistryRef.h"
#include "Framework/ServiceRegistryHelpers.h"
#include "Framework/Signpost.h"
#include "Framework/DefaultsHelpers.h"

#include "CommonMessageBackendsHelpers.h"
#include <Monitoring/Monitoring.h>
#include "Headers/DataHeader.h"

#include <RtypesCore.h>
#include <fairmq/ProgOptions.h>

#include <uv.h>
#include <boost/program_options/variables_map.hpp>
#include <csignal>

O2_DECLARE_DYNAMIC_LOG(rate_limiting);

namespace o2::framework
{

class EndOfStreamContext;
class ProcessingContext;

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
  int64_t maxMemory = 2000;
  int64_t maxTimeframes = 1000;
};

struct MetricIndices {
  size_t arrowBytesCreated = -1;
  size_t arrowBytesDestroyed = -1;
  size_t arrowMessagesCreated = -1;
  size_t arrowMessagesDestroyed = -1;
  size_t arrowBytesExpired = -1;
  size_t shmOfferBytesConsumed = -1;
  size_t timeframesRead = -1;
  size_t timeframesConsumed = -1;
  size_t timeframesExpired = -1;
  // Timeslices counting
  size_t timeslicesStarted = -1;
  size_t timeslicesExpired = -1;
  size_t timeslicesDone = -1;
};

std::vector<MetricIndices> createDefaultIndices(std::vector<DeviceMetricsInfo>& allDevicesMetrics)
{
  std::vector<MetricIndices> results;

  for (auto& info : allDevicesMetrics) {
    results.emplace_back(MetricIndices{
      .arrowBytesCreated = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-bytes-created"),
      .arrowBytesDestroyed = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-bytes-destroyed"),
      .arrowMessagesCreated = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-messages-created"),
      .arrowMessagesDestroyed = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-messages-destroyed"),
      .arrowBytesExpired = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-bytes-expired"),
      .shmOfferBytesConsumed = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "shm-offer-bytes-consumed"),
      .timeframesRead = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "df-sent"),
      .timeframesConsumed = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "consumed-timeframes"),
      .timeframesExpired = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "expired-timeframes"),
      .timeslicesStarted = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "timeslices-started"),
      .timeslicesExpired = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "timeslices-expired"),
      .timeslicesDone = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "timeslices-done"),
    });
  }
  return results;
}

struct ResourceState {
  int64_t available;
  int64_t offered = 0;
  int64_t lastDeviceOffered = 0;
};
struct ResourceStats {
  int64_t enoughCount; /// How many times the resources were enough
  int64_t lowCount;    /// How many times the resources were not enough
};
struct ResourceSpec {
  char const* name;
  char const* unit;
  char const* api;                /// The callback to give resources to a device
  int64_t maxAvailable;           /// Maximum available quantity for a resource
  int64_t maxQuantum;             /// Largest offer which can be given
  int64_t minQuantum;             /// Smallest offer which can be given
  int64_t metricOfferScaleFactor; /// The scale factor between the metric accounting and offers accounting
};

auto offerResources(ResourceState& resourceState,
                    ResourceSpec const& resourceSpec,
                    ResourceStats& resourceStats,
                    std::vector<DeviceSpec> const& specs,
                    std::vector<DeviceInfo> const& infos,
                    DevicesManager& manager,
                    int64_t offerConsumedCurrentValue,
                    int64_t offerExpiredCurrentValue,
                    int64_t acquiredResourceCurrentValue,
                    int64_t disposedResourceCurrentValue,
                    size_t timestamp,
                    DeviceMetricsInfo& driverMetrics,
                    std::function<void(DeviceMetricsInfo&, int value, size_t timestamp)>& availableResourceMetric,
                    std::function<void(DeviceMetricsInfo&, int value, size_t timestamp)>& unusedOfferedResourceMetric,
                    std::function<void(DeviceMetricsInfo&, int value, size_t timestamp)>& offeredResourceMetric,
                    void* signpostId) -> void
{
  O2_SIGNPOST_ID_FROM_POINTER(sid, rate_limiting, signpostId);
  /// We loop over the devices, starting from where we stopped last time
  /// offering the minimum offer to each one
  int64_t lastCandidate = -1;
  int64_t possibleOffer = resourceSpec.minQuantum;

  for (size_t di = 0; di < specs.size(); di++) {
    if (resourceState.available < possibleOffer) {
      if (resourceStats.lowCount == 0) {
        O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "not enough",
                               "We do not have enough %{public}s (%llu %{public}s) to offer %llu %{public}s. Total offerings %{bytes}llu %{string}s.",
                               resourceSpec.name, resourceState.available, resourceSpec.unit,
                               possibleOffer, resourceSpec.unit,
                               resourceState.offered, resourceSpec.unit);
      }
      resourceStats.lowCount++;
      resourceStats.enoughCount = 0;
      break;
    } else {
      if (resourceStats.enoughCount == 0) {
        O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "enough",
                               "We are back in a state where we enough %{public}s: %llu %{public}s",
                               resourceSpec.name,
                               resourceState.available,
                               resourceSpec.unit);
      }
      resourceStats.lowCount = 0;
      resourceStats.enoughCount++;
    }
    size_t candidate = (resourceState.lastDeviceOffered + di) % specs.size();

    auto& info = infos[candidate];
    // Do not bother for inactive devices
    // FIXME: there is probably a race condition if the device died and we did not
    //        took notice yet...
    if (info.active == false || info.readyToQuit) {
      O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "offer",
                             "Device %s is inactive not offering %{public}s to it.",
                             specs[candidate].name.c_str(), resourceSpec.name);
      continue;
    }
    if (specs[candidate].name != "internal-dpl-aod-reader") {
      O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "offer",
                             "Device %s is not a reader. Not offering %{public}s to it.",
                             specs[candidate].name.c_str(),
                             resourceSpec.name);
      continue;
    }
    possibleOffer = std::min(resourceSpec.maxQuantum, resourceState.available);
    O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "offer",
                           "Offering %llu %{public}s out of %llu to %{public}s",
                           possibleOffer, resourceSpec.unit, resourceState.available, specs[candidate].id.c_str());
    manager.queueMessage(specs[candidate].id.c_str(), fmt::format(fmt::runtime(resourceSpec.api), possibleOffer).data());
    resourceState.available -= possibleOffer;
    resourceState.offered += possibleOffer;
    lastCandidate = candidate;
  }
  // We had at least a valid candidate, so
  // next time we offer to the next device.
  if (lastCandidate >= 0) {
    resourceState.lastDeviceOffered = lastCandidate + 1;
  }

  // unusedOfferedSharedMemory is the amount of memory which was offered and which we know it was
  // not used so far. So we need to account for the amount which got actually read (readerBytesCreated)
  // and the amount which we know was given back.
  static int64_t lastResourceOfferConsumed = 0;
  static int64_t lastUnusedOfferedResource = 0;
  if (offerConsumedCurrentValue != lastResourceOfferConsumed) {
    O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "offer",
                           "Offer consumed so far %llu", offerConsumedCurrentValue);
    lastResourceOfferConsumed = offerConsumedCurrentValue;
  }
  int unusedOfferedResource = (resourceState.offered - (offerExpiredCurrentValue + offerConsumedCurrentValue) / resourceSpec.metricOfferScaleFactor);
  if (lastUnusedOfferedResource != unusedOfferedResource) {
    O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "offer",
                           "unusedOfferedResource(%{public}s):%{bytes}d = offered:%{bytes}llu - (expired:%{bytes}llu + consumed:%{bytes}llu) / %lli",
                           resourceSpec.name,
                           unusedOfferedResource, resourceState.offered,
                           offerExpiredCurrentValue / resourceSpec.metricOfferScaleFactor,
                           offerConsumedCurrentValue / resourceSpec.metricOfferScaleFactor,
                           resourceSpec.metricOfferScaleFactor);
    lastUnusedOfferedResource = unusedOfferedResource;
  }
  // availableSharedMemory is the amount of memory which we know is available to be offered.
  // We subtract the amount which we know was already offered but it's unused and we then balance how
  // much was created with how much was destroyed.
  resourceState.available = resourceSpec.maxAvailable + ((disposedResourceCurrentValue - acquiredResourceCurrentValue) / resourceSpec.metricOfferScaleFactor) - unusedOfferedResource;
  availableResourceMetric(driverMetrics, resourceState.available, timestamp);
  unusedOfferedResourceMetric(driverMetrics, unusedOfferedResource, timestamp);

  offeredResourceMetric(driverMetrics, resourceState.offered, timestamp);
};

auto processTimeslices = [](size_t index, DeviceMetricsInfo& deviceMetrics, bool& changed,
                            int64_t& totalMetricValue, size_t& lastTimestamp) {
  assert(index < deviceMetrics.metrics.size());
  changed |= deviceMetrics.changed[index];
  MetricInfo info = deviceMetrics.metrics[index];
  assert(info.storeIdx < deviceMetrics.uint64Metrics.size());
  auto& data = deviceMetrics.uint64Metrics[info.storeIdx];
  auto value = (int64_t)data[(info.pos - 1) % data.size()];
  totalMetricValue += value;
  auto const& timestamps = DeviceMetricsHelper::getTimestampsStore<uint64_t>(deviceMetrics)[info.storeIdx];
  lastTimestamp = std::max(lastTimestamp, timestamps[(info.pos - 1) % data.size()]);
};

o2::framework::ServiceSpec ArrowSupport::arrowBackendSpec()
{
  using o2::monitoring::Metric;
  using o2::monitoring::Monitoring;
  using o2::monitoring::tags::Key;
  using o2::monitoring::tags::Value;

  return ServiceSpec{
    .name = "arrow-backend",
    .init = CommonMessageBackendsHelpers<ArrowContext>::createCallback(),
    .configure = CommonServices::noConfiguration(),
    .preProcessing = CommonMessageBackendsHelpers<ArrowContext>::clearContext(),
    .postProcessing = CommonMessageBackendsHelpers<ArrowContext>::sendCallback(),
    .preEOS = CommonMessageBackendsHelpers<ArrowContext>::clearContextEOS(),
    .postEOS = CommonMessageBackendsHelpers<ArrowContext>::sendCallbackEOS(),
    .metricHandling = [](ServiceRegistryRef registry,
                         ServiceMetricsInfo const& sm,
                         size_t timestamp) {
                       int64_t totalBytesCreated = 0;
                       int64_t shmOfferBytesConsumed = 0;
                       int64_t totalBytesDestroyed = 0;
                       int64_t totalBytesExpired = 0;
                       int64_t totalMessagesCreated = 0;
                       int64_t totalMessagesDestroyed = 0;
                       int64_t totalTimeframesRead = 0;
                       int64_t totalTimeframesConsumed = 0;
                       int64_t totalTimeframesExpired = 0;
                       int64_t totalTimeslicesStarted = 0;
                       int64_t totalTimeslicesDone = 0;
                       int64_t totalTimeslicesExpired = 0;
                       auto &driverMetrics = sm.driverMetricsInfo;
                       auto &allDeviceMetrics = sm.deviceMetricsInfos;
                       auto &specs = sm.deviceSpecs;
                       auto &infos = sm.deviceInfos;

                       // Aggregated driver metrics for timeslice rate limiting
                       auto createUint64DriverMetric = [&driverMetrics](char const*name) -> auto {
                          return DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, name);
                       };
                       auto createIntDriverMetric = [&driverMetrics](char const*name) -> auto {
                          return DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, name);
                       };

                       static auto stateMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "rate-limit-state");
                       static auto totalBytesCreatedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-bytes-created");
                       static auto shmOfferConsumedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-shm-offer-bytes-consumed");
                       // These are really to monitor the rate limiting
                       static auto unusedOfferedSharedMemoryMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "total-unused-offered-shared-memory");
                       static auto unusedOfferedTimeslicesMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "total-unused-offered-timeslices");
                       static auto availableSharedMemoryMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "total-available-shared-memory");
                       static auto availableTimeslicesMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "total-available-timeslices");
                       static auto offeredSharedMemoryMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "total-offered-shared-memory");
                       static auto offeredTimeslicesMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "total-offered-timeslices");

                       static auto totalBytesDestroyedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-bytes-destroyed");
                       static auto totalBytesExpiredMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-bytes-expired");
                       static auto totalMessagesCreatedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-messages-created");
                       static auto totalMessagesDestroyedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-messages-destroyed");
                       static auto totalTimeframesReadMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-timeframes-read");
                       static auto totalTimeframesConsumedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-timeframes-consumed");
                       static auto totalTimeframesInFlyMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "total-timeframes-in-fly");

                       static auto totalTimeslicesStartedMetric = createUint64DriverMetric("total-timeslices-started");
                       static auto totalTimeslicesExpiredMetric = createUint64DriverMetric("total-timeslices-expired");
                       static auto totalTimeslicesDoneMetric = createUint64DriverMetric("total-timeslices-done");
                       static auto totalTimeslicesInFlyMetric = createIntDriverMetric("total-timeslices-in-fly");

                       static auto totalBytesDeltaMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "arrow-bytes-delta");
                       static auto changedCountMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "changed-metrics-count");
                       static auto totalSignalsMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "aod-reader-signals");
                       static auto signalLatencyMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "aod-signal-latency");
                       static auto skippedSignalsMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "aod-skipped-signals");
                       static auto remainingBytes = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "aod-remaining-bytes");
                       auto& manager = registry.get<DevicesManager>();

                       bool changed = false;

                       size_t lastTimestamp = 0;
                       static std::vector<MetricIndices> allIndices = createDefaultIndices(allDeviceMetrics);
                       for (size_t mi = 0; mi < allDeviceMetrics.size(); ++mi) {
                         auto& deviceMetrics = allDeviceMetrics[mi];
                         if (deviceMetrics.changed.size() != deviceMetrics.metrics.size()) {
                           throw std::runtime_error("deviceMetrics.size() != allDeviceMetrics.size()");
                         }
                         auto& indices = allIndices[mi];
                         {
                           size_t index = indices.arrowBytesCreated;
                           assert(index < deviceMetrics.metrics.size());
                           changed |= deviceMetrics.changed[index];
                           MetricInfo info = deviceMetrics.metrics[index];
                           assert(info.storeIdx < deviceMetrics.uint64Metrics.size());
                           auto& data = deviceMetrics.uint64Metrics[info.storeIdx];
                           auto const& timestamps = DeviceMetricsHelper::getTimestampsStore<uint64_t>(deviceMetrics)[info.storeIdx];
                           auto value = (int64_t)data[(info.pos - 1) % data.size()];
                           totalBytesCreated += value;
                           lastTimestamp = std::max(lastTimestamp, timestamps[(info.pos - 1) % data.size()]);
                         }
                         {
                           size_t index = indices.shmOfferBytesConsumed;
                           assert(index < deviceMetrics.metrics.size());
                           changed |= deviceMetrics.changed[index];
                           MetricInfo info = deviceMetrics.metrics[index];
                           assert(info.storeIdx < deviceMetrics.uint64Metrics.size());
                           auto& data = deviceMetrics.uint64Metrics[info.storeIdx];
                           auto const& timestamps = DeviceMetricsHelper::getTimestampsStore<uint64_t>(deviceMetrics)[info.storeIdx];
                           auto value = (int64_t)data[(info.pos - 1) % data.size()];
                           shmOfferBytesConsumed += value;
                           lastTimestamp = std::max(lastTimestamp, timestamps[(info.pos - 1) % data.size()]);
                         }
                         {
                           size_t index = indices.arrowBytesDestroyed;
                           assert(index < deviceMetrics.metrics.size());
                           changed |= deviceMetrics.changed[index];
                           MetricInfo info = deviceMetrics.metrics[index];
                           assert(info.storeIdx < deviceMetrics.uint64Metrics.size());
                           auto& data = deviceMetrics.uint64Metrics[info.storeIdx];
                           auto value = (int64_t)data[(info.pos - 1) % data.size()];
                           totalBytesDestroyed += value;
                           auto const& timestamps = DeviceMetricsHelper::getTimestampsStore<uint64_t>(deviceMetrics)[info.storeIdx];
                           lastTimestamp = std::max(lastTimestamp, timestamps[(info.pos - 1) % data.size()]);
                         }
                         {
                           size_t index = indices.arrowBytesExpired;
                           assert(index < deviceMetrics.metrics.size());
                           changed |= deviceMetrics.changed[index];
                           MetricInfo info = deviceMetrics.metrics[index];
                           assert(info.storeIdx < deviceMetrics.uint64Metrics.size());
                           auto& data = deviceMetrics.uint64Metrics[info.storeIdx];
                           auto value = (int64_t)data[(info.pos - 1) % data.size()];
                           totalBytesExpired += value;
                           auto const& timestamps = DeviceMetricsHelper::getTimestampsStore<uint64_t>(deviceMetrics)[info.storeIdx];
                           lastTimestamp = std::max(lastTimestamp, timestamps[(info.pos - 1) % data.size()]);
                         }
                         {
                           size_t index = indices.arrowMessagesCreated;
                           assert(index < deviceMetrics.metrics.size());
                           MetricInfo info = deviceMetrics.metrics[index];
                           changed |= deviceMetrics.changed[index];
                           assert(info.storeIdx < deviceMetrics.uint64Metrics.size());
                           auto& data = deviceMetrics.uint64Metrics[info.storeIdx];
                           auto value = (int64_t)data[(info.pos - 1) % data.size()];
                           totalMessagesCreated += value;
                           auto const& timestamps = DeviceMetricsHelper::getTimestampsStore<uint64_t>(deviceMetrics)[info.storeIdx];
                           lastTimestamp = std::max(lastTimestamp, timestamps[(info.pos - 1) % data.size()]);
                         }
                         {
                           size_t index = indices.arrowMessagesDestroyed;
                           assert(index < deviceMetrics.metrics.size());
                           MetricInfo info = deviceMetrics.metrics[index];
                           changed |= deviceMetrics.changed[index];
                           assert(info.storeIdx < deviceMetrics.uint64Metrics.size());
                           auto& data = deviceMetrics.uint64Metrics[info.storeIdx];
                           auto value = (int64_t)data[(info.pos - 1) % data.size()];
                           totalMessagesDestroyed += value;
                           auto const& timestamps = DeviceMetricsHelper::getTimestampsStore<uint64_t>(deviceMetrics)[info.storeIdx];
                           lastTimestamp = std::max(lastTimestamp, timestamps[(info.pos - 1) % data.size()]);
                         }
                         {
                           size_t index = indices.timeframesRead;
                           assert(index < deviceMetrics.metrics.size());
                           changed |= deviceMetrics.changed[index];
                           MetricInfo info = deviceMetrics.metrics[index];
                           assert(info.storeIdx < deviceMetrics.uint64Metrics.size());
                           auto& data = deviceMetrics.uint64Metrics[info.storeIdx];
                           auto value = (int64_t)data[(info.pos - 1) % data.size()];
                           totalTimeframesRead += value;
                           auto const& timestamps = DeviceMetricsHelper::getTimestampsStore<uint64_t>(deviceMetrics)[info.storeIdx];
                           lastTimestamp = std::max(lastTimestamp, timestamps[(info.pos - 1) % data.size()]);
                         }
                         {
                           size_t index = indices.timeframesConsumed;
                           assert(index < deviceMetrics.metrics.size());
                           changed |= deviceMetrics.changed[index];
                           MetricInfo info = deviceMetrics.metrics[index];
                           assert(info.storeIdx < deviceMetrics.uint64Metrics.size());
                           auto& data = deviceMetrics.uint64Metrics[info.storeIdx];
                           auto value = (int64_t)data[(info.pos - 1) % data.size()];
                           totalTimeframesConsumed += value;
                           auto const& timestamps = DeviceMetricsHelper::getTimestampsStore<uint64_t>(deviceMetrics)[info.storeIdx];
                           lastTimestamp = std::max(lastTimestamp, timestamps[(info.pos - 1) % data.size()]);
                         }
                         {
                           size_t index = indices.timeframesExpired;
                           assert(index < deviceMetrics.metrics.size());
                           changed |= deviceMetrics.changed[index];
                           MetricInfo info = deviceMetrics.metrics[index];
                           assert(info.storeIdx < deviceMetrics.uint64Metrics.size());
                           auto& data = deviceMetrics.uint64Metrics[info.storeIdx];
                           auto value = (int64_t)data[(info.pos - 1) % data.size()];
                           totalTimeframesExpired += value;
                           auto const& timestamps = DeviceMetricsHelper::getTimestampsStore<uint64_t>(deviceMetrics)[info.storeIdx];
                           lastTimestamp = std::max(lastTimestamp, timestamps[(info.pos - 1) % data.size()]);
                         }
                         processTimeslices(indices.timeslicesStarted, deviceMetrics, changed, totalTimeslicesStarted, lastTimestamp);
                         processTimeslices(indices.timeslicesExpired, deviceMetrics, changed, totalTimeslicesExpired, lastTimestamp);
                         processTimeslices(indices.timeslicesDone, deviceMetrics, changed, totalTimeslicesDone, lastTimestamp);
                       }
                       static uint64_t unchangedCount = 0;
                       if (changed) {
                         totalBytesCreatedMetric(driverMetrics, totalBytesCreated, timestamp);
                         totalBytesDestroyedMetric(driverMetrics, totalBytesDestroyed, timestamp);
                         totalBytesExpiredMetric(driverMetrics, totalBytesExpired, timestamp);
                         shmOfferConsumedMetric(driverMetrics, shmOfferBytesConsumed, timestamp);
                         totalMessagesCreatedMetric(driverMetrics, totalMessagesCreated, timestamp);
                         totalMessagesDestroyedMetric(driverMetrics, totalMessagesDestroyed, timestamp);
                         totalTimeframesReadMetric(driverMetrics, totalTimeframesRead, timestamp);
                         totalTimeframesConsumedMetric(driverMetrics, totalTimeframesConsumed, timestamp);
                         totalTimeframesInFlyMetric(driverMetrics, (int)(totalTimeframesRead - totalTimeframesConsumed), timestamp);
                         totalTimeslicesStartedMetric(driverMetrics, totalTimeslicesStarted, timestamp);
                         totalTimeslicesExpiredMetric(driverMetrics, totalTimeslicesExpired, timestamp);
                         totalTimeslicesDoneMetric(driverMetrics, totalTimeslicesDone, timestamp);
                         totalTimeslicesInFlyMetric(driverMetrics, (int)(totalTimeslicesStarted - totalTimeslicesDone), timestamp);
                         totalBytesDeltaMetric(driverMetrics, totalBytesCreated - totalBytesExpired - totalBytesDestroyed, timestamp);
                       } else {
                         unchangedCount++;
                       }
                       changedCountMetric(driverMetrics, unchangedCount, timestamp);

                       static const ResourceSpec shmResourceSpec{
                         .name = "shared memory",
                         .unit = "MB",
                         .api = "/shm-offer {}",
                         .maxAvailable = (int64_t)registry.get<RateLimitConfig>().maxMemory,
                         .maxQuantum = 100,
                         .minQuantum = 50,
                         .metricOfferScaleFactor = 1000000,
                       };
                       static const ResourceSpec timesliceResourceSpec{
                         .name = "timeslice",
                         .unit = "timeslices",
                         .api = "/timeslice-offer {}",
                         .maxAvailable = (int64_t)registry.get<RateLimitConfig>().maxTimeframes,
                         .maxQuantum = 1,
                         .minQuantum = 1,
                         .metricOfferScaleFactor = 1,
                       };
                       static ResourceState shmResourceState{
                         .available = shmResourceSpec.maxAvailable,
                       };
                       static ResourceState timesliceResourceState{
                         .available = timesliceResourceSpec.maxAvailable,
                       };
                       static ResourceStats shmResourceStats{
                         .enoughCount = shmResourceState.available - shmResourceSpec.minQuantum > 0 ? 1 : 0,
                         .lowCount = shmResourceState.available - shmResourceSpec.minQuantum > 0 ? 0 : 1
                       };
                       static ResourceStats timesliceResourceStats{
                         .enoughCount = shmResourceState.available - shmResourceSpec.minQuantum > 0 ? 1 : 0,
                         .lowCount = shmResourceState.available - shmResourceSpec.minQuantum > 0 ? 0 : 1
                       };

                       offerResources(timesliceResourceState, timesliceResourceSpec, timesliceResourceStats,
                                      specs, infos, manager, totalTimeframesConsumed, totalTimeslicesExpired,
                                      totalTimeslicesStarted, totalTimeslicesDone, timestamp, driverMetrics,
                                      availableTimeslicesMetric, unusedOfferedTimeslicesMetric, offeredTimeslicesMetric,
                                      (void*)&sm);

                       offerResources(shmResourceState, shmResourceSpec, shmResourceStats,
                                      specs, infos, manager, shmOfferBytesConsumed, totalBytesExpired,
                                      totalBytesCreated, totalBytesDestroyed, timestamp, driverMetrics,
                                      availableSharedMemoryMetric, unusedOfferedSharedMemoryMetric, offeredSharedMemoryMetric,
                                      (void*)&sm); },
    .postDispatching = [](ProcessingContext& ctx, void* service) {
                       using DataHeader = o2::header::DataHeader;
                       auto* arrow = reinterpret_cast<ArrowContext*>(service);
                       auto totalBytes = 0;
                       auto totalMessages = 0;
                       O2_SIGNPOST_ID_FROM_POINTER(sid, rate_limiting, &arrow);
                       for (auto& input : ctx.inputs()) {
                         if (input.header == nullptr) {
                           continue;
                         }
                         auto const* dh = DataRefUtils::getHeader<DataHeader*>(input);
                         auto payloadSize = DataRefUtils::getPayloadSize(input);
                         if (dh->serialization != o2::header::gSerializationMethodArrow) {
                           O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "offer",
                                                  "Message %{public}.4s/%{public}.16s is not of kind arrow, therefore we are not accounting its shared memory.",
                                                  dh->dataOrigin.str, dh->dataDescription.str);
                           continue;
                         }
                         bool forwarded = std::ranges::any_of(ctx.services().get<DeviceSpec const>().forwards, [&dh](auto const& forward) { return DataSpecUtils::match(forward.matcher, *dh); });
                         if (forwarded) {
                           O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "offer",
                                                  "Message %{public}.4s/%{public}.16s is forwarded so we are not returning its memory.",
                                                  dh->dataOrigin.str, dh->dataDescription.str);
                           continue;
                         }
                         O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "offer",
                                                "Message %{public}.4s/%{public}.16s is being deleted. We will return %{bytes}f MB.",
                                                dh->dataOrigin.str, dh->dataDescription.str, payloadSize / 1000000.);
                         totalBytes += payloadSize;
                         totalMessages += 1;
                       }
                       arrow->updateBytesDestroyed(totalBytes);
                       O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "give back",
                                              "%{bytes}f MB bytes being given back to reader, totaling %{bytes}f MB",
                                              totalBytes / 1000000., arrow->bytesDestroyed() / 1000000.);
                       arrow->updateMessagesDestroyed(totalMessages);
                       auto& stats = ctx.services().get<DataProcessingStats>();
                       stats.updateStats({static_cast<short>(ProcessingStatsId::ARROW_BYTES_DESTROYED), DataProcessingStats::Op::Set, static_cast<int64_t>(arrow->bytesDestroyed())});
                       stats.updateStats({static_cast<short>(ProcessingStatsId::ARROW_MESSAGES_DESTROYED), DataProcessingStats::Op::Set, static_cast<int64_t>(arrow->messagesDestroyed())});
                       stats.processCommandQueue(); },
    .driverInit = [](ServiceRegistryRef registry, DeviceConfig const& dc) {
                       auto config = new RateLimitConfig{};
                       int readers = std::stoll(dc.options["readers"].as<std::string>());
                       if (dc.options.count("aod-memory-rate-limit") && dc.options["aod-memory-rate-limit"].defaulted() == false) {
                         config->maxMemory = std::stoll(dc.options["aod-memory-rate-limit"].as<std::string>()) / 1000000;
                       } else {
                         config->maxMemory = readers * 2000;
                       }
                       if (dc.options.count("timeframes-rate-limit") && dc.options["timeframes-rate-limit"].defaulted() == false) {
                         config->maxTimeframes = std::stoll(dc.options["timeframes-rate-limit"].as<std::string>());
                       } else {
                         config->maxTimeframes = readers * DefaultsHelpers::pipelineLength();
                       }
                       static bool once = false;
                       // Until we guarantee this is called only once...
                       if (!once) {
                         O2_SIGNPOST_ID_GENERATE(sid, rate_limiting);
                         O2_SIGNPOST_EVENT_EMIT_INFO(rate_limiting, sid, "setup",
                                                     "Rate limiting set up at %{bytes}llu MB and %llu timeframes distributed over %d readers",
                                                     config->maxMemory, config->maxTimeframes, readers);
                         registry.registerService(ServiceRegistryHelpers::handleForService<RateLimitConfig>(config));
                         once = true;
                       } },
    .adjustTopology = [](WorkflowSpecNode& node, ConfigContext const& ctx) {
      auto& workflow = node.specs;
      auto spawner = std::ranges::find_if(workflow, [](DataProcessorSpec const& spec) { return spec.name.starts_with("internal-dpl-aod-spawner"); });
      auto analysisCCDB = std::ranges::find_if(workflow, [](DataProcessorSpec const& spec) { return spec.name.starts_with("internal-dpl-aod-ccdb"); });
      auto builder = std::ranges::find_if(workflow, [](DataProcessorSpec const& spec) { return spec.name.starts_with("internal-dpl-aod-index-builder"); });
      auto reader = std::ranges::find_if(workflow, [](DataProcessorSpec const& spec) { return spec.name.starts_with("internal-dpl-aod-reader"); });
      auto writer = std::ranges::find_if(workflow, [](DataProcessorSpec const& spec) { return spec.name.starts_with("internal-dpl-aod-writer"); });
      auto& dec = ctx.services().get<DanglingEdgesContext>();
      dec.requestedAODs.clear();
      dec.requestedDYNs.clear();
      dec.providedDYNs.clear();
      dec.providedTIMs.clear();
      dec.requestedTIMs.clear();

      auto inputSpecLessThan = [](InputSpec const& lhs, InputSpec const& rhs) { return DataSpecUtils::describe(lhs) < DataSpecUtils::describe(rhs); };
      auto outputSpecLessThan = [](OutputSpec const& lhs, OutputSpec const& rhs) { return DataSpecUtils::describe(lhs) < DataSpecUtils::describe(rhs); };

      if (builder != workflow.end()) {
        // collect currently requested IDXs
        dec.requestedIDXs.clear();
        for (auto& d : workflow | views::exclude_by_name(builder->name)) {
          d.inputs |
            views::partial_match_filter(header::DataOrigin{"IDX"}) |
            sinks::update_input_list{dec.requestedIDXs};
        }
        // recreate inputs and outputs
        builder->inputs.clear();
        builder->outputs.clear();

        // load real AlgorithmSpec before deployment
        builder->algorithm = PluginManager::loadAlgorithmFromPlugin("O2FrameworkOnDemandTablesSupport", "IndexTableBuilder", ctx);
        AnalysisSupportHelpers::addMissingOutputsToBuilder(dec.requestedIDXs, dec.requestedAODs, dec.requestedDYNs, *builder);
      }

      if (spawner != workflow.end()) {
        // collect currently requested DYNs
        for (auto& d : workflow | views::exclude_by_name(spawner->name)) {
          d.inputs |
            views::partial_match_filter(header::DataOrigin{"DYN"}) |
            sinks::update_input_list{dec.requestedDYNs};
          d.outputs |
            views::partial_match_filter(header::DataOrigin{"DYN"}) |
            sinks::append_to{dec.providedDYNs};
        }
        std::ranges::sort(dec.requestedDYNs, inputSpecLessThan);
        std::ranges::sort(dec.providedDYNs, outputSpecLessThan);
        dec.spawnerInputs.clear();
        dec.requestedDYNs |
          views::filter_not_matching(dec.providedDYNs) |
          sinks::append_to{dec.spawnerInputs};
        // recreate inputs and outputs
        spawner->outputs.clear();
        spawner->inputs.clear();

        // load real AlgorithmSpec before deployment
        spawner->algorithm = PluginManager::loadAlgorithmFromPlugin("O2FrameworkOnDemandTablesSupport", "ExtendedTableSpawner", ctx);
        AnalysisSupportHelpers::addMissingOutputsToSpawner({}, dec.spawnerInputs, dec.requestedAODs, *spawner);
      }

      if (analysisCCDB != workflow.end()) {
        for (auto& d : workflow | views::exclude_by_name(analysisCCDB->name)) {
          d.inputs | views::partial_match_filter(header::DataOrigin{"ATIM"}) | sinks::update_input_list{dec.requestedTIMs};
          d.outputs | views::partial_match_filter(header::DataOrigin{"ATIM"}) | sinks::append_to{dec.providedTIMs};
        }
        std::ranges::sort(dec.requestedTIMs, inputSpecLessThan);
        std::ranges::sort(dec.providedTIMs, outputSpecLessThan);
        // Use ranges::to<std::vector<>> in C++23...
        dec.analysisCCDBInputs.clear();
        dec.requestedTIMs | views::filter_not_matching(dec.providedTIMs) | sinks::append_to{dec.analysisCCDBInputs};

        // recreate inputs and outputs
        analysisCCDB->outputs.clear();
        analysisCCDB->inputs.clear();
        // load real AlgorithmSpec before deployment
        // FIXME how can I make the lookup depend on DYN tables as well??
        analysisCCDB->algorithm = PluginManager::loadAlgorithmFromPlugin("O2FrameworkCCDBSupport", "AnalysisCCDBFetcherPlugin", ctx);
        AnalysisSupportHelpers::addMissingOutputsToBuilder(dec.analysisCCDBInputs, dec.requestedAODs, dec.requestedDYNs, *analysisCCDB);
      }

      if (writer != workflow.end()) {
        workflow.erase(writer);
      }

      if (reader != workflow.end()) {
        // If reader and/or builder were adjusted, remove unneeded outputs
        // update currently requested AODs
        for (auto& d : workflow) {
          d.inputs |
            views::partial_match_filter(AODOrigins) |
            sinks::update_input_list{dec.requestedAODs};
        }

        // remove unmatched outputs
        auto o_end = std::remove_if(reader->outputs.begin(), reader->outputs.end(), [&](OutputSpec const& o) {
          return !DataSpecUtils::partialMatch(o, o2::header::DataDescription{"TFNumber"}) && !DataSpecUtils::partialMatch(o, o2::header::DataDescription{"TFFilename"}) && std::none_of(dec.requestedAODs.begin(), dec.requestedAODs.end(), [&](InputSpec const& i) { return DataSpecUtils::match(i, o); });
        });
        reader->outputs.erase(o_end, reader->outputs.end());
        if (reader->outputs.empty()) {
          // nothing to read
          workflow.erase(reader);
        } else {
          // load reader algorithm before deployment
          auto mctracks2aod = std::find_if(workflow.begin(), workflow.end(), [](auto const& x) { return x.name == "mctracks-to-aod"; });
          if (mctracks2aod == workflow.end()) { // add normal reader algorithm only if no on-the-fly generator is injected
            reader->algorithm = CommonDataProcessors::wrapWithTimesliceConsumption(PluginManager::loadAlgorithmFromPlugin("O2FrameworkAnalysisSupport", "ROOTFileReader", ctx));
          } // otherwise the algorithm was set in injectServiceDevices
        }
      }

      WorkflowHelpers::injectAODWriter(workflow, ctx);

      // Move the dummy sink at the end, if needed
      for (size_t i = 0; i < workflow.size(); ++i) {
        if (workflow[i].name == "internal-dpl-injected-dummy-sink") {
          workflow.push_back(workflow[i]);
          workflow.erase(workflow.begin() + i);
          break;
        }
      } },
    .kind = ServiceKind::Global};
}

o2::framework::ServiceSpec ArrowSupport::arrowTableSlicingCacheDefSpec()
{
  return ServiceSpec{
    .name = "arrow-slicing-cache-def",
    .uniqueId = CommonServices::simpleServiceId<ArrowTableSlicingCacheDef>(),
    .init = CommonServices::simpleServiceInit<ArrowTableSlicingCacheDef, ArrowTableSlicingCacheDef, ServiceKind::Global>(),
    .kind = ServiceKind::Global};
}

o2::framework::ServiceSpec ArrowSupport::arrowTableSlicingCacheSpec()
{
  return ServiceSpec{
    .name = "arrow-slicing-cache",
    .uniqueId = CommonServices::simpleServiceId<ArrowTableSlicingCache>(),
    .init = [](ServiceRegistryRef services, DeviceState&, fair::mq::ProgOptions&) { return ServiceHandle{TypeIdHelpers::uniqueId<ArrowTableSlicingCache>(),
                                                                                                         new ArrowTableSlicingCache(Cache{services.get<ArrowTableSlicingCacheDef>().bindingsKeys},
                                                                                                                                    Cache{services.get<ArrowTableSlicingCacheDef>().bindingsKeysUnsorted}),
                                                                                                         ServiceKind::Stream, typeid(ArrowTableSlicingCache).name()}; },
    .configure = CommonServices::noConfiguration(),
    .preProcessing = [](ProcessingContext& pc, void* service_ptr) {
      auto* service = static_cast<ArrowTableSlicingCache*>(service_ptr);
      auto& caches = service->bindingsKeys;
      for (auto i = 0u; i < caches.size(); ++i) {
        if (caches[i].enabled && pc.inputs().getPos(caches[i].binding.c_str()) >= 0) {
          auto status = service->updateCacheEntry(i, pc.inputs().get<TableConsumer>(caches[i].matcher)->asArrowTable());
          if (!status.ok()) {
            throw runtime_error_f("Failed to update slice cache for %s/%s", caches[i].binding.c_str(), caches[i].key.c_str());
          }
        }
      }
      auto& unsortedCaches = service->bindingsKeysUnsorted;
      for (auto i = 0u; i < unsortedCaches.size(); ++i) {
        if (unsortedCaches[i].enabled && pc.inputs().getPos(unsortedCaches[i].binding.c_str()) >= 0) {
          auto status = service->updateCacheEntryUnsorted(i, pc.inputs().get<TableConsumer>(unsortedCaches[i].matcher)->asArrowTable());
          if (!status.ok()) {
            throw runtime_error_f("failed to update slice cache (unsorted) for %s/%s", unsortedCaches[i].binding.c_str(), unsortedCaches[i].key.c_str());
          }
        }
      } },
    .kind = ServiceKind::Stream};
}

} // namespace o2::framework
