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

#include "Framework/AODReaderHelpers.h"
#include "Framework/ArrowContext.h"
#include "Framework/ArrowTableSlicingCache.h"
#include "Framework/SliceCache.h"
#include "Framework/DataProcessor.h"
#include "Framework/DataProcessingStats.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/ConfigContext.h"
#include "Framework/CommonDataProcessors.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataSpecViews.h"
#include "Framework/DeviceSpec.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/Tracing.h"
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

#include "CommonMessageBackendsHelpers.h"
#include <Monitoring/Monitoring.h>
#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"

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
  int64_t maxTimeframes = 0;
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
};

std::vector<MetricIndices> createDefaultIndices(std::vector<DeviceMetricsInfo>& allDevicesMetrics)
{
  std::vector<MetricIndices> results;

  for (auto& info : allDevicesMetrics) {
    MetricIndices indices;
    indices.arrowBytesCreated = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-bytes-created");
    indices.arrowBytesDestroyed = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-bytes-destroyed");
    indices.arrowMessagesCreated = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-messages-created");
    indices.arrowMessagesDestroyed = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-messages-destroyed");
    indices.arrowBytesExpired = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "arrow-bytes-expired");
    indices.shmOfferBytesConsumed = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "shm-offer-bytes-consumed");
    indices.timeframesRead = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "df-sent");
    indices.timeframesConsumed = DeviceMetricsHelper::bookNumericMetric<uint64_t>(info, "consumed-timeframes");
    results.push_back(indices);
  }
  return results;
}

uint64_t calculateAvailableSharedMemory(ServiceRegistryRef registry)
{
  return registry.get<RateLimitConfig>().maxMemory;
}

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
                       auto &driverMetrics = sm.driverMetricsInfo;
                       auto &allDeviceMetrics = sm.deviceMetricsInfos;
                       auto &specs = sm.deviceSpecs;
                       auto &infos = sm.deviceInfos;
                       O2_SIGNPOST_ID_FROM_POINTER(sid, rate_limiting, &sm);

                       static auto stateMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "rate-limit-state");
                       static auto totalBytesCreatedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-bytes-created");
                       static auto shmOfferConsumedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-shm-offer-bytes-consumed");
                       static auto unusedOfferedSharedMemoryMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "total-unused-offered-shared-memory");
                       static auto availableSharedMemoryMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "total-available-shared-memory");
                       static auto offeredSharedMemoryMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "total-offered-shared-memory");
                       static auto totalBytesDestroyedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-bytes-destroyed");
                       static auto totalBytesExpiredMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-bytes-expired");
                       static auto totalMessagesCreatedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-messages-created");
                       static auto totalMessagesDestroyedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-messages-destroyed");
                       static auto totalTimeframesReadMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-timeframes-read");
                       static auto totalTimeframesConsumedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-timeframes-consumed");
                       static auto totalTimeframesInFlyMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "total-timeframes-in-fly");
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
                         totalBytesDeltaMetric(driverMetrics, totalBytesCreated - totalBytesExpired - totalBytesDestroyed, timestamp);
                       } else {
                         unchangedCount++;
                       }
                       changedCountMetric(driverMetrics, unchangedCount, timestamp);
                       auto maxTimeframes = registry.get<RateLimitConfig>().maxTimeframes;
                       if (maxTimeframes && (totalTimeframesRead - totalTimeframesConsumed) > maxTimeframes) {
                         return;
                       }
                       struct ResourceState {
                          int64_t available;
                          int64_t offered = 0;
                          int64_t lastDeviceOffered = 0;
                       };
                       struct ResourceStats {
                          int64_t enoughCount;
                          int64_t lowCount;
                       };
                       struct ResourceSpec{
                          int64_t maxAvailable;
                          int64_t maxQuantum;
                          int64_t minQuantum;
                       };
                       static const ResourceSpec resourceSpec{
                         .maxAvailable = (int64_t)calculateAvailableSharedMemory(registry),
                         .maxQuantum = 100,
                         .minQuantum = 50,
                       };
                       static ResourceState resourceState{
                         .available = resourceSpec.maxAvailable,
                       };
                       static ResourceStats resourceStats{
                         .enoughCount = resourceState.available - resourceSpec.minQuantum > 0 ? 1 : 0,
                         .lowCount = resourceState.available - resourceSpec.minQuantum > 0 ? 0 : 1
                       };

                       /// We loop over the devices, starting from where we stopped last time
                       /// offering MIN_QUANTUM_SHARED_MEMORY of shared memory to each reader.
                       int64_t lastCandidate = -1;
                       int64_t possibleOffer = resourceSpec.minQuantum;

                       for (size_t di = 0; di < specs.size(); di++) {
                         if (resourceState.available < possibleOffer) {
                           if (resourceStats.lowCount == 0) {
                             O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "not enough",
                                                    "We do not have enough shared memory (%{bytes}llu MB) to offer %{bytes}llu MB. Total offerings %{bytes}llu",
                                                     resourceState.available, possibleOffer, resourceState.offered);
                           }
                           resourceStats.lowCount++;
                           resourceStats.enoughCount = 0;
                           break;
                         } else {
                           if (resourceStats.enoughCount == 0) {
                             O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "enough",
                                                    "We are back in a state where we enough shared memory: %{bytes}llu MB", resourceState.available);
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
                                                  "Device %s is inactive not offering memory to it.", specs[candidate].name.c_str());
                           continue;
                         }
                         if (specs[candidate].name != "internal-dpl-aod-reader") {
                           O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "offer",
                                                  "Device %s is not a reader. Not offering memory to it.", specs[candidate].name.c_str());
                           continue;
                         }
                         possibleOffer = std::min(resourceSpec.maxQuantum, resourceState.available);
                         O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "offer",
                                                "Offering %{bytes}llu MB out of %{bytes}llu to %{public}s",
                                                possibleOffer, resourceState.available, specs[candidate].id.c_str());
                         manager.queueMessage(specs[candidate].id.c_str(), fmt::format("/shm-offer {}", possibleOffer).data());
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
                       static int64_t lastShmOfferConsumed = 0;
                       static int64_t lastUnusedOfferedMemory = 0;
                       if (shmOfferBytesConsumed != lastShmOfferConsumed) {
                         O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "offer",
                                                "Offer consumed so far %{bytes}llu", shmOfferBytesConsumed);
                         lastShmOfferConsumed = shmOfferBytesConsumed;
                       }
                       int unusedOfferedMemory = (resourceState.offered - (totalBytesExpired + shmOfferBytesConsumed) / 1000000);
                       if (lastUnusedOfferedMemory != unusedOfferedMemory) {
                         O2_SIGNPOST_EVENT_EMIT(rate_limiting, sid, "offer",
                                                "unusedOfferedMemory:%{bytes}d = offered:%{bytes}llu - (expired:%{bytes}llu + consumed:%{bytes}llu) / 1000000",
                                                 unusedOfferedMemory, resourceState.offered, totalBytesExpired / 1000000, shmOfferBytesConsumed / 1000000);
                         lastUnusedOfferedMemory = unusedOfferedMemory;
                       }
                       // availableSharedMemory is the amount of memory which we know is available to be offered.
                       // We subtract the amount which we know was already offered but it's unused and we then balance how
                       // much was created with how much was destroyed.
                       resourceState.available = resourceSpec.maxAvailable + ((totalBytesDestroyed - totalBytesCreated) / 1000000) - unusedOfferedMemory;
                       availableSharedMemoryMetric(driverMetrics, resourceState.available, timestamp);
                       unusedOfferedSharedMemoryMetric(driverMetrics, unusedOfferedMemory, timestamp);

                       offeredSharedMemoryMetric(driverMetrics, resourceState.offered, timestamp); },
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
                         bool forwarded = false;
                         for (auto const& forward : ctx.services().get<DeviceSpec const>().forwards) {
                           if (DataSpecUtils::match(forward.matcher, *dh)) {
                             forwarded = true;
                             break;
                           }
                         }
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
                         config->maxMemory = readers * 500;
                       }
                       if (dc.options.count("timeframes-rate-limit") && dc.options["timeframes-rate-limit"].as<std::string>() == "readers") {
                         config->maxTimeframes = readers;
                       } else {
                         config->maxTimeframes = std::stoll(dc.options["timeframes-rate-limit"].as<std::string>());
                       }
                       static bool once = false;
                       // Until we guarantee this is called only once...
                       if (!once) {
                         O2_SIGNPOST_ID_GENERATE(sid, rate_limiting);
                         O2_SIGNPOST_EVENT_EMIT_INFO(rate_limiting, sid, "setup",
                                                     "Rate limiting set up at %{bytes}llu MB distributed over %d readers",
                                                     config->maxMemory, readers);
                         registry.registerService(ServiceRegistryHelpers::handleForService<RateLimitConfig>(config));
                         once = true;
                       } },
    .adjustTopology = [](WorkflowSpecNode& node, ConfigContext const& ctx) {
      auto& workflow = node.specs;
      auto spawner = std::find_if(workflow.begin(), workflow.end(), [](DataProcessorSpec const& spec) { return spec.name == "internal-dpl-aod-spawner"; });
      auto analysisCCDB = std::find_if(workflow.begin(), workflow.end(), [](DataProcessorSpec const& spec) { return spec.name == "internal-dpl-aod-ccdb"; });
      auto builder = std::find_if(workflow.begin(), workflow.end(), [](DataProcessorSpec const& spec) { return spec.name == "internal-dpl-aod-index-builder"; });
      auto reader = std::find_if(workflow.begin(), workflow.end(), [](DataProcessorSpec const& spec) { return spec.name == "internal-dpl-aod-reader"; });
      auto writer = std::find_if(workflow.begin(), workflow.end(), [](DataProcessorSpec const& spec) { return spec.name == "internal-dpl-aod-writer"; });
      auto &ac = ctx.services().get<AnalysisContext>();
      ac.requestedAODs.clear();
      ac.requestedDYNs.clear();
      ac.providedDYNs.clear();
      ac.providedTIMs.clear();
      ac.requestedTIMs.clear();


      auto inputSpecLessThan = [](InputSpec const& lhs, InputSpec const& rhs) { return DataSpecUtils::describe(lhs) < DataSpecUtils::describe(rhs); };
      auto outputSpecLessThan = [](OutputSpec const& lhs, OutputSpec const& rhs) { return DataSpecUtils::describe(lhs) < DataSpecUtils::describe(rhs); };

      if (builder != workflow.end()) {
        // collect currently requested IDXs
        ac.requestedIDXs.clear();
        for (auto& d : workflow) {
          if (d.name == builder->name) {
            continue;
          }
          for (auto& i : d.inputs) {
            if (DataSpecUtils::partialMatch(i, header::DataOrigin{"IDX"})) {
              auto copy = i;
              DataSpecUtils::updateInputList(ac.requestedIDXs, std::move(copy));
            }
          }
        }
        // recreate inputs and outputs
        builder->inputs.clear();
        builder->outputs.clear();
        // replace AlgorithmSpec
        //  FIXME: it should be made more generic, so it does not need replacement...
        builder->algorithm = readers::AODReaderHelpers::indexBuilderCallback(ac.requestedIDXs);
        AnalysisSupportHelpers::addMissingOutputsToBuilder(ac.requestedIDXs, ac.requestedAODs, ac.requestedDYNs, *builder);
      }

      if (spawner != workflow.end()) {
        // collect currently requested DYNs
        for (auto& d : workflow) {
          if (d.name == spawner->name) {
            continue;
          }
          for (auto const& i : d.inputs) {
            if (DataSpecUtils::partialMatch(i, header::DataOrigin{"DYN"})) {
              auto copy = i;
              DataSpecUtils::updateInputList(ac.requestedDYNs, std::move(copy));
            }
          }
          for (auto const& o : d.outputs) {
            if (DataSpecUtils::partialMatch(o, header::DataOrigin{"DYN"})) {
              ac.providedDYNs.emplace_back(o);
            }
          }
        }
        std::sort(ac.requestedDYNs.begin(), ac.requestedDYNs.end(), inputSpecLessThan);
        std::sort(ac.providedDYNs.begin(), ac.providedDYNs.end(), outputSpecLessThan);
        ac.spawnerInputs.clear();
        for (auto& input : ac.requestedDYNs) {
          if (std::none_of(ac.providedDYNs.begin(), ac.providedDYNs.end(), [&input](auto const& x) { return DataSpecUtils::match(input, x); })) {
            ac.spawnerInputs.emplace_back(input);
          }
        }
        // recreate inputs and outputs
        spawner->outputs.clear();
        spawner->inputs.clear();
        // replace AlgorithmSpec
        // FIXME: it should be made more generic, so it does not need replacement...
        spawner->algorithm = readers::AODReaderHelpers::aodSpawnerCallback(ac.spawnerInputs);
        AnalysisSupportHelpers::addMissingOutputsToSpawner({}, ac.spawnerInputs, ac.requestedAODs, *spawner);
      }

      if (analysisCCDB != workflow.end()) {
        for (auto& d : workflow | views::exclude_by_name(analysisCCDB->name)) {
          d.inputs | views::partial_match_filter(header::DataOrigin{"ATIM"}) | sinks::update_input_list{ac.requestedTIMs};
          d.outputs | views::partial_match_filter(header::DataOrigin{"ATIM"}) | sinks::append_to{ac.providedTIMs};
        }
        std::sort(ac.requestedTIMs.begin(), ac.requestedTIMs.end(), inputSpecLessThan);
        std::sort(ac.providedTIMs.begin(), ac.providedTIMs.end(), outputSpecLessThan);
        // Use ranges::to<std::vector<>> in C++23...
        ac.analysisCCDBInputs.clear();
        ac.requestedTIMs | views::filter_not_matching(ac.providedTIMs) | sinks::append_to{ac.analysisCCDBInputs};

        // recreate inputs and outputs
        analysisCCDB->outputs.clear();
        analysisCCDB->inputs.clear();
        // replace AlgorithmSpec
        // FIXME: it should be made more generic, so it does not need replacement...
        // FIXME how can I make the lookup depend on DYN tables as well??
        analysisCCDB->algorithm = PluginManager::loadAlgorithmFromPlugin("O2FrameworkCCDBSupport", "AnalysisCCDBFetcherPlugin", ctx);
        AnalysisSupportHelpers::addMissingOutputsToAnalysisCCDBFetcher({}, ac.analysisCCDBInputs, ac.requestedAODs, ac.requestedDYNs, *analysisCCDB);
      }

      if (writer != workflow.end()) {
        workflow.erase(writer);
      }

      if (reader != workflow.end()) {
        // If reader and/or builder were adjusted, remove unneeded outputs
        // update currently requested AODs
        for (auto& d : workflow) {
          for (auto const& i : d.inputs) {
            if (DataSpecUtils::partialMatch(i, AODOrigins)) {
              auto copy = i;
              DataSpecUtils::updateInputList(ac.requestedAODs, std::move(copy));
            }
          }
        }

        // remove unmatched outputs
        auto o_end = std::remove_if(reader->outputs.begin(), reader->outputs.end(), [&](OutputSpec const& o) {
          return !DataSpecUtils::partialMatch(o, o2::header::DataDescription{"TFNumber"}) && !DataSpecUtils::partialMatch(o, o2::header::DataDescription{"TFFilename"}) && std::none_of(ac.requestedAODs.begin(), ac.requestedAODs.end(), [&](InputSpec const& i) { return DataSpecUtils::match(i, o); });
        });
        reader->outputs.erase(o_end, reader->outputs.end());
        if (reader->outputs.empty()) {
          // nothing to read
          workflow.erase(reader);
        }
      }



      // replace writer as some outputs may have become dangling and some are now consumed
      auto [outputsInputs, isDangling] = WorkflowHelpers::analyzeOutputs(workflow);

      // create DataOutputDescriptor
      std::shared_ptr<DataOutputDirector> dod = AnalysisSupportHelpers::getDataOutputDirector(ctx);

      // select outputs of type AOD which need to be saved
      // ATTENTION: if there are dangling outputs the getGlobalAODSink
      // has to be created in any case!
      ac.outputsInputsAOD.clear();

      for (auto ii = 0u; ii < outputsInputs.size(); ii++) {
        if (DataSpecUtils::partialMatch(outputsInputs[ii], extendedAODOrigins)) {
          auto ds = dod->getDataOutputDescriptors(outputsInputs[ii]);
          if (!ds.empty() || isDangling[ii]) {
            ac.outputsInputsAOD.emplace_back(outputsInputs[ii]);
          }
        }
      }

      // file sink for any AOD output
      if (!ac.outputsInputsAOD.empty()) {
        // add TFNumber and TFFilename as input to the writer
        ac.outputsInputsAOD.emplace_back("tfn", "TFN", "TFNumber");
        ac.outputsInputsAOD.emplace_back("tff", "TFF", "TFFilename");
        workflow.push_back(AnalysisSupportHelpers::getGlobalAODSink(ctx));
      }
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
          auto status = service->updateCacheEntry(i, pc.inputs().get<TableConsumer>(caches[i].binding.c_str())->asArrowTable());
          if (!status.ok()) {
            throw runtime_error_f("Failed to update slice cache for %s/%s", caches[i].binding.c_str(), caches[i].key.c_str());
          }
        }
      }
      auto& unsortedCaches = service->bindingsKeysUnsorted;
      for (auto i = 0u; i < unsortedCaches.size(); ++i) {
        if (unsortedCaches[i].enabled && pc.inputs().getPos(unsortedCaches[i].binding.c_str()) >= 0) {
          auto status = service->updateCacheEntryUnsorted(i, pc.inputs().get<TableConsumer>(unsortedCaches[i].binding.c_str())->asArrowTable());
          if (!status.ok()) {
            throw runtime_error_f("failed to update slice cache (unsorted) for %s/%s", unsortedCaches[i].binding.c_str(), unsortedCaches[i].key.c_str());
          }
        }
      } },
    .kind = ServiceKind::Stream};
}

} // namespace o2::framework
