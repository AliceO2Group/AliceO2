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
#include "Framework/DataProcessor.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/DeviceSpec.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/Tracing.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceMetricsHelper.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DevicesManager.h"

#include "CommonMessageBackendsHelpers.h"
#include <Monitoring/Monitoring.h>
#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"

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
  int64_t maxMemory = 2000;
};

static int64_t memLimit = 0;

struct MetricIndices {
  size_t arrowBytesCreated = 0;
  size_t shmOfferConsumed = 0;
  size_t arrowBytesDestroyed = 0;
  size_t arrowMessagesCreated = 0;
  size_t arrowMessagesDestroyed = 0;
  size_t arrowBytesExpired = 0;
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
    results.push_back(indices);
  }
  return results;
}

uint64_t calculateAvailableSharedMemory(ServiceRegistry& registry)
{
  return registry.get<RateLimitConfig>().maxMemory;
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
                       int64_t shmOfferConsumed = 0;
                       int64_t totalBytesDestroyed = 0;
                       int64_t totalBytesExpired = 0;
                       int64_t totalMessagesCreated = 0;
                       int64_t totalMessagesDestroyed = 0;
                       static RateLimitingState currentState = RateLimitingState::UNKNOWN;
                       static auto stateMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "rate-limit-state");
                       static auto totalBytesCreatedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-bytes-created");
                       static auto shmOfferConsumedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "shm-offer-bytes-consumed");
                       static auto unusedOfferedMemoryMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "unusedOfferedMemory");
                       static auto availableSharedMemoryMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "available-shared-memory");
                       static auto offeredSharedMemoryMetric = DeviceMetricsHelper::createNumericMetric<int>(driverMetrics, "offered-shared-memory");
                       static auto totalBytesDestroyedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-bytes-destroyed");
                       static auto totalBytesExpiredMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-bytes-expired");
                       static auto totalMessagesCreatedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-messages-created");
                       static auto totalMessagesDestroyedMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "total-arrow-messages-destroyed");
                       static auto totalBytesDeltaMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "arrow-bytes-delta");
                       static auto totalSignalsMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "aod-reader-signals");
                       static auto signalLatencyMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "aod-signal-latency");
                       static auto skippedSignalsMetric = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "aod-skipped-signals");
                       static auto remainingBytes = DeviceMetricsHelper::createNumericMetric<uint64_t>(driverMetrics, "aod-remaining-bytes");
                       auto& manager = registry.get<DevicesManager>();

                       bool changed = false;
                       bool hasMetrics = false;
                       // Find  the last timestamp when we signaled.
                       static size_t signalIndex = DeviceMetricsHelper::metricIdxByName("aod-reader-signals", driverMetrics);
                       if (signalIndex < driverMetrics.metrics.size()) {
                         MetricInfo& info = driverMetrics.metrics.at(signalIndex);
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
                             auto value = (int64_t)data.at((info.pos - 1) % data.size());
                             totalBytesCreated += value;
                             lastTimestamp = std::max(lastTimestamp, deviceMetrics.timestamps[index][(info.pos - 1) % data.size()]);
                             firstTimestamp = std::min(lastTimestamp, firstTimestamp);
                           }
                         }
                         {
                           size_t index = indices.shmOfferConsumed;
                           if (index < deviceMetrics.metrics.size()) {
                             hasMetrics = true;
                             changed |= deviceMetrics.changed.at(index);
                             MetricInfo info = deviceMetrics.metrics.at(index);
                             auto& data = deviceMetrics.uint64Metrics.at(info.storeIdx);
                             auto value = (int64_t)data.at((info.pos - 1) % data.size());
                             shmOfferConsumed += value;
                             lastTimestamp = std::max(lastTimestamp, deviceMetrics.timestamps[index][(info.pos - 1) % data.size()]);
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
                           size_t index = indices.arrowBytesExpired;
                           if (index < deviceMetrics.metrics.size()) {
                             hasMetrics = true;
                             changed |= deviceMetrics.changed.at(index);
                             MetricInfo info = deviceMetrics.metrics.at(index);
                             auto& data = deviceMetrics.uint64Metrics.at(info.storeIdx);
                             totalBytesExpired += (int64_t)data.at((info.pos - 1) % data.size());
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
                         totalBytesExpiredMetric(driverMetrics, totalBytesExpired, timestamp);
                         shmOfferConsumedMetric(driverMetrics, shmOfferConsumed, timestamp);
                         totalMessagesCreatedMetric(driverMetrics, totalMessagesCreated, timestamp);
                         totalMessagesDestroyedMetric(driverMetrics, totalMessagesDestroyed, timestamp);
                         totalBytesDeltaMetric(driverMetrics, totalBytesCreated - totalBytesExpired - totalBytesDestroyed, timestamp);
                       }
                       bool done = false;
                       static int stateTransitions = 0;
                       static int signalsCount = 0;
                       static int skippedCount = 0;
                       static uint64_t now = 0;
                       now = uv_hrtime();
                       static RateLimitingState lastReportedState = RateLimitingState::UNKNOWN;
                       static uint64_t lastReportTime = 0;
                       static int64_t MAX_SHARED_MEMORY = calculateAvailableSharedMemory(registry);
                       constexpr int64_t MAX_QUANTUM_SHARED_MEMORY = 300;
                       constexpr int64_t MIN_QUANTUM_SHARED_MEMORY = 100;

                       static int64_t availableSharedMemory = MAX_SHARED_MEMORY;
                       static int64_t offeredSharedMemory = 0;
                       static int64_t lastDeviceOffered = 0;
                       /// We loop over the devices, starting from where we stopped last time
                       /// offering 1 GB of shared memory to each reader.
                       int64_t lastCandidate = -1;
                       static int enoughSharedMemoryCount = availableSharedMemory - MIN_QUANTUM_SHARED_MEMORY > 0 ? 1 : 0;
                       static int lowSharedMemoryCount = availableSharedMemory - MIN_QUANTUM_SHARED_MEMORY > 0 ? 0 : 1;
                       int64_t possibleOffer = MIN_QUANTUM_SHARED_MEMORY;
                       for (size_t di = 0; di < specs.size(); di++) {
                         if (availableSharedMemory < possibleOffer) {
                           if (lowSharedMemoryCount == 0) {
                             LOGP(INFO, "We do not have enough shared memory ({}MB) to offer {}MB", availableSharedMemory, possibleOffer);
                           }
                           lowSharedMemoryCount++;
                           enoughSharedMemoryCount = 0;
                           break;
                         } else {
                           if (enoughSharedMemoryCount == 0) {
                             LOGP(INFO, "We are back in a state where we enough shared memory: {}MB", availableSharedMemory);
                           }
                           enoughSharedMemoryCount++;
                           lowSharedMemoryCount = 0;
                         }
                         size_t candidate = (lastDeviceOffered + di) % specs.size();
                         if (specs[candidate].name != "internal-dpl-aod-reader") {
                           continue;
                         }
                         possibleOffer = std::min(MAX_QUANTUM_SHARED_MEMORY, availableSharedMemory);
                         LOGP(info, "Offering {}MB out of {} to {}", possibleOffer, availableSharedMemory, specs[candidate].id);
                         manager.queueMessage(specs[candidate].id.c_str(), fmt::format("/shm-offer {}", possibleOffer).data());
                         availableSharedMemory -= possibleOffer;
                         offeredSharedMemory += possibleOffer;
                         lastCandidate = candidate;
                       }
                       // We had at least a valid candidate, so
                       // next time we offer to the next device.
                       if (lastCandidate >= 0) {
                         lastDeviceOffered = lastCandidate + 1;
                       }

                       // unusedOfferedMemory is the amount of memory which was offered and which we know it was
                       // not used so far. So we need to account for the amount which got actually read (readerBytesCreated)
                       // and the amount which we know was given back.
                       static int64_t lastShmOfferConsumed = 0;
                       static int64_t lastUnusedOfferedMemory = 0;
                       if (shmOfferConsumed != lastShmOfferConsumed) {
                         LOGP(INFO, "Offer consumed so far {}", shmOfferConsumed);
                         lastShmOfferConsumed = shmOfferConsumed;
                       }
                       int unusedOfferedMemory = (offeredSharedMemory - (totalBytesExpired + shmOfferConsumed) / 1000000);
                       if (lastUnusedOfferedMemory != unusedOfferedMemory) {
                         LOGP(INFO, "Unused offer {}", unusedOfferedMemory);
                         lastUnusedOfferedMemory = unusedOfferedMemory;
                       }
                       // availableSharedMemory is the amount of memory which we know is available to be offered.
                       // We subtract the amount which we know was already offered but it's unused and we then balance how
                       // much was created with how much was destroyed.
                       availableSharedMemory = MAX_SHARED_MEMORY + ((totalBytesDestroyed - totalBytesCreated) / 1000000) - unusedOfferedMemory;
                       availableSharedMemoryMetric(driverMetrics, availableSharedMemory, timestamp);
                       unusedOfferedMemoryMetric(driverMetrics, unusedOfferedMemory, timestamp);

                       offeredSharedMemoryMetric(driverMetrics, offeredSharedMemory, timestamp);
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
                           LOGP(INFO, "Message {}/{} is not of kind arrow, therefore we are not accounting its shared memory", dh->dataOrigin, dh->dataDescription);
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
                           LOGP(INFO, "Message {}/{} is forwarded so we are not returning its memory.", dh->dataOrigin, dh->dataDescription);
                           continue;
                         }
                         LOGP(INFO, "Message {}/{} is being deleted. We will return {}MB.", dh->dataOrigin, dh->dataDescription, dh->payloadSize / 1000000.);
                         totalBytes += dh->payloadSize;
                         totalMessages += 1;
                       }
                       arrow->updateBytesDestroyed(totalBytes);
                       LOGP(INFO, "{}MB bytes being given back to reader, totaling {}MB", totalBytes / 1000000., arrow->bytesDestroyed() / 1000000.);
                       arrow->updateMessagesDestroyed(totalMessages);
                       auto& monitoring = ctx.services().get<Monitoring>();
                       monitoring.send(Metric{(uint64_t)arrow->bytesDestroyed(), "arrow-bytes-destroyed"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
                       monitoring.send(Metric{(uint64_t)arrow->messagesDestroyed(), "arrow-messages-destroyed"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
                       monitoring.flushBuffer();
                     },
                     nullptr,
                     nullptr,
                     [](ServiceRegistry& registry, boost::program_options::variables_map const& vm) {
                       auto config = new RateLimitConfig{};
                       int readers = std::stoll(vm["readers"].as<std::string>());
                       long long int minReaderMemory = readers * 500;
                       if (vm.count("aod-memory-rate-limit")) {
                         config->maxMemory = std::max(minReaderMemory, std::stoll(vm["aod-memory-rate-limit"].as<std::string>()) / 1000000);
                       }
                       static bool once = false;
                       // Until we guarantee this is called only once...
                       if (!once) {
                         LOGP(INFO, "Rate limiting set up at {}MB distributed over {} readers", config->maxMemory, readers);
                         registry.registerService(ServiceRegistryHelpers::handleForService<RateLimitConfig>(config));
                         once = true;
                       }
                     },
                     ServiceKind::Global};
}

} // namespace o2::framework
