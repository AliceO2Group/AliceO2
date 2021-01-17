// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/CommonServices.h"
#include "Framework/ParallelContext.h"
#include "Framework/ControlService.h"
#include "Framework/DriverClient.h"
#include "Framework/CallbackService.h"
#include "Framework/TimesliceIndex.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/DeviceSpec.h"
#include "Framework/LocalRootFileService.h"
#include "Framework/DataRelayer.h"
#include "Framework/Signpost.h"
#include "Framework/DataProcessingStats.h"
#include "Framework/CommonMessageBackends.h"
#include "Framework/DanglingContext.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/Tracing.h"
#include "Framework/Monitoring.h"
#include "TextDriverClient.h"
#include "../src/DataProcessingStatus.h"

#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>
#include <Monitoring/MonitoringFactory.h>
#include <InfoLogger/InfoLogger.hxx>

#include <options/FairMQProgOptions.h>

#include <cstdlib>

using AliceO2::InfoLogger::InfoLogger;
using AliceO2::InfoLogger::InfoLoggerContext;
using o2::configuration::ConfigurationFactory;
using o2::configuration::ConfigurationInterface;
using o2::monitoring::Monitoring;
using o2::monitoring::MonitoringFactory;
using Metric = o2::monitoring::Metric;
using Key = o2::monitoring::tags::Key;
using Value = o2::monitoring::tags::Value;

namespace o2::framework
{

/// This is a global service because read only
template <>
struct ServiceKindExtractor<InfoLoggerContext> {
  constexpr static ServiceKind kind = ServiceKind::Global;
};

o2::framework::ServiceSpec CommonServices::monitoringSpec()
{
  return ServiceSpec{"monitoring",
                     [](ServiceRegistry&, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
                       void* service = MonitoringFactory::Get(options.GetPropertyAsString("monitoring-backend")).release();
                       return ServiceHandle{TypeIdHelpers::uniqueId<Monitoring>(), service};
                     },
                     noConfiguration(),
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
                     nullptr,
                     nullptr,
                     nullptr,
                     ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::infologgerContextSpec()
{
  return ServiceSpec{"infologger-contex",
                     simpleServiceInit<InfoLoggerContext, InfoLoggerContext>(),
                     noConfiguration(),
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
                     nullptr,
                     nullptr,
                     nullptr,
                     ServiceKind::Serial};
}

// Creates the sink for FairLogger / InfoLogger integration
auto createInfoLoggerSinkHelper(InfoLogger* logger, InfoLoggerContext* ctx)
{
  return [logger,
          ctx](const std::string& content, const fair::LogMetaData& metadata) {
    // translate FMQ metadata
    InfoLogger::InfoLogger::Severity severity = InfoLogger::Severity::Undefined;
    int level = InfoLogger::undefinedMessageOption.level;

    if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::nolog)) {
      // discard
      return;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::fatal)) {
      severity = InfoLogger::Severity::Fatal;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::error)) {
      severity = InfoLogger::Severity::Error;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::warn)) {
      severity = InfoLogger::Severity::Warning;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::state)) {
      severity = InfoLogger::Severity::Info;
      level = 10;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::info)) {
      severity = InfoLogger::Severity::Info;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug)) {
      severity = InfoLogger::Severity::Debug;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug1)) {
      severity = InfoLogger::Severity::Debug;
      level = 10;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug2)) {
      severity = InfoLogger::Severity::Debug;
      level = 20;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug3)) {
      severity = InfoLogger::Severity::Debug;
      level = 30;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug4)) {
      severity = InfoLogger::Severity::Debug;
      level = 40;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::trace)) {
      severity = InfoLogger::Severity::Debug;
      level = 50;
    }

    InfoLogger::InfoLoggerMessageOption opt = {
      severity,
      level,
      InfoLogger::undefinedMessageOption.errorCode,
      metadata.file.c_str(),
      atoi(metadata.line.c_str())};

    if (logger) {
      logger->log(opt, *ctx, "DPL: %s", content.c_str());
    }
  };
};

o2::framework::ServiceSpec CommonServices::infologgerSpec()
{
  return ServiceSpec{"infologger",
                     [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
                       auto infoLoggerMode = options.GetPropertyAsString("infologger-mode");
                       if (infoLoggerMode != "") {
                         setenv("INFOLOGGER_MODE", infoLoggerMode.c_str(), 1);
                       }
                       auto infoLoggerService = new InfoLogger;
                       auto infoLoggerContext = &services.get<InfoLoggerContext>();

                       auto infoLoggerSeverity = options.GetPropertyAsString("infologger-severity");
                       if (infoLoggerSeverity != "") {
                         fair::Logger::AddCustomSink("infologger", infoLoggerSeverity, createInfoLoggerSinkHelper(infoLoggerService, infoLoggerContext));
                       }
                       return ServiceHandle{TypeIdHelpers::uniqueId<InfoLogger>(), infoLoggerService};
                     },
                     noConfiguration(),
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
                     nullptr,
                     nullptr,
                     nullptr,
                     ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::configurationSpec()
{
  return ServiceSpec{
    "configuration",
    [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto backend = options.GetPropertyAsString("configuration");
      if (backend == "command-line") {
        return ServiceHandle{0, nullptr};
      }
      return ServiceHandle{TypeIdHelpers::uniqueId<ConfigurationInterface>(),
                           ConfigurationFactory::getConfiguration(backend).release()};
    },
    noConfiguration(),
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
    nullptr,
    nullptr,
    nullptr,
    ServiceKind::Global};
}

o2::framework::ServiceSpec CommonServices::driverClientSpec()
{
  return ServiceSpec{
    "driverClient",
    [](ServiceRegistry& services, DeviceState& state, fair::mq::ProgOptions& options) -> ServiceHandle {
      return ServiceHandle{TypeIdHelpers::uniqueId<DriverClient>(),
                           new TextDriverClient(services, state)};
    },
    noConfiguration(),
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
    nullptr,
    nullptr,
    nullptr,
    ServiceKind::Global};
}

o2::framework::ServiceSpec CommonServices::controlSpec()
{
  return ServiceSpec{
    "control",
    [](ServiceRegistry& services, DeviceState& state, fair::mq::ProgOptions& options) -> ServiceHandle {
      return ServiceHandle{TypeIdHelpers::uniqueId<ControlService>(),
                           new ControlService(services, state)};
    },
    noConfiguration(),
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
    nullptr,
    nullptr,
    nullptr,
    ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::rootFileSpec()
{
  return ServiceSpec{
    "localrootfile",
    simpleServiceInit<LocalRootFileService, LocalRootFileService>(),
    noConfiguration(),
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
    nullptr,
    nullptr,
    nullptr,
    ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::parallelSpec()
{
  return ServiceSpec{
    "parallel",
    [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto& spec = services.get<DeviceSpec const>();
      return ServiceHandle{TypeIdHelpers::uniqueId<ParallelContext>(),
                           new ParallelContext(spec.rank, spec.nSlots)};
    },
    noConfiguration(),
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
    nullptr,
    nullptr,
    nullptr,
    ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::timesliceIndex()
{
  return ServiceSpec{
    "timesliceindex",
    simpleServiceInit<TimesliceIndex, TimesliceIndex>(),
    noConfiguration(),
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
    nullptr,
    nullptr,
    nullptr,
    ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::callbacksSpec()
{
  return ServiceSpec{
    "callbacks",
    simpleServiceInit<CallbackService, CallbackService>(),
    noConfiguration(),
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
    nullptr,
    nullptr,
    nullptr,
    ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::dataRelayer()
{
  return ServiceSpec{
    "datarelayer",
    [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto& spec = services.get<DeviceSpec const>();
      return ServiceHandle{TypeIdHelpers::uniqueId<DataRelayer>(),
                           new DataRelayer(spec.completionPolicy,
                                           spec.inputs,
                                           services.get<Monitoring>(),
                                           services.get<TimesliceIndex>())};
    },
    noConfiguration(),
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
    nullptr,
    nullptr,
    nullptr,
    ServiceKind::Serial};
}

struct TracingInfrastructure {
  int processingCount;
};

o2::framework::ServiceSpec CommonServices::tracingSpec()
{
  return ServiceSpec{
    "tracing",
    [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      return ServiceHandle{TypeIdHelpers::uniqueId<TracingInfrastructure>(), new TracingInfrastructure()};
    },
    noConfiguration(),
    [](ProcessingContext&, void* service) {
      TracingInfrastructure* t = reinterpret_cast<TracingInfrastructure*>(service);
      t->processingCount += 1;
    },
    [](ProcessingContext&, void* service) {
      TracingInfrastructure* t = reinterpret_cast<TracingInfrastructure*>(service);
      t->processingCount += 1;
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
    nullptr,
    ServiceKind::Serial};
}

// FIXME: allow configuring the default number of threads per device
//        This should probably be done by overriding the preFork
//        callback and using the boost program options there to
//        get the default number of threads.
o2::framework::ServiceSpec CommonServices::threadPool(int numWorkers)
{
  return ServiceSpec{
    "threadpool",
    [numWorkers](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      ThreadPool* pool = new ThreadPool();
      pool->poolSize = numWorkers;
      return ServiceHandle{TypeIdHelpers::uniqueId<ThreadPool>(), pool};
    },
    [numWorkers](InitContext&, void* service) -> void* {
      ThreadPool* t = reinterpret_cast<ThreadPool*>(service);
      t->poolSize = numWorkers;
      return service;
    },
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    [numWorkers](ServiceRegistry& service) -> void {
      auto numWorkersS = std::to_string(numWorkers);
      setenv("UV_THREADPOOL_SIZE", numWorkersS.c_str(), 0);
    },
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    ServiceKind::Serial};
}

namespace
{
/// This will send metrics for the relayer at regular intervals of
/// 5 seconds, in order to avoid overloading the system.
auto sendRelayerMetrics(ServiceRegistry& registry, DataProcessingStats& stats) -> void
{
  if (stats.beginIterationTimestamp - stats.lastSlowMetricSentTimestamp < 5000) {
    return;
  }
  ZoneScopedN("send metrics");
  auto& relayerStats = registry.get<DataRelayer>().getStats();
  auto& monitoring = registry.get<Monitoring>();

  O2_SIGNPOST_START(MonitoringStatus::ID, MonitoringStatus::SEND, 0, 0, O2_SIGNPOST_BLUE);

  monitoring.send(Metric{(int)relayerStats.malformedInputs, "malformed_inputs"}.addTag(Key::Subsystem, Value::DPL));
  monitoring.send(Metric{(int)relayerStats.droppedComputations, "dropped_computations"}.addTag(Key::Subsystem, Value::DPL));
  monitoring.send(Metric{(int)relayerStats.droppedIncomingMessages, "dropped_incoming_messages"}.addTag(Key::Subsystem, Value::DPL));
  monitoring.send(Metric{(int)relayerStats.relayedMessages, "relayed_messages"}.addTag(Key::Subsystem, Value::DPL));

  monitoring.send(Metric{(int)stats.pendingInputs, "inputs/relayed/pending"}.addTag(Key::Subsystem, Value::DPL));
  monitoring.send(Metric{(int)stats.incomplete, "inputs/relayed/incomplete"}.addTag(Key::Subsystem, Value::DPL));
  monitoring.send(Metric{(int)stats.inputParts, "inputs/relayed/total"}.addTag(Key::Subsystem, Value::DPL));
  monitoring.send(Metric{stats.lastElapsedTimeMs, "elapsed_time_ms"}.addTag(Key::Subsystem, Value::DPL));
  monitoring.send(Metric{stats.lastTotalProcessedSize, "processed_input_size_byte"}
                    .addTag(Key::Subsystem, Value::DPL));
  monitoring.send(Metric{(stats.lastTotalProcessedSize.load() / (stats.lastElapsedTimeMs.load() ? stats.lastElapsedTimeMs.load() : 1) / 1000),
                         "processing_rate_mb_s"}
                    .addTag(Key::Subsystem, Value::DPL));
  monitoring.send(Metric{stats.lastLatency.minLatency, "min_input_latency_ms"}
                    .addTag(Key::Subsystem, Value::DPL));
  monitoring.send(Metric{stats.lastLatency.maxLatency, "max_input_latency_ms"}
                    .addTag(Key::Subsystem, Value::DPL));
  monitoring.send(Metric{(stats.lastTotalProcessedSize / (stats.lastLatency.maxLatency ? stats.lastLatency.maxLatency : 1) / 1000), "input_rate_mb_s"}
                    .addTag(Key::Subsystem, Value::DPL));

  stats.lastSlowMetricSentTimestamp.store(stats.beginIterationTimestamp.load());
  O2_SIGNPOST_END(MonitoringStatus::ID, MonitoringStatus::SEND, 0, 0, O2_SIGNPOST_BLUE);
};

/// This will flush metrics only once every second.
auto flushMetrics(ServiceRegistry& registry, DataProcessingStats& stats) -> void
{
  if (stats.beginIterationTimestamp - stats.lastMetricFlushedTimestamp < 1000) {
    return;
  }
  ZoneScopedN("flush metrics");
  auto& monitoring = registry.get<Monitoring>();
  auto& relayer = registry.get<DataRelayer>();

  O2_SIGNPOST_START(MonitoringStatus::ID, MonitoringStatus::FLUSH, 0, 0, O2_SIGNPOST_RED);
  // Send all the relevant metrics for the relayer to update the GUI
  // FIXME: do a delta with the previous version if too many metrics are still
  // sent...
  for (size_t si = 0; si < stats.statesSize.load(); ++si) {
    auto value = std::atomic_load_explicit(&stats.relayerState[si], std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    monitoring.send({value, fmt::format("data_relayer/{}", si)});
  }
  relayer.sendContextState();
  monitoring.flushBuffer();
  stats.lastMetricFlushedTimestamp.store(stats.beginIterationTimestamp.load());
  O2_SIGNPOST_END(MonitoringStatus::ID, MonitoringStatus::FLUSH, 0, 0, O2_SIGNPOST_RED);
};
} // namespace

o2::framework::ServiceSpec CommonServices::dataProcessingStats()
{
  return ServiceSpec{
    "data-processing-stats",
    [](ServiceRegistry& services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      DataProcessingStats* stats = new DataProcessingStats();
      return ServiceHandle{TypeIdHelpers::uniqueId<DataProcessingStats>(), stats};
    },
    noConfiguration(),
    nullptr,
    nullptr,
    [](DanglingContext& context, void* service) {
      DataProcessingStats* stats = (DataProcessingStats*)service;
      sendRelayerMetrics(context.services(), *stats);
      flushMetrics(context.services(), *stats);
    },
    [](DanglingContext& context, void* service) {
      DataProcessingStats* stats = (DataProcessingStats*)service;
      sendRelayerMetrics(context.services(), *stats);
      flushMetrics(context.services(), *stats);
    },
    [](EndOfStreamContext& context, void* service) {
      DataProcessingStats* stats = (DataProcessingStats*)service;
      sendRelayerMetrics(context.services(), *stats);
      flushMetrics(context.services(), *stats);
    },
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    ServiceKind::Serial};
}

std::vector<ServiceSpec> CommonServices::defaultServices(int numThreads)
{
  std::vector<ServiceSpec> specs{
    timesliceIndex(),
    driverClientSpec(),
    monitoringSpec(),
    infologgerContextSpec(),
    infologgerSpec(),
    configurationSpec(),
    controlSpec(),
    rootFileSpec(),
    parallelSpec(),
    callbacksSpec(),
    dataRelayer(),
    dataProcessingStats(),
    CommonMessageBackends::fairMQBackendSpec(),
    CommonMessageBackends::arrowBackendSpec(),
    CommonMessageBackends::stringBackendSpec(),
    CommonMessageBackends::rawBufferBackendSpec()};
  if (numThreads) {
    specs.push_back(threadPool(numThreads));
  }
  return specs;
}

} // namespace o2::framework
