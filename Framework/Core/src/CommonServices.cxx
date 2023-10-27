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
#include "Framework/CommonServices.h"
#include "Framework/AsyncQueue.h"
#include "Framework/ParallelContext.h"
#include "Framework/ControlService.h"
#include "Framework/DriverClient.h"
#include "Framework/CallbackService.h"
#include "Framework/ServiceSpec.h"
#include "Framework/TimesliceIndex.h"
#include "Framework/DataTakingContext.h"
#include "Framework/DataSender.h"
#include "Framework/ServiceRegistryRef.h"
#include "Framework/DeviceSpec.h"
#include "Framework/LocalRootFileService.h"
#include "Framework/DataRelayer.h"
#include "Framework/Signpost.h"
#include "Framework/DataProcessingStats.h"
#include "Framework/DataProcessingStates.h"
#include "Framework/TimingHelpers.h"
#include "Framework/CommonMessageBackends.h"
#include "Framework/DanglingContext.h"
#include "Framework/DataProcessingHelpers.h"
#include "InputRouteHelpers.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/RunningWorkflowInfo.h"
#include "Framework/Tracing.h"
#include "Framework/Monitoring.h"
#include "Framework/AsyncQueue.h"
#include "Framework/Plugins.h"
#include "Framework/DeviceContext.h"
#include "Framework/DataProcessingContext.h"
#include "Framework/StreamContext.h"
#include "Framework/DeviceState.h"
#include "Framework/DeviceConfig.h"
#include "Framework/DefaultsHelpers.h"

#include "TextDriverClient.h"
#include "WSDriverClient.h"
#include "HTTPParser.h"
#include "../src/DataProcessingStatus.h"
#include "DecongestionService.h"
#include "ArrowSupport.h"
#include "DPLMonitoringBackend.h"
#include "Headers/STFHeader.h"
#include "Headers/DataHeader.h"

#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>
#include <Monitoring/MonitoringFactory.h>

#include <fairmq/Device.h>
#include <fairmq/shmem/Monitor.h>
#include <fairmq/shmem/Common.h>
#include <fairmq/ProgOptions.h>
#include <uv.h>

#include <cstdlib>
#include <cstring>

using o2::configuration::ConfigurationFactory;
using o2::configuration::ConfigurationInterface;
using o2::monitoring::Monitoring;
using o2::monitoring::MonitoringFactory;
using Metric = o2::monitoring::Metric;
using Key = o2::monitoring::tags::Key;
using Value = o2::monitoring::tags::Value;

// This is to allow C++20 aggregate initialisation
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

namespace o2::framework
{

#define MONITORING_QUEUE_SIZE 100
o2::framework::ServiceSpec CommonServices::monitoringSpec()
{
  return ServiceSpec{
    .name = "monitoring",
    .init = [](ServiceRegistryRef registry, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      void* service = nullptr;
      bool isWebsocket = strncmp(options.GetPropertyAsString("driver-client-backend").c_str(), "ws://", 4) == 0;
      bool isDefault = options.GetPropertyAsString("monitoring-backend") == "default";
      bool useDPL = (isWebsocket && isDefault) || options.GetPropertyAsString("monitoring-backend") == "dpl://";
      o2::monitoring::Monitoring* monitoring;
      if (useDPL) {
        monitoring = new Monitoring();
        auto dplBackend = std::make_unique<DPLMonitoringBackend>(registry);
        (dynamic_cast<o2::monitoring::Backend*>(dplBackend.get()))->setVerbosity(o2::monitoring::Verbosity::Debug);
        monitoring->addBackend(std::move(dplBackend));
      } else {
        auto backend = isDefault ? "infologger://" : options.GetPropertyAsString("monitoring-backend");
        monitoring = MonitoringFactory::Get(backend).release();
      }
      service = monitoring;
      monitoring->enableBuffering(MONITORING_QUEUE_SIZE);
      assert(registry.get<DeviceSpec const>().name.empty() == false);
      monitoring->addGlobalTag("pipeline_id", std::to_string(registry.get<DeviceSpec const>().inputTimesliceId));
      monitoring->addGlobalTag("dataprocessor_name", registry.get<DeviceSpec const>().name);
      monitoring->addGlobalTag("dpl_instance", options.GetPropertyAsString("shm-segment-id"));
      return ServiceHandle{TypeIdHelpers::uniqueId<Monitoring>(), service};
    },
    .configure = noConfiguration(),
    .start = [](ServiceRegistryRef services, void* service) {
      auto* monitoring = (o2::monitoring::Monitoring*)service;

      auto extRunNumber = services.get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("runNumber", "unspecified");
      if (extRunNumber == "unspecified") {
        return;
      }
      try {
        monitoring->setRunNumber(std::stoul(extRunNumber));
      } catch (...) {
      } },
    .exit = [](ServiceRegistryRef registry, void* service) {
                       auto* monitoring = reinterpret_cast<Monitoring*>(service);
                       delete monitoring; },
    .kind = ServiceKind::Serial};
}

// An asyncronous service that executes actions in at the end of the data processing
o2::framework::ServiceSpec CommonServices::asyncQueue()
{
  return ServiceSpec{
    .name = "async-queue",
    .init = simpleServiceInit<AsyncQueue, AsyncQueue>(),
    .configure = noConfiguration(),
    .kind = ServiceKind::Serial};
}

// Make it a service so that it can be used easily from the analysis
// FIXME: Moreover, it makes sense that this will be duplicated on a per thread
// basis when we get to it.
o2::framework::ServiceSpec CommonServices::timingInfoSpec()
{
  return ServiceSpec{
    .name = "timing-info",
    .uniqueId = simpleServiceId<TimingInfo>(),
    .init = simpleServiceInit<TimingInfo, TimingInfo, ServiceKind::Stream>(),
    .configure = noConfiguration(),
    .kind = ServiceKind::Stream};
}

o2::framework::ServiceSpec CommonServices::streamContextSpec()
{
  return ServiceSpec{
    .name = "stream-context",
    .uniqueId = simpleServiceId<StreamContext>(),
    .init = simpleServiceInit<StreamContext, StreamContext, ServiceKind::Stream>(),
    .configure = noConfiguration(),
    .kind = ServiceKind::Stream};
}

o2::framework::ServiceSpec CommonServices::datatakingContextSpec()
{
  return ServiceSpec{
    .name = "datataking-contex",
    .uniqueId = simpleServiceId<DataTakingContext>(),
    .init = simpleServiceInit<DataTakingContext, DataTakingContext, ServiceKind::Stream>(),
    .configure = noConfiguration(),
    .preProcessing = [](ProcessingContext& processingContext, void* service) {
      auto& context = processingContext.services().get<DataTakingContext>();
      for (auto const& ref : processingContext.inputs()) {
        const o2::framework::DataProcessingHeader *dph = o2::header::get<DataProcessingHeader*>(ref.header);
        const auto* dh = o2::header::get<o2::header::DataHeader*>(ref.header);
        if (!dph || !dh) {
          continue;
        }
        context.runNumber = fmt::format("{}", dh->runNumber);
        break;
      } },
    // Notice this will be executed only once, because the service is declared upfront.
    .start = [](ServiceRegistryRef services, void* service) {
      auto& context = services.get<DataTakingContext>();

      context.deploymentMode = DefaultsHelpers::deploymentMode();

      auto extRunNumber = services.get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("runNumber", "unspecified");
      if (extRunNumber != "unspecified" || context.runNumber == "0") {
        context.runNumber = extRunNumber;
      }
      auto extLHCPeriod = services.get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("lhc_period", "unspecified");
      if (extLHCPeriod != "unspecified") {
        context.lhcPeriod = extLHCPeriod;
      } else {
        static const char* months[12] = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"};
        time_t now = time(nullptr);
        auto ltm = gmtime(&now);
        context.lhcPeriod = months[ltm->tm_mon];
        LOG(info) << "LHCPeriod is not available, using current month " << context.lhcPeriod;
      }

      auto extRunType = services.get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("run_type", "unspecified");
      if (extRunType != "unspecified") {
        context.runType = extRunType;
      }
      auto extEnvId = services.get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("environment_id", "unspecified");
      if (extEnvId != "unspecified") {
        context.envId = extEnvId;
      }
      auto extDetectors = services.get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("detectors", "unspecified");
      if (extDetectors != "unspecified") {
        context.detectors = extDetectors;
      }
      auto forcedRaw = services.get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("force_run_as_raw", "false");
      context.forcedRaw = forcedRaw == "true"; },
    .kind = ServiceKind::Stream};
}

struct MissingService {
};

o2::framework::ServiceSpec CommonServices::configurationSpec()
{
  return ServiceSpec{
    .name = "configuration",
    .init = [](ServiceRegistryRef services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto backend = options.GetPropertyAsString("configuration");
      if (backend == "command-line") {
        return ServiceHandle{0, nullptr};
      }
      return ServiceHandle{TypeIdHelpers::uniqueId<ConfigurationInterface>(),
                           ConfigurationFactory::getConfiguration(backend).release()};
    },
    .configure = noConfiguration(),
    .driverStartup = [](ServiceRegistryRef registry, DeviceConfig const& dc) {
      if (dc.options.count("configuration") == 0) {
        registry.registerService(ServiceHandle{0, nullptr});
        return;
      }
      auto backend = dc.options["configuration"].as<std::string>();
      registry.registerService(ServiceHandle{TypeIdHelpers::uniqueId<ConfigurationInterface>(),
                                             ConfigurationFactory::getConfiguration(backend).release()}); },
    .kind = ServiceKind::Global};
}

o2::framework::ServiceSpec CommonServices::driverClientSpec()
{
  return ServiceSpec{
    .name = "driverClient",
    .init = [](ServiceRegistryRef services, DeviceState& state, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto backend = options.GetPropertyAsString("driver-client-backend");
      if (backend == "stdout://") {
        return ServiceHandle{TypeIdHelpers::uniqueId<DriverClient>(),
                             new TextDriverClient(services, state)};
      }
      auto [ip, port] = o2::framework::parse_websocket_url(backend.c_str());
      return ServiceHandle{TypeIdHelpers::uniqueId<DriverClient>(),
                           new WSDriverClient(services, ip.c_str(), port)};
    },
    .configure = noConfiguration(),
    .kind = ServiceKind::Global};
}

o2::framework::ServiceSpec CommonServices::controlSpec()
{
  return ServiceSpec{
    .name = "control",
    .init = [](ServiceRegistryRef services, DeviceState& state, fair::mq::ProgOptions& options) -> ServiceHandle {
      return ServiceHandle{TypeIdHelpers::uniqueId<ControlService>(),
                           new ControlService(services, state)};
    },
    .configure = noConfiguration(),
    .kind = ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::rootFileSpec()
{
  return ServiceSpec{
    .name = "localrootfile",
    .init = simpleServiceInit<LocalRootFileService, LocalRootFileService>(),
    .configure = noConfiguration(),
    .kind = ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::parallelSpec()
{
  return ServiceSpec{
    .name = "parallel",
    .init = [](ServiceRegistryRef services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto& spec = services.get<DeviceSpec const>();
      return ServiceHandle{TypeIdHelpers::uniqueId<ParallelContext>(),
                           new ParallelContext(spec.rank, spec.nSlots)};
    },
    .configure = noConfiguration(),
    .kind = ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::timesliceIndex()
{
  return ServiceSpec{
    .name = "timesliceindex",
    .init = [](ServiceRegistryRef services, DeviceState& state, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto& spec = services.get<DeviceSpec const>();
      return ServiceHandle{TypeIdHelpers::uniqueId<TimesliceIndex>(),
                           new TimesliceIndex(InputRouteHelpers::maxLanes(spec.inputs), state.inputChannelInfos)};
    },
    .configure = noConfiguration(),
    .kind = ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::callbacksSpec()
{
  return ServiceSpec{
    .name = "callbacks",
    .init = simpleServiceInit<CallbackService, CallbackService>(),
    .configure = noConfiguration(),
    .kind = ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::dataRelayer()
{
  return ServiceSpec{
    .name = "datarelayer",
    .init = [](ServiceRegistryRef services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto& spec = services.get<DeviceSpec const>();
      return ServiceHandle{TypeIdHelpers::uniqueId<DataRelayer>(),
                           new DataRelayer(spec.completionPolicy,
                                           spec.inputs,
                                           services.get<TimesliceIndex>(),
                                           services)};
    },
    .configure = noConfiguration(),
    .kind = ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::dataSender()
{
  return ServiceSpec{
    .name = "datasender",
    .init = [](ServiceRegistryRef services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      return ServiceHandle{TypeIdHelpers::uniqueId<DataSender>(),
                           new DataSender(services)};
    },
    .configure = noConfiguration(),
    .preProcessing = [](ProcessingContext&, void* service) {
      auto& dataSender = *reinterpret_cast<DataSender*>(service);
      dataSender.reset(); },
    .postDispatching = [](ProcessingContext& ctx, void* service) {
      auto& dataSender = *reinterpret_cast<DataSender*>(service);
      // If the quit was requested, the post dispatching can still happen
      // but with an empty set of data.
      if (ctx.services().get<DeviceState>().quitRequested == false) {
        dataSender.verifyMissingSporadic();
      } },
    .kind = ServiceKind::Serial};
}

struct TracingInfrastructure {
  int processingCount;
};

o2::framework::ServiceSpec CommonServices::tracingSpec()
{
  return ServiceSpec{
    .name = "tracing",
    .init = [](ServiceRegistryRef, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
      return ServiceHandle{.hash = TypeIdHelpers::uniqueId<TracingInfrastructure>(),
                           .instance = new TracingInfrastructure(),
                           .kind = ServiceKind::Serial};
    },
    .configure = noConfiguration(),
    .preProcessing = [](ProcessingContext&, void* service) {
      auto* t = reinterpret_cast<TracingInfrastructure*>(service);
      t->processingCount += 1; },
    .postProcessing = [](ProcessingContext&, void* service) {
      auto* t = reinterpret_cast<TracingInfrastructure*>(service);
      t->processingCount += 1; },
    .kind = ServiceKind::Serial};
}

struct CCDBSupport {
};

// CCDB Support service
o2::framework::ServiceSpec CommonServices::ccdbSupportSpec()
{
  return ServiceSpec{
    .name = "ccdb-support",
    .init = [](ServiceRegistryRef services, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
      // iterate on all the outputs matchers
      auto& spec = services.get<DeviceSpec const>();
      for (auto& output : spec.outputs) {
        if (DataSpecUtils::match(output.matcher, ConcreteDataTypeMatcher{"FLP", "DISTSUBTIMEFRAME"})) {
          LOGP(debug, "Optional inputs support enabled");
          return ServiceHandle{.hash = TypeIdHelpers::uniqueId<CCDBSupport>(), .instance = new CCDBSupport, .kind = ServiceKind::Serial};
        }
      }
      return ServiceHandle{.hash = TypeIdHelpers::uniqueId<CCDBSupport>(), .instance = nullptr, .kind = ServiceKind::Serial};
    },
    .configure = noConfiguration(),
    .finaliseOutputs = [](ProcessingContext& pc, void* service) {
      if (!service) {
        return;
      }
      if (pc.outputs().countDeviceOutputs(true) == 0) {
        LOGP(debug, "We are w/o outputs, do not automatically add DISTSUBTIMEFRAME to outgoing messages");
        return;
      }
      auto& timingInfo = pc.services().get<TimingInfo>();

      // For any output that is a FLP/DISTSUBTIMEFRAME with subspec != 0,
      // we create a new message.
      InputSpec matcher{"matcher", ConcreteDataTypeMatcher{"FLP", "DISTSUBTIMEFRAME"}};
      for (auto& output : pc.services().get<DeviceSpec const>().outputs) {
        if ((output.timeslice % output.maxTimeslices) != 0) {
          continue;
        }
        if (DataSpecUtils::match(output.matcher, ConcreteDataTypeMatcher{"FLP", "DISTSUBTIMEFRAME"})) {
          auto concrete = DataSpecUtils::asConcreteDataMatcher(output.matcher);
          if (concrete.subSpec == 0) {
            continue;
          }
          auto& stfDist = pc.outputs().make<o2::header::STFHeader>(Output{concrete.origin, concrete.description, concrete.subSpec, output.matcher.lifetime});
          stfDist.id = timingInfo.timeslice;
          stfDist.firstOrbit = timingInfo.firstTForbit;
          stfDist.runNumber = timingInfo.runNumber;
        }
      } },
    .kind = ServiceKind::Global};
}

// Decongestion service
// If we do not have any Timeframe input, it means we must be creating timeslices
// in order and that we should propagate the oldest possible timeslice at the end
// of each processing step.
o2::framework::ServiceSpec CommonServices::decongestionSpec()
{
  return ServiceSpec{
    .name = "decongestion",
    .init = [](ServiceRegistryRef services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto* decongestion = new DecongestionService();
      for (auto& input : services.get<DeviceSpec const>().inputs) {
        if (input.matcher.lifetime == Lifetime::Timeframe) {
          LOGP(detail, "Found a Timeframe input, we cannot update the oldest possible timeslice");
          decongestion->isFirstInTopology = false;
          break;
        }
      }
      auto& queue = services.get<AsyncQueue>();
      decongestion->oldestPossibleTimesliceTask = AsyncQueueHelpers::create(queue, {"oldest-possible-timeslice", 100});
      return ServiceHandle{TypeIdHelpers::uniqueId<DecongestionService>(), decongestion, ServiceKind::Serial};
    },
    .postForwarding = [](ProcessingContext& ctx, void* service) {
      auto* decongestion = reinterpret_cast<DecongestionService*>(service);
      if (decongestion->isFirstInTopology == false) {
        LOGP(debug, "We are not the first in the topology, do not update the oldest possible timeslice");
        return;
      }
      auto& timesliceIndex = ctx.services().get<TimesliceIndex>();
      auto& relayer = ctx.services().get<DataRelayer>();
      timesliceIndex.updateOldestPossibleOutput();
      auto& proxy = ctx.services().get<FairMQDeviceProxy>();
      auto oldestPossibleOutput = relayer.getOldestPossibleOutput();
      if (decongestion->nextEnumerationTimesliceRewinded && decongestion->nextEnumerationTimeslice < oldestPossibleOutput.timeslice.value) {
        LOGP(detail, "Not sending oldestPossible if nextEnumerationTimeslice was rewinded");
        return;
      }

      if (decongestion->lastTimeslice && oldestPossibleOutput.timeslice.value == decongestion->lastTimeslice) {
        LOGP(debug, "Not sending already sent value");
        return;
      }
      if (oldestPossibleOutput.timeslice.value < decongestion->lastTimeslice) {
        LOGP(error, "We are trying to send an oldest possible timeslice {} that is older than the last one we already sent {}",
             oldestPossibleOutput.timeslice.value, decongestion->lastTimeslice);
        return;
      }

      LOGP(debug, "Broadcasting oldest possible output {} due to {} ({})", oldestPossibleOutput.timeslice.value,
           oldestPossibleOutput.slot.index == -1 ? "channel" : "slot",
           oldestPossibleOutput.slot.index == -1 ? oldestPossibleOutput.channel.value : oldestPossibleOutput.slot.index);
      if (decongestion->orderedCompletionPolicyActive) {
        auto oldNextTimeslice = decongestion->nextTimeslice;
        decongestion->nextTimeslice = std::max(decongestion->nextTimeslice, (int64_t)oldestPossibleOutput.timeslice.value);
        if (oldNextTimeslice != decongestion->nextTimeslice) {
          LOGP(error, "Some Lifetime::Timeframe data got dropped starting at {}", oldNextTimeslice);
          timesliceIndex.rescan();
        }
      }
      DataProcessingHelpers::broadcastOldestPossibleTimeslice(proxy, oldestPossibleOutput.timeslice.value);

      for (int fi = 0; fi < proxy.getNumForwardChannels(); fi++) {
        auto& info = proxy.getForwardChannelInfo(ChannelIndex{fi});
        auto& state = proxy.getForwardChannelState(ChannelIndex{fi});
        // TODO: this we could cache in the proxy at the bind moment.
        if (info.channelType != ChannelAccountingType::DPL) {
          LOG(debug) << "Skipping channel";
          continue;
        }
        if (DataProcessingHelpers::sendOldestPossibleTimeframe(info, state, oldestPossibleOutput.timeslice.value)) {
          LOGP(debug, "Forwarding to channel {} oldest possible timeslice {}, prio 20", info.name, oldestPossibleOutput.timeslice.value);
        }
      }
      decongestion->lastTimeslice = oldestPossibleOutput.timeslice.value; },
    .domainInfoUpdated = [](ServiceRegistryRef services, size_t oldestPossibleTimeslice, ChannelIndex channel) {
      auto& decongestion = services.get<DecongestionService>();
      auto& relayer = services.get<DataRelayer>();
      auto& timesliceIndex = services.get<TimesliceIndex>();
      auto& proxy = services.get<FairMQDeviceProxy>();
      LOGP(debug, "Received oldest possible timeframe {} from channel {}", oldestPossibleTimeslice, channel.value);
      relayer.setOldestPossibleInput({oldestPossibleTimeslice}, channel);
      timesliceIndex.updateOldestPossibleOutput();
      auto oldestPossibleOutput = relayer.getOldestPossibleOutput();

      if (oldestPossibleOutput.timeslice.value == decongestion.lastTimeslice) {
        LOGP(debug, "Not sending already sent value");
        return;
      }
      if (oldestPossibleOutput.timeslice.value < decongestion.lastTimeslice) {
        LOGP(error, "We are trying to send an oldest possible timeslice {} that is older than the last one we sent {}",
             oldestPossibleOutput.timeslice.value, decongestion.lastTimeslice);
        return;
      }
      auto &queue = services.get<AsyncQueue>();
      auto& spec = services.get<DeviceSpec const>();
      auto *device = services.get<RawDeviceService>().device();
      /// We use the oldest possible timeslice to debounce, so that only the latest one
      /// at the end of one iteration is sent.
      LOGP(debug, "Queueing oldest possible timeslice {} propagation for execution.", oldestPossibleOutput.timeslice.value);
      AsyncQueueHelpers::post(
        queue, decongestion.oldestPossibleTimesliceTask, [oldestPossibleOutput, &decongestion, &proxy, &spec, device, &timesliceIndex]() {
          if (decongestion.lastTimeslice >= oldestPossibleOutput.timeslice.value) {
            LOGP(debug, "Not sending already sent value {} >= {}", decongestion.lastTimeslice, oldestPossibleOutput.timeslice.value);
            return;
          }
          LOGP(debug, "Running oldest possible timeslice {} propagation.", oldestPossibleOutput.timeslice.value);
          DataProcessingHelpers::broadcastOldestPossibleTimeslice(proxy, oldestPossibleOutput.timeslice.value);

          for (int fi = 0; fi < proxy.getNumForwardChannels(); fi++) {
            auto& info = proxy.getForwardChannelInfo(ChannelIndex{fi});
            auto& state = proxy.getForwardChannelState(ChannelIndex{fi});
            // TODO: this we could cache in the proxy at the bind moment.
            if (info.channelType != ChannelAccountingType::DPL) {
              LOG(debug) << "Skipping channel";
              continue;
            }
            if (DataProcessingHelpers::sendOldestPossibleTimeframe(info, state, oldestPossibleOutput.timeslice.value)) {
              LOGP(debug, "Forwarding to channel {} oldest possible timeslice {}, prio 20", info.name, oldestPossibleOutput.timeslice.value);
            }
          }
          decongestion.lastTimeslice = oldestPossibleOutput.timeslice.value;
          if (decongestion.orderedCompletionPolicyActive) {
            int64_t oldNextTimeslice = decongestion.nextTimeslice;
            decongestion.nextTimeslice = std::max(decongestion.nextTimeslice, (int64_t)oldestPossibleOutput.timeslice.value);
            if (oldNextTimeslice != decongestion.nextTimeslice) {
              LOGP(error, "Some Lifetime::Timeframe data got dropped starting at {}", oldNextTimeslice);
              timesliceIndex.rescan();
            }
          }
        },
        TimesliceId{oldestPossibleTimeslice}, -1); },
    .kind = ServiceKind::Serial};
}

// FIXME: allow configuring the default number of threads per device
//        This should probably be done by overriding the preFork
//        callback and using the boost program options there to
//        get the default number of threads.
o2::framework::ServiceSpec CommonServices::threadPool(int numWorkers)
{
  return ServiceSpec{
    .name = "threadpool",
    .init = [](ServiceRegistryRef services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto* pool = new ThreadPool();
      // FIXME: this will require some extra argument for the configuration context of a service
      pool->poolSize = 1;
      return ServiceHandle{TypeIdHelpers::uniqueId<ThreadPool>(), pool};
    },
    .configure = [](InitContext&, void* service) -> void* {
      auto* t = reinterpret_cast<ThreadPool*>(service);
      // FIXME: this will require some extra argument for the configuration context of a service
      t->poolSize = 1;
      return service;
    },
    .postForkParent = [](ServiceRegistryRef services) -> void {
      // FIXME: this will require some extra argument for the configuration context of a service
      auto numWorkersS = std::to_string(1);
      setenv("UV_THREADPOOL_SIZE", numWorkersS.c_str(), 0);
    },
    .kind = ServiceKind::Serial};
}

namespace
{
auto sendRelayerMetrics(ServiceRegistryRef registry, DataProcessingStats& stats) -> void
{
  // Update the timer to make sure we have  the correct time when sending out the stats.
  uv_update_time(registry.get<DeviceState>().loop);
  // Derive the amount of shared memory used
  auto& runningWorkflow = registry.get<RunningWorkflowInfo const>();
  using namespace fair::mq::shmem;
  auto& spec = registry.get<DeviceSpec const>();

  // FIXME: Ugly, but we do it only every 5 seconds...
  if (spec.name == "readout-proxy") {
    auto device = registry.get<RawDeviceService>().device();
    long freeMemory = -1;
    try {
      freeMemory = fair::mq::shmem::Monitor::GetFreeMemory(ShmId{makeShmIdStr(device->fConfig->GetProperty<uint64_t>("shmid"))}, runningWorkflow.shmSegmentId);
    } catch (...) {
    }
    if (freeMemory == -1) {
      try {
        freeMemory = fair::mq::shmem::Monitor::GetFreeMemory(SessionId{device->fConfig->GetProperty<std::string>("session")}, runningWorkflow.shmSegmentId);
      } catch (...) {
      }
    }
    stats.updateStats({static_cast<unsigned short>(static_cast<int>(ProcessingStatsId::AVAILABLE_MANAGED_SHM_BASE) + (runningWorkflow.shmSegmentId % 512)), DataProcessingStats::Op::SetIfPositive, freeMemory});
  }

  ZoneScopedN("send metrics");
  auto device = registry.get<RawDeviceService>().device();

  int64_t totalBytesIn = 0;
  int64_t totalBytesOut = 0;

  for (auto& channel : device->GetChannels()) {
    totalBytesIn += channel.second[0].GetBytesRx();
    totalBytesOut += channel.second[0].GetBytesTx();
  }

  stats.updateStats({static_cast<short>(ProcessingStatsId::TOTAL_BYTES_IN), DataProcessingStats::Op::Set, totalBytesIn / 1000000});
  stats.updateStats({static_cast<short>(ProcessingStatsId::TOTAL_BYTES_OUT), DataProcessingStats::Op::Set, totalBytesOut / 1000000});

  stats.updateStats({static_cast<short>(ProcessingStatsId::TOTAL_RATE_IN_MB_S), DataProcessingStats::Op::InstantaneousRate, totalBytesIn / 1000000});
  stats.updateStats({static_cast<short>(ProcessingStatsId::TOTAL_RATE_OUT_MB_S), DataProcessingStats::Op::InstantaneousRate, totalBytesOut / 1000000});
};

auto flushStates(ServiceRegistryRef registry, DataProcessingStates& states) -> void
{
  states.flushChangedStates([&states, registry](std::string const& spec, int64_t timestamp, std::string_view value) mutable -> void {
    auto& client = registry.get<ControlService>();
    client.push(spec, value, timestamp);
  });
}

O2_DECLARE_DYNAMIC_LOG(monitoring_service);

/// This will flush metrics only once every second.
auto flushMetrics(ServiceRegistryRef registry, DataProcessingStats& stats) -> void
{
  O2_SIGNPOST_ID_GENERATE(sid, monitoring_service);
  O2_SIGNPOST_START(monitoring_service, sid, "flush", "flushing metrics");
  auto& monitoring = registry.get<Monitoring>();
  auto& relayer = registry.get<DataRelayer>();

  // Send all the relevant metrics for the relayer to update the GUI
  stats.flushChangedMetrics([&monitoring](DataProcessingStats::MetricSpec const& spec, int64_t timestamp, int64_t value) mutable -> void {
    // convert timestamp to a time_point
    auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(std::chrono::milliseconds(timestamp));
    auto metric = o2::monitoring::Metric{spec.name, Metric::DefaultVerbosity, tp};
    if (spec.kind == DataProcessingStats::Kind::UInt64) {
      if (value < 0) {
        LOG(debug) << "Value for " << spec.name << " is negative, setting to 0";
        value = 0;
      }
      metric.addValue((uint64_t)value, "value");
    } else {
      if (value > (int64_t)std::numeric_limits<int>::max()) {
        LOG(warning) << "Value for " << spec.name << " is too large, setting to INT_MAX";
        value = (int64_t)std::numeric_limits<int>::max();
      }
      if (value < (int64_t)std::numeric_limits<int>::min()) {
        value = (int64_t)std::numeric_limits<int>::min();
        LOG(warning) << "Value for " << spec.name << " is too small, setting to INT_MIN";
      }
      metric.addValue((int)value, "value");
    }
    if (spec.scope == DataProcessingStats::Scope::DPL) {
      metric.addTag(o2::monitoring::tags::Key::Subsystem, o2::monitoring::tags::Value::DPL);
    }
    monitoring.send(std::move(metric));
  });
  relayer.sendContextState();
  monitoring.flushBuffer();
  O2_SIGNPOST_END(monitoring_service, sid, "flush", "done flushing metrics");
};
} // namespace

o2::framework::ServiceSpec CommonServices::dataProcessingStats()
{
  return ServiceSpec{
    .name = "data-processing-stats",
    .init = [](ServiceRegistryRef services, DeviceState& state, fair::mq::ProgOptions& options) -> ServiceHandle {
      timespec now;
      clock_gettime(CLOCK_REALTIME, &now);
      uv_update_time(state.loop);
      uint64_t offset = now.tv_sec * 1000 - uv_now(state.loop);
      auto* stats = new DataProcessingStats(TimingHelpers::defaultRealtimeBaseConfigurator(offset, state.loop),
                                            TimingHelpers::defaultCPUTimeConfigurator(state.loop));
      auto& runningWorkflow = services.get<RunningWorkflowInfo const>();

      // It makes no sense to update the stats more often than every 5s
      int quickUpdateInterval = 5000;
      uint64_t quickRefreshInterval = 7000;
      uint64_t onlineRefreshLatency = 60000; // For metrics which are reported online, we flush them every 60s regardless of their state.
      using MetricSpec = DataProcessingStats::MetricSpec;
      using Kind = DataProcessingStats::Kind;
      using Scope = DataProcessingStats::Scope;

#ifdef NDEBUG
      bool enableDebugMetrics = false;
#else
      bool enableDebugMetrics = true;
#endif
      bool arrowAndResourceLimitingMetrics = false;
      DeploymentMode deploymentMode = DefaultsHelpers::deploymentMode();
      if (deploymentMode != DeploymentMode::OnlineDDS && deploymentMode != DeploymentMode::OnlineECS && deploymentMode != DeploymentMode::OnlineAUX && deploymentMode != DeploymentMode::FST) {
        arrowAndResourceLimitingMetrics = true;
      }

      std::vector<DataProcessingStats::MetricSpec> metrics = {
        MetricSpec{.name = "errors",
                   .metricId = (int)ProcessingStatsId::ERROR_COUNT,
                   .kind = Kind::UInt64,
                   .scope = Scope::Online,
                   .minPublishInterval = quickUpdateInterval,
                   .maxRefreshLatency = quickRefreshInterval},
        MetricSpec{.name = "exceptions",
                   .metricId = (int)ProcessingStatsId::EXCEPTION_COUNT,
                   .kind = Kind::UInt64,
                   .scope = Scope::Online,
                   .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "inputs/relayed/pending",
                   .metricId = (int)ProcessingStatsId::PENDING_INPUTS,
                   .kind = Kind::UInt64,
                   .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "inputs/relayed/incomplete",
                   .metricId = (int)ProcessingStatsId::INCOMPLETE_INPUTS,
                   .kind = Kind::UInt64,
                   .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "inputs/relayed/total",
                   .metricId = (int)ProcessingStatsId::TOTAL_INPUTS,
                   .kind = Kind::UInt64,
                   .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "elapsed_time_ms",
                   .metricId = (int)ProcessingStatsId::LAST_ELAPSED_TIME_MS,
                   .kind = Kind::UInt64,
                   .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "total_wall_time_ms",
                   .metricId = (int)ProcessingStatsId::TOTAL_WALL_TIME_MS,
                   .kind = Kind::UInt64,
                   .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "last_processed_input_size_byte",
                   .metricId = (int)ProcessingStatsId::LAST_PROCESSED_SIZE,
                   .kind = Kind::UInt64,
                   .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "total_processed_input_size_byte",
                   .metricId = (int)ProcessingStatsId::TOTAL_PROCESSED_SIZE,
                   .kind = Kind::UInt64,
                   .scope = Scope::Online,
                   .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "total_sigusr1",
                   .metricId = (int)ProcessingStatsId::TOTAL_SIGUSR1,
                   .kind = Kind::UInt64,
                   .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "consumed-timeframes",
                   .metricId = (int)ProcessingStatsId::CONSUMED_TIMEFRAMES,
                   .kind = Kind::UInt64,
                   .minPublishInterval = 0,
                   .maxRefreshLatency = quickRefreshInterval,
                   .sendInitialValue = true},
        MetricSpec{.name = "min_input_latency_ms",
                   .metricId = (int)ProcessingStatsId::LAST_MIN_LATENCY,
                   .kind = Kind::UInt64,
                   .scope = Scope::Online,
                   .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "max_input_latency_ms",
                   .metricId = (int)ProcessingStatsId::LAST_MAX_LATENCY,
                   .kind = Kind::UInt64,
                   .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "total_rate_in_mb_s",
                   .metricId = (int)ProcessingStatsId::TOTAL_RATE_IN_MB_S,
                   .kind = Kind::Rate,
                   .scope = Scope::Online,
                   .minPublishInterval = quickUpdateInterval,
                   .maxRefreshLatency = onlineRefreshLatency,
                   .sendInitialValue = true},
        MetricSpec{.name = "total_rate_out_mb_s",
                   .metricId = (int)ProcessingStatsId::TOTAL_RATE_OUT_MB_S,
                   .kind = Kind::Rate,
                   .scope = Scope::Online,
                   .minPublishInterval = quickUpdateInterval,
                   .maxRefreshLatency = onlineRefreshLatency,
                   .sendInitialValue = true},
        MetricSpec{.name = "processing_rate_hz",
                   .metricId = (int)ProcessingStatsId::PROCESSING_RATE_HZ,
                   .kind = Kind::Rate,
                   .scope = Scope::Online,
                   .minPublishInterval = quickUpdateInterval,
                   .maxRefreshLatency = onlineRefreshLatency,
                   .sendInitialValue = true},
        MetricSpec{.name = "cpu_usage_fraction",
                   .metricId = (int)ProcessingStatsId::CPU_USAGE_FRACTION,
                   .kind = Kind::Rate,
                   .scope = Scope::Online,
                   .minPublishInterval = quickUpdateInterval,
                   .maxRefreshLatency = onlineRefreshLatency,
                   .sendInitialValue = true},
        MetricSpec{.name = "performed_computations",
                   .metricId = (int)ProcessingStatsId::PERFORMED_COMPUTATIONS,
                   .kind = Kind::UInt64,
                   .scope = Scope::Online,
                   .minPublishInterval = quickUpdateInterval,
                   .maxRefreshLatency = onlineRefreshLatency,
                   .sendInitialValue = true},
        MetricSpec{.name = "total_bytes_in",
                   .metricId = (int)ProcessingStatsId::TOTAL_BYTES_IN,
                   .kind = Kind::UInt64,
                   .scope = Scope::Online,
                   .minPublishInterval = quickUpdateInterval,
                   .maxRefreshLatency = onlineRefreshLatency,
                   .sendInitialValue = true},
        MetricSpec{.name = "total_bytes_out",
                   .metricId = (int)ProcessingStatsId::TOTAL_BYTES_OUT,
                   .kind = Kind::UInt64,
                   .scope = Scope::Online,
                   .minPublishInterval = quickUpdateInterval,
                   .maxRefreshLatency = onlineRefreshLatency,
                   .sendInitialValue = true},
        MetricSpec{.name = fmt::format("available_managed_shm_{}", runningWorkflow.shmSegmentId),
                   .metricId = (int)ProcessingStatsId::AVAILABLE_MANAGED_SHM_BASE + (runningWorkflow.shmSegmentId % 512),
                   .kind = Kind::UInt64,
                   .scope = Scope::Online,
                   .minPublishInterval = 500,
                   .maxRefreshLatency = onlineRefreshLatency,
                   .sendInitialValue = true},
        MetricSpec{.name = "malformed_inputs", .metricId = static_cast<short>(ProcessingStatsId::MALFORMED_INPUTS), .kind = Kind::UInt64, .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "dropped_computations", .metricId = static_cast<short>(ProcessingStatsId::DROPPED_COMPUTATIONS), .kind = Kind::UInt64, .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "dropped_incoming_messages", .metricId = static_cast<short>(ProcessingStatsId::DROPPED_INCOMING_MESSAGES), .kind = Kind::UInt64, .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "relayed_messages", .metricId = static_cast<short>(ProcessingStatsId::RELAYED_MESSAGES), .kind = Kind::UInt64, .minPublishInterval = quickUpdateInterval},
        MetricSpec{.name = "arrow-bytes-destroyed",
                   .enabled = arrowAndResourceLimitingMetrics,
                   .metricId = static_cast<short>(ProcessingStatsId::ARROW_BYTES_DESTROYED),
                   .kind = Kind::UInt64,
                   .scope = Scope::DPL,
                   .minPublishInterval = 0,
                   .maxRefreshLatency = 10000,
                   .sendInitialValue = true},
        MetricSpec{.name = "arrow-messages-destroyed",
                   .enabled = arrowAndResourceLimitingMetrics,
                   .metricId = static_cast<short>(ProcessingStatsId::ARROW_MESSAGES_DESTROYED),
                   .kind = Kind::UInt64,
                   .scope = Scope::DPL,
                   .minPublishInterval = 0,
                   .maxRefreshLatency = 10000,
                   .sendInitialValue = true},
        MetricSpec{.name = "arrow-bytes-created",
                   .enabled = arrowAndResourceLimitingMetrics,
                   .metricId = static_cast<short>(ProcessingStatsId::ARROW_BYTES_CREATED),
                   .kind = Kind::UInt64,
                   .scope = Scope::DPL,
                   .minPublishInterval = 0,
                   .maxRefreshLatency = 10000,
                   .sendInitialValue = true},
        MetricSpec{.name = "arrow-messages-created",
                   .enabled = arrowAndResourceLimitingMetrics,
                   .metricId = static_cast<short>(ProcessingStatsId::ARROW_MESSAGES_CREATED),
                   .kind = Kind::UInt64,
                   .scope = Scope::DPL,
                   .minPublishInterval = 0,
                   .maxRefreshLatency = 10000,
                   .sendInitialValue = true},
        MetricSpec{.name = "arrow-bytes-expired",
                   .enabled = arrowAndResourceLimitingMetrics,
                   .metricId = static_cast<short>(ProcessingStatsId::ARROW_BYTES_EXPIRED),
                   .kind = Kind::UInt64,
                   .scope = Scope::DPL,
                   .minPublishInterval = 0,
                   .maxRefreshLatency = 10000,
                   .sendInitialValue = true},
        MetricSpec{.name = "shm-offer-bytes-consumed",
                   .enabled = arrowAndResourceLimitingMetrics,
                   .metricId = static_cast<short>(ProcessingStatsId::SHM_OFFER_BYTES_CONSUMED),
                   .kind = Kind::UInt64,
                   .scope = Scope::DPL,
                   .minPublishInterval = 0,
                   .maxRefreshLatency = 10000,
                   .sendInitialValue = true},
        MetricSpec{.name = "resources-missing",
                   .enabled = enableDebugMetrics,
                   .metricId = static_cast<short>(ProcessingStatsId::RESOURCES_MISSING),
                   .kind = Kind::UInt64,
                   .scope = Scope::DPL,
                   .minPublishInterval = 1000,
                   .maxRefreshLatency = 1000,
                   .sendInitialValue = true},
        MetricSpec{.name = "resources-insufficient",
                   .enabled = enableDebugMetrics,
                   .metricId = static_cast<short>(ProcessingStatsId::RESOURCES_INSUFFICIENT),
                   .kind = Kind::UInt64,
                   .scope = Scope::DPL,
                   .minPublishInterval = 1000,
                   .maxRefreshLatency = 1000,
                   .sendInitialValue = true},
        MetricSpec{.name = "resources-satisfactory",
                   .enabled = enableDebugMetrics,
                   .metricId = static_cast<short>(ProcessingStatsId::RESOURCES_SATISFACTORY),
                   .kind = Kind::UInt64,
                   .scope = Scope::DPL,
                   .minPublishInterval = 1000,
                   .maxRefreshLatency = 1000,
                   .sendInitialValue = true},
        MetricSpec{.name = "resource-offer-expired",
                   .enabled = arrowAndResourceLimitingMetrics,
                   .metricId = static_cast<short>(ProcessingStatsId::RESOURCE_OFFER_EXPIRED),
                   .kind = Kind::UInt64,
                   .scope = Scope::DPL,
                   .minPublishInterval = 0,
                   .maxRefreshLatency = 10000,
                   .sendInitialValue = true}};

      for (auto& metric : metrics) {
        stats->registerMetric(metric);
      }

      return ServiceHandle{TypeIdHelpers::uniqueId<DataProcessingStats>(), stats};
    },
    .configure = noConfiguration(),
    .preProcessing = [](ProcessingContext& context, void* service) {
      auto* stats = (DataProcessingStats*)service;
      flushMetrics(context.services(), *stats);
    },
    .postProcessing = [](ProcessingContext& context, void* service) {
      auto* stats = (DataProcessingStats*)service;
      stats->updateStats({(short)ProcessingStatsId::PERFORMED_COMPUTATIONS, DataProcessingStats::Op::Add, 1});
      flushMetrics(context.services(), *stats); },
    .preDangling = [](DanglingContext& context, void* service) {
       auto* stats = (DataProcessingStats*)service;
       sendRelayerMetrics(context.services(), *stats);
       flushMetrics(context.services(), *stats); },
    .postDangling = [](DanglingContext& context, void* service) {
       auto* stats = (DataProcessingStats*)service;
       sendRelayerMetrics(context.services(), *stats);
       flushMetrics(context.services(), *stats); },
    .preEOS = [](EndOfStreamContext& context, void* service) {
      auto* stats = (DataProcessingStats*)service;
      sendRelayerMetrics(context.services(), *stats);
      flushMetrics(context.services(), *stats); },
    .postDispatching = [](ProcessingContext& context, void* service) {
      auto* stats = (DataProcessingStats*)service;
      flushMetrics(context.services(), *stats); },
    .preLoop = [](ServiceRegistryRef ref, void* service) {
      auto* stats = (DataProcessingStats*)service;
      flushMetrics(ref, *stats); },
    .kind = ServiceKind::Serial};
}

// This is similar to the dataProcessingStats, but it designed to synchronize
// history-less metrics which are e.g. used for the GUI.
o2::framework::ServiceSpec CommonServices::dataProcessingStates()
{
  return ServiceSpec{
    .name = "data-processing-states",
    .init = [](ServiceRegistryRef services, DeviceState& state, fair::mq::ProgOptions& options) -> ServiceHandle {
      timespec now;
      clock_gettime(CLOCK_REALTIME, &now);
      uv_update_time(state.loop);
      uint64_t offset = now.tv_sec * 1000 - uv_now(state.loop);
      auto* states = new DataProcessingStates(TimingHelpers::defaultRealtimeBaseConfigurator(offset, state.loop),
                                              TimingHelpers::defaultCPUTimeConfigurator(state.loop));
      states->registerState({"dummy_state", (short)ProcessingStateId::DUMMY_STATE});
      return ServiceHandle{TypeIdHelpers::uniqueId<DataProcessingStates>(), states};
    },
    .configure = noConfiguration(),
    .postProcessing = [](ProcessingContext& context, void* service) {
      auto* states = (DataProcessingStates*)service;
      states->processCommandQueue(); },
    .preDangling = [](DanglingContext& context, void* service) {
       auto* states = (DataProcessingStates*)service;
       flushStates(context.services(), *states); },
    .postDangling = [](DanglingContext& context, void* service) {
       auto* states = (DataProcessingStates*)service;
       flushStates(context.services(), *states); },
    .preEOS = [](EndOfStreamContext& context, void* service) {
      auto* states = (DataProcessingStates*)service;
      flushStates(context.services(), *states); },
    .kind = ServiceKind::Global};
}

struct GUIMetrics {
};

o2::framework::ServiceSpec CommonServices::guiMetricsSpec()
{
  return ServiceSpec{
    .name = "gui-metrics",
    .init = [](ServiceRegistryRef services, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      auto* stats = new GUIMetrics();
      auto& monitoring = services.get<Monitoring>();
      auto& spec = services.get<DeviceSpec const>();
      monitoring.send({(int)spec.inputChannels.size(), fmt::format("oldest_possible_timeslice/h"), o2::monitoring::Verbosity::Debug});
      monitoring.send({(int)1, fmt::format("oldest_possible_timeslice/w"), o2::monitoring::Verbosity::Debug});
      monitoring.send({(int)spec.outputChannels.size(), fmt::format("oldest_possible_output/h"), o2::monitoring::Verbosity::Debug});
      monitoring.send({(int)1, fmt::format("oldest_possible_output/w"), o2::monitoring::Verbosity::Debug});
      return ServiceHandle{TypeIdHelpers::uniqueId<GUIMetrics>(), stats};
    },
    .configure = noConfiguration(),
    .postProcessing = [](ProcessingContext& context, void* service) {
      auto& relayer = context.services().get<DataRelayer>();
      auto& monitoring = context.services().get<Monitoring>();
      auto& spec = context.services().get<DeviceSpec const>();
      auto oldestPossibleOutput = relayer.getOldestPossibleOutput();
      for (size_t ci; ci < spec.outputChannels.size(); ++ci) {
        monitoring.send({(uint64_t)oldestPossibleOutput.timeslice.value, fmt::format("oldest_possible_output/{}", ci), o2::monitoring::Verbosity::Debug});
      } },
    .domainInfoUpdated = [](ServiceRegistryRef registry, size_t timeslice, ChannelIndex channel) {
      auto& monitoring = registry.get<Monitoring>();
      monitoring.send({(uint64_t)timeslice, fmt::format("oldest_possible_timeslice/{}", channel.value), o2::monitoring::Verbosity::Debug}); },
    .active = false,
    .kind = ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::objectCache()
{
  return ServiceSpec{
    .name = "object-cache",
    .init = [](ServiceRegistryRef, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
      auto* cache = new ObjectCache();
      return ServiceHandle{TypeIdHelpers::uniqueId<ObjectCache>(), cache};
    },
    .configure = noConfiguration(),
    .kind = ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::dataProcessorContextSpec()
{
  return ServiceSpec{
    .name = "data-processing-context",
    .init = [](ServiceRegistryRef, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
      return ServiceHandle{TypeIdHelpers::uniqueId<DataProcessorContext>(), new DataProcessorContext()};
    },
    .configure = noConfiguration(),
    .exit = [](ServiceRegistryRef, void* service) { auto* context = (DataProcessorContext*)service; delete context; },
    .kind = ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::deviceContextSpec()
{
  return ServiceSpec{
    .name = "device-context",
    .init = [](ServiceRegistryRef, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
      return ServiceHandle{TypeIdHelpers::uniqueId<DeviceContext>(), new DeviceContext()};
    },
    .configure = noConfiguration(),
    .kind = ServiceKind::Serial};
}

o2::framework::ServiceSpec CommonServices::dataAllocatorSpec()
{
  return ServiceSpec{
    .name = "data-allocator",
    .uniqueId = simpleServiceId<DataAllocator>(),
    .init = [](ServiceRegistryRef ref, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
      return ServiceHandle{
        .hash = TypeIdHelpers::uniqueId<DataAllocator>(),
        .instance = new DataAllocator(ref),
        .kind = ServiceKind::Stream,
        .name = "data-allocator",
      };
    },
    .configure = noConfiguration(),
    .kind = ServiceKind::Stream};
}

/// Split a string into a vector of strings using : as a separator.
std::vector<ServiceSpec> CommonServices::defaultServices(std::string extraPlugins, int numThreads)
{
  std::vector<ServiceSpec> specs{
    dataProcessorContextSpec(),
    streamContextSpec(),
    dataAllocatorSpec(),
    asyncQueue(),
    timingInfoSpec(),
    timesliceIndex(),
    driverClientSpec(),
    datatakingContextSpec(),
    monitoringSpec(),
    configurationSpec(),
    controlSpec(),
    rootFileSpec(),
    parallelSpec(),
    callbacksSpec(),
    dataProcessingStats(),
    dataProcessingStates(),
    dataRelayer(),
    CommonMessageBackends::fairMQDeviceProxy(),
    dataSender(),
    objectCache(),
    ccdbSupportSpec()};

  DeploymentMode deploymentMode = DefaultsHelpers::deploymentMode();
  if (deploymentMode != DeploymentMode::OnlineDDS && deploymentMode != DeploymentMode::OnlineECS && deploymentMode != DeploymentMode::OnlineAUX && deploymentMode != DeploymentMode::FST) {
    specs.push_back(ArrowSupport::arrowBackendSpec());
  }
  specs.push_back(CommonMessageBackends::fairMQBackendSpec());
  specs.push_back(CommonMessageBackends::stringBackendSpec());
  specs.push_back(decongestionSpec());

  std::string loadableServicesStr = extraPlugins;
  // Do not load InfoLogger by default if we are not at P2.
  if (deploymentMode == DeploymentMode::OnlineDDS || deploymentMode == DeploymentMode::OnlineECS || deploymentMode == DeploymentMode::OnlineAUX) {
    if (loadableServicesStr.empty() == false) {
      loadableServicesStr += ",";
    }
    loadableServicesStr += "O2FrameworkDataTakingSupport:InfoLoggerContext,O2FrameworkDataTakingSupport:InfoLogger";
  }
  // Load plugins depending on the environment
  std::vector<LoadableService> loadableServices = {};
  char* loadableServicesEnv = getenv("DPL_LOAD_SERVICES");
  // String to define the services to load is:
  //
  // library1:name1,library2:name2,...
  if (loadableServicesEnv) {
    if (loadableServicesStr.empty() == false) {
      loadableServicesStr += ",";
    }
    loadableServicesStr += loadableServicesEnv;
  }
  loadableServices = ServiceHelpers::parseServiceSpecString(loadableServicesStr.c_str());
  ServiceHelpers::loadFromPlugin(loadableServices, specs);
  // I should make it optional depending wether the GUI is there or not...
  specs.push_back(CommonServices::guiMetricsSpec());
  if (numThreads) {
    specs.push_back(threadPool(numThreads));
  }
  return specs;
}

std::vector<ServiceSpec> CommonServices::arrowServices()
{
  return {
    ArrowSupport::arrowTableSlicingCacheDefSpec(),
    ArrowSupport::arrowTableSlicingCacheSpec() //
  };
}

} // namespace o2::framework
#pragma GCC diagnostic pop
