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
#ifdef DPL_ENABLE_TRACING
#define TRACY_ENABLE
#include <tracy/TracyClient.cpp>
#endif
#include "Framework/AsyncQueue.h"
#include "Framework/DataProcessingDevice.h"
#include "Framework/ChannelMatching.h"
#include "Framework/ControlService.h"
#include "Framework/ComputingQuotaEvaluator.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataProcessingStates.h"
#include "Framework/DataProcessor.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DeviceState.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/DispatchControl.h"
#include "Framework/DanglingContext.h"
#include "Framework/DomainInfoHeader.h"
#include "Framework/DriverClient.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/FairOptionsRetriever.h"
#include "ConfigurationOptionsRetriever.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Framework/CallbackService.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/InputRecord.h"
#include "Framework/InputSpan.h"
#include "Framework/Signpost.h"
#include "Framework/TimingHelpers.h"
#include "Framework/SourceInfoHeader.h"
#include "Framework/Logger.h"
#include "Framework/DriverClient.h"
#include "Framework/Monitoring.h"
#include "Framework/TimesliceIndex.h"
#include "Framework/VariableContextHelpers.h"
#include "Framework/DataProcessingContext.h"
#include "Framework/DeviceContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/StreamContext.h"
#include "Framework/DefaultsHelpers.h"

#include "PropertyTreeHelpers.h"
#include "DataProcessingStatus.h"
#include "DecongestionService.h"
#include "Framework/DataProcessingHelpers.h"
#include "DataRelayerHelpers.h"
#include "ProcessingPoliciesHelpers.h"
#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"

#include "ScopedExit.h"

#include <Framework/Tracing.h>

#include <fairmq/Parts.h>
#include <fairmq/Socket.h>
#include <fairmq/ProgOptions.h>
#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>
#include <TMessage.h>
#include <TClonesArray.h>

#include <algorithm>
#include <vector>
#include <numeric>
#include <memory>
#include <unordered_map>
#include <uv.h>
#include <execinfo.h>
#include <sstream>
#include <boost/property_tree/json_parser.hpp>

using namespace o2::framework;
using ConfigurationInterface = o2::configuration::ConfigurationInterface;
using DataHeader = o2::header::DataHeader;

constexpr int DEFAULT_MAX_CHANNEL_AHEAD = 128;

namespace o2::framework
{

template <>
struct ServiceKindExtractor<ConfigurationInterface> {
  constexpr static ServiceKind kind = ServiceKind::Global;
};

/// We schedule a timer to reduce CPU usage.
/// Watching stdin for commands probably a better approach.
void on_idle_timer(uv_timer_t* handle)
{
  ZoneScopedN("Idle timer");
  auto* state = (DeviceState*)handle->data;
  state->loopReason |= DeviceState::TIMER_EXPIRED;
}

void on_transition_requested_expired(uv_timer_t* handle)
{
  ZoneScopedN("Transition expired");
  auto* state = (DeviceState*)handle->data;
  state->loopReason |= DeviceState::TIMER_EXPIRED;
  LOGP(info, "Timer expired. Forcing transition to READY");
  state->transitionHandling = TransitionHandlingState::Expired;
}

void on_communication_requested(uv_async_t* s)
{
  auto* state = (DeviceState*)s->data;
  state->loopReason |= DeviceState::METRICS_MUST_FLUSH;
}

DeviceSpec const& getRunningDevice(RunningDeviceRef const& running, ServiceRegistryRef const& services)
{
  auto& devices = services.get<o2::framework::RunningWorkflowInfo const>().devices;
  return devices[running.index];
}

struct locked_execution {
  ServiceRegistryRef& ref;
  locked_execution(ServiceRegistryRef& ref_) : ref(ref_) { ref.lock(); }
  ~locked_execution() { ref.unlock(); }
};

DataProcessingDevice::DataProcessingDevice(RunningDeviceRef running, ServiceRegistry& registry, ProcessingPolicies& policies)
  : mRunningDevice{running},
    mConfigRegistry{nullptr},
    mServiceRegistry{registry},
    mProcessingPolicies{policies}
{
  GetConfig()->Subscribe<std::string>("dpl", [&registry = mServiceRegistry](const std::string& key, std::string value) {
    if (key == "cleanup") {
      auto ref = ServiceRegistryRef{registry, ServiceRegistry::globalDeviceSalt()};
      auto& deviceState = ref.get<DeviceState>();
      int64_t cleanupCount = deviceState.cleanupCount.load();
      int64_t newCleanupCount = std::stoll(value);
      if (newCleanupCount <= cleanupCount) {
        return;
      }
      deviceState.cleanupCount.store(newCleanupCount);
      for (auto& info : deviceState.inputChannelInfos) {
        fair::mq::Parts parts;
        while (info.channel->Receive(parts, 0)) {
          LOGP(debug, "Dropping {} parts", parts.Size());
          if (parts.Size() == 0) {
            break;
          }
        }
      }
    }
  });

  std::function<void(const fair::mq::State)> stateWatcher = [this, &registry = mServiceRegistry](const fair::mq::State state) -> void {
    auto ref = ServiceRegistryRef{registry, ServiceRegistry::globalDeviceSalt()};
    auto& deviceState = ref.get<DeviceState>();
    auto& control = ref.get<ControlService>();
    auto& callbacks = ref.get<CallbackService>();
    control.notifyDeviceState(fair::mq::GetStateName(state));
    callbacks.call<CallbackService::Id::DeviceStateChanged>(ServiceRegistryRef{ref}, (int)state);

    if (deviceState.nextFairMQState.empty() == false) {
      auto state = deviceState.nextFairMQState.back();
      (void)this->ChangeState(state);
      deviceState.nextFairMQState.pop_back();
    }
  };

  // 99 is to execute DPL callbacks last
  this->SubscribeToStateChange("99-dpl", stateWatcher);

  // One task for now.
  mStreams.resize(1);
  mHandles.resize(1);

  ServiceRegistryRef ref{mServiceRegistry};
  mAwakeHandle = (uv_async_t*)malloc(sizeof(uv_async_t));
  auto& state = ref.get<DeviceState>();
  assert(state.loop);
  int res = uv_async_init(state.loop, mAwakeHandle, on_communication_requested);
  mAwakeHandle->data = &state;
  if (res < 0) {
    LOG(error) << "Unable to initialise subscription";
  }

  /// This should post a message on the queue...
  SubscribeToNewTransition("dpl", [wakeHandle = mAwakeHandle](fair::mq::Transition t) {
    int res = uv_async_send(wakeHandle);
    if (res < 0) {
      LOG(error) << "Unable to notify subscription";
    }
    LOG(debug) << "State transition requested";
  });
}

// Callback to execute the processing. Notice how the data is
// is a vector of DataProcessorContext so that we can index the correct
// one with the thread id. For the moment we simply use the first one.
void run_callback(uv_work_t* handle)
{
  ZoneScopedN("run_callback");
  auto* task = (TaskStreamInfo*)handle->data;
  auto ref = ServiceRegistryRef{*task->registry, ServiceRegistry::globalStreamSalt(task->id.index + 1)};
  DataProcessingDevice::doPrepare(ref);
  DataProcessingDevice::doRun(ref);
  //  FrameMark;
}

// Once the processing in a thread is done, this is executed on the main thread.
void run_completion(uv_work_t* handle, int status)
{
  auto* task = (TaskStreamInfo*)handle->data;
  // Notice that the completion, while running on the main thread, still
  // has a salt which is associated to the actual stream which was doing the computation
  auto ref = ServiceRegistryRef{*task->registry, ServiceRegistry::globalStreamSalt(task->id.index + 1)};
  auto& state = ref.get<DeviceState>();
  auto& quotaEvaluator = ref.get<ComputingQuotaEvaluator>();

  using o2::monitoring::Metric;
  using o2::monitoring::Monitoring;
  using o2::monitoring::tags::Key;
  using o2::monitoring::tags::Value;

  static std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats&)> reportConsumedOffer = [ref](ComputingQuotaOffer const& accumulatedConsumed, ComputingQuotaStats& stats) {
    auto& dpStats = ref.get<DataProcessingStats>();
    stats.totalConsumedBytes += accumulatedConsumed.sharedMemory;

    dpStats.updateStats({static_cast<short>(ProcessingStatsId::SHM_OFFER_BYTES_CONSUMED), DataProcessingStats::Op::Set, stats.totalConsumedBytes});
    dpStats.processCommandQueue();
    assert(stats.totalConsumedBytes == dpStats.metrics[(short)ProcessingStatsId::SHM_OFFER_BYTES_CONSUMED]);
  };

  static std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats const&)> reportExpiredOffer = [ref](ComputingQuotaOffer const& offer, ComputingQuotaStats const& stats) {
    auto& dpStats = ref.get<DataProcessingStats>();
    dpStats.updateStats({static_cast<short>(ProcessingStatsId::RESOURCE_OFFER_EXPIRED), DataProcessingStats::Op::Set, stats.totalExpiredOffers});
    dpStats.updateStats({static_cast<short>(ProcessingStatsId::ARROW_BYTES_EXPIRED), DataProcessingStats::Op::Set, stats.totalExpiredBytes});
    dpStats.processCommandQueue();
  };

  for (auto& consumer : state.offerConsumers) {
    quotaEvaluator.consume(task->id.index, consumer, reportConsumedOffer);
  }
  state.offerConsumers.clear();
  quotaEvaluator.handleExpired(reportExpiredOffer);
  quotaEvaluator.dispose(task->id.index);
  task->running = false;
  ZoneScopedN("run_completion");
}

// Context for polling
struct PollerContext {
  enum struct PollerState : char { Stopped,
                                   Disconnected,
                                   Connected,
                                   Suspended };
  char const* name = nullptr;
  uv_loop_t* loop = nullptr;
  DataProcessingDevice* device = nullptr;
  DeviceState* state = nullptr;
  fair::mq::Socket* socket = nullptr;
  InputChannelInfo* channelInfo = nullptr;
  int fd = -1;
  bool read = true;
  PollerState pollerState = PollerState::Stopped;
};

void on_socket_polled(uv_poll_t* poller, int status, int events)
{
  auto* context = (PollerContext*)poller->data;
  context->state->loopReason |= DeviceState::DATA_SOCKET_POLLED;
  switch (events) {
    case UV_READABLE: {
      ZoneScopedN("socket readable event");
      LOG(debug) << "socket polled UV_READABLE: " << context->name;
      context->state->loopReason |= DeviceState::DATA_INCOMING;
    } break;
    case UV_WRITABLE: {
      ZoneScopedN("socket writeable");
      if (context->read) {
        LOG(debug) << "socket polled UV_CONNECT" << context->name;
        uv_poll_start(poller, UV_READABLE | UV_DISCONNECT | UV_PRIORITIZED, &on_socket_polled);
        context->state->loopReason |= DeviceState::DATA_CONNECTED;
      } else {
        LOG(debug) << "socket polled UV_WRITABLE" << context->name;
        context->state->loopReason |= DeviceState::DATA_OUTGOING;
        // If the socket is writable, fairmq will handle the rest, so we can stop polling and
        // just wait for the disconnect.
        uv_poll_start(poller, UV_DISCONNECT | UV_PRIORITIZED, &on_socket_polled);
      }
      context->pollerState = PollerContext::PollerState::Connected;
    } break;
    case UV_DISCONNECT: {
      ZoneScopedN("socket disconnect");
      LOG(debug) << "socket polled UV_DISCONNECT";
    } break;
    case UV_PRIORITIZED: {
      ZoneScopedN("socket prioritized");
      LOG(debug) << "socket polled UV_PRIORITIZED";
    } break;
  }
  // We do nothing, all the logic for now stays in DataProcessingDevice::doRun()
}

void on_out_of_band_polled(uv_poll_t* poller, int status, int events)
{
  auto* context = (PollerContext*)poller->data;
  context->state->loopReason |= DeviceState::OOB_ACTIVITY;
  if (status < 0) {
    LOGP(fatal, "Error while polling {}: {}", context->name, status);
    uv_poll_start(poller, UV_WRITABLE, &on_out_of_band_polled);
  }
  switch (events) {
    case UV_READABLE: {
      ZoneScopedN("socket readable event");
      context->state->loopReason |= DeviceState::DATA_INCOMING;
      assert(context->channelInfo);
      LOGP(debug, "oob socket {} polled UV_READABLE.",
           context->name,
           context->channelInfo->hasPendingEvents);
      context->channelInfo->readPolled = true;
    } break;
    case UV_WRITABLE: {
      ZoneScopedN("socket writeable");
      if (context->read) {
        LOG(debug) << "socket polled UV_CONNECT" << context->name;
        uv_poll_start(poller, UV_READABLE | UV_DISCONNECT | UV_PRIORITIZED, &on_out_of_band_polled);
      } else {
        LOG(debug) << "socket polled UV_WRITABLE" << context->name;
        context->state->loopReason |= DeviceState::DATA_OUTGOING;
      }
    } break;
    case UV_DISCONNECT: {
      ZoneScopedN("socket disconnect");
      LOG(debug) << "socket polled UV_DISCONNECT";
      uv_poll_start(poller, UV_WRITABLE, &on_out_of_band_polled);
    } break;
    case UV_PRIORITIZED: {
      ZoneScopedN("socket prioritized");
      LOG(debug) << "socket polled UV_PRIORITIZED";
    } break;
  }
  // We do nothing, all the logic for now stays in DataProcessingDevice::doRun()
}

/// This  takes care  of initialising  the device  from its  specification. In
/// particular it needs to:
///
/// * Fetch parameters from configuration
/// * Materialize the correct callbacks for expiring records. We need to do it
///   here because the configuration is available only at this point.
/// * Invoke the actual init callback, which returns the processing callback.
void DataProcessingDevice::Init()
{
  auto ref = ServiceRegistryRef{mServiceRegistry};
  auto& context = ref.get<DataProcessorContext>();
  auto& spec = getRunningDevice(mRunningDevice, ref);
  context.statelessProcess = spec.algorithm.onProcess;
  context.statefulProcess = nullptr;
  context.error = spec.algorithm.onError;
  context.initError = spec.algorithm.onInitError;
  TracyAppInfo(spec.name.data(), spec.name.size());
  ZoneScopedN("DataProcessingDevice::Init");

  auto configStore = DeviceConfigurationHelpers::getConfiguration(mServiceRegistry, spec.name.c_str(), spec.options);
  if (configStore == nullptr) {
    std::vector<std::unique_ptr<ParamRetriever>> retrievers;
    retrievers.emplace_back(std::make_unique<FairOptionsRetriever>(GetConfig()));
    configStore = std::make_unique<ConfigParamStore>(spec.options, std::move(retrievers));
    configStore->preload();
    configStore->activate();
  }

  using boost::property_tree::ptree;

  /// Dump the configuration so that we can get it from the driver.
  for (auto& entry : configStore->store()) {
    std::stringstream ss;
    std::string str;
    if (entry.second.empty() == false) {
      boost::property_tree::json_parser::write_json(ss, entry.second, false);
      str = ss.str();
      str.pop_back(); // remove EoL
    } else {
      str = entry.second.get_value<std::string>();
    }
    std::string configString = fmt::format("[CONFIG] {}={} 1 {}", entry.first, str, configStore->provenance(entry.first.c_str())).c_str();
    mServiceRegistry.get<DriverClient>(ServiceRegistry::globalDeviceSalt()).tell(configString.c_str());
  }

  mConfigRegistry = std::make_unique<ConfigParamRegistry>(std::move(configStore));

  // Setup the error handlers for init
  if (context.initError) {
    context.initErrorHandling = [&errorCallback = context.initError,
                                 &serviceRegistry = mServiceRegistry](RuntimeErrorRef e) {
      ZoneScopedN("Error handling");
      /// FIXME: we should pass the salt in, so that the message
      ///        can access information which were stored in the stream.
      ServiceRegistryRef ref{serviceRegistry, ServiceRegistry::globalDeviceSalt()};
      auto& err = error_from_ref(e);
      LOGP(error, "Exception caught: {} ", err.what);
      demangled_backtrace_symbols(err.backtrace, err.maxBacktrace, STDERR_FILENO);
      auto& stats = ref.get<DataProcessingStats>();
      stats.updateStats({(int)ProcessingStatsId::EXCEPTION_COUNT, DataProcessingStats::Op::Add, 1});
      InitErrorContext errorContext{ref, e};
      errorCallback(errorContext);
    };
  } else {
    context.initErrorHandling = [&serviceRegistry = mServiceRegistry](RuntimeErrorRef e) {
      ZoneScopedN("Error handling");
      auto& err = error_from_ref(e);
      /// FIXME: we should pass the salt in, so that the message
      ///        can access information which were stored in the stream.
      LOGP(error, "Exception caught: {} ", err.what);
      ServiceRegistryRef ref{serviceRegistry, ServiceRegistry::globalDeviceSalt()};
      demangled_backtrace_symbols(err.backtrace, err.maxBacktrace, STDERR_FILENO);
      auto& stats = ref.get<DataProcessingStats>();
      stats.updateStats({(int)ProcessingStatsId::EXCEPTION_COUNT, DataProcessingStats::Op::Add, 1});
      exit(1);
    };
  }

  context.expirationHandlers.clear();
  context.init = spec.algorithm.onInit;
  if (context.init) {
    static bool noCatch = getenv("O2_NO_CATCHALL_EXCEPTIONS") && strcmp(getenv("O2_NO_CATCHALL_EXCEPTIONS"), "0");
    InitContext initContext{*mConfigRegistry, mServiceRegistry};

    if (noCatch) {
      try {
        context.statefulProcess = context.init(initContext);
      } catch (o2::framework::RuntimeErrorRef e) {
        ZoneScopedN("error handling");
        if (context.initErrorHandling) {
          (context.initErrorHandling)(e);
        }
      }
    } else {
      try {
        context.statefulProcess = context.init(initContext);
      } catch (std::exception& ex) {
        ZoneScopedN("error handling");
        /// Convert a standard exception to a RuntimeErrorRef
        /// Notice how this will lose the backtrace information
        /// and report the exception coming from here.
        auto e = runtime_error(ex.what());
        (context.initErrorHandling)(e);
      } catch (o2::framework::RuntimeErrorRef e) {
        ZoneScopedN("error handling");
        (context.initErrorHandling)(e);
      }
    }
  }
  auto& state = ref.get<DeviceState>();
  state.inputChannelInfos.resize(spec.inputChannels.size());
  /// Internal channels which will never create an actual message
  /// should be considered as in "Pull" mode, since we do not
  /// expect them to create any data.
  int validChannelId = 0;
  for (size_t ci = 0; ci < spec.inputChannels.size(); ++ci) {
    auto& name = spec.inputChannels[ci].name;
    if (name.find(spec.channelPrefix + "from_internal-dpl-clock") == 0) {
      state.inputChannelInfos[ci].state = InputChannelState::Pull;
      state.inputChannelInfos[ci].id = {ChannelIndex::INVALID};
      validChannelId++;
    } else {
      state.inputChannelInfos[ci].id = {validChannelId++};
    }
  }

  // Invoke the callback policy for this device.
  if (spec.callbacksPolicy.policy != nullptr) {
    InitContext initContext{*mConfigRegistry, mServiceRegistry};
    spec.callbacksPolicy.policy(mServiceRegistry.get<CallbackService>(ServiceRegistry::globalDeviceSalt()), initContext);
  }

  // Services which are stream should be initialised now
  auto* options = GetConfig();
  for (size_t si = 0; si < mStreams.size(); ++si) {
    ServiceRegistry::Salt streamSalt = ServiceRegistry::streamSalt(si + 1, ServiceRegistry::globalDeviceSalt().dataProcessorId);
    mServiceRegistry.lateBindStreamServices(state, *options, streamSalt);
  }
}

void on_signal_callback(uv_signal_t* handle, int signum)
{
  ZoneScopedN("Signal callaback");
  LOG(debug) << "Signal " << signum << " received.";
  auto* registry = (ServiceRegistry*)handle->data;
  if (!registry) {
    LOG(debug) << "No registry active. Ignoring signal";
    return;
  }
  ServiceRegistryRef ref{*registry};
  auto& state = ref.get<DeviceState>();
  auto& quotaEvaluator = ref.get<ComputingQuotaEvaluator>();
  auto& stats = ref.get<DataProcessingStats>();
  state.loopReason |= DeviceState::SIGNAL_ARRIVED;
  size_t ri = 0;
  while (ri != quotaEvaluator.mOffers.size()) {
    auto& offer = quotaEvaluator.mOffers[ri];
    // We were already offered some sharedMemory, so we
    // do not consider the offer.
    // FIXME: in principle this should account for memory
    //        available and being offered, however we
    //        want to get out of the woods for now.
    if (offer.valid && offer.sharedMemory != 0) {
      return;
    }
    ri++;
  }
  // Find the first empty offer and have 1GB of shared memory there
  for (auto& offer : quotaEvaluator.mOffers) {
    if (offer.valid == false) {
      offer.cpu = 0;
      offer.memory = 0;
      offer.sharedMemory = 1000000000;
      offer.valid = true;
      offer.user = -1;
      break;
    }
  }
  stats.updateStats({(int)ProcessingStatsId::TOTAL_SIGUSR1, DataProcessingStats::Op::Add, 1});
}

static auto toBeForwardedHeader = [](void* header) -> bool {
  // If is now possible that the record is not complete when
  // we forward it, because of a custom completion policy.
  // this means that we need to skip the empty entries in the
  // record for being forwarded.
  if (header == nullptr) {
    return false;
  }
  auto sih = o2::header::get<SourceInfoHeader*>(header);
  if (sih) {
    return false;
  }

  auto dih = o2::header::get<DomainInfoHeader*>(header);
  if (dih) {
    return false;
  }

  auto dh = o2::header::get<DataHeader*>(header);
  if (!dh) {
    return false;
  }
  auto dph = o2::header::get<DataProcessingHeader*>(header);
  if (!dph) {
    return false;
  }
  return true;
};

static auto toBeforwardedMessageSet = [](std::vector<ChannelIndex>& cachedForwardingChoices,
                                         FairMQDeviceProxy& proxy,
                                         std::unique_ptr<fair::mq::Message>& header,
                                         std::unique_ptr<fair::mq::Message>& payload,
                                         size_t total,
                                         bool consume) {
  if (header.get() == nullptr) {
    // Missing an header is not an error anymore.
    // it simply means that we did not receive the
    // given input, but we were asked to
    // consume existing, so we skip it.
    return false;
  }
  if (payload.get() == nullptr && consume == true) {
    // If the payload is not there, it means we already
    // processed it with ConsumeExisiting. Therefore we
    // need to do something only if this is the last consume.
    header.reset(nullptr);
    return false;
  }

  auto fdph = o2::header::get<DataProcessingHeader*>(header->GetData());
  if (fdph == nullptr) {
    LOG(error) << "Data is missing DataProcessingHeader";
    return false;
  }
  auto fdh = o2::header::get<DataHeader*>(header->GetData());
  if (fdh == nullptr) {
    LOG(error) << "Data is missing DataHeader";
    return false;
  }

  // We need to find the forward route only for the first
  // part of a split payload. All the others will use the same.
  // but always check if we have a sequence of multiple payloads
  if (fdh->splitPayloadIndex == 0 || fdh->splitPayloadParts <= 1 || total > 1) {
    proxy.getMatchingForwardChannelIndexes(cachedForwardingChoices, *fdh, fdph->startTime);
  }
  return cachedForwardingChoices.empty() == false;
};

// This is how we do the forwarding, i.e. we push
// the inputs which are shared between this device and others
// to the next one in the daisy chain.
// FIXME: do it in a smarter way than O(N^2)
static auto forwardInputs = [](ServiceRegistryRef registry, TimesliceSlot slot, std::vector<MessageSet>& currentSetOfInputs,
                               TimesliceIndex::OldestOutputInfo oldestTimeslice, bool copy, bool consume = true) {
  ZoneScopedN("forward inputs");
  auto& proxy = registry.get<FairMQDeviceProxy>();
  // we collect all messages per forward in a map and send them together
  std::vector<fair::mq::Parts> forwardedParts;
  forwardedParts.resize(proxy.getNumForwards());
  std::vector<ChannelIndex> cachedForwardingChoices{};

  for (size_t ii = 0, ie = currentSetOfInputs.size(); ii < ie; ++ii) {
    auto& messageSet = currentSetOfInputs[ii];
    // In case the messageSet is empty, there is nothing to be done.
    if (messageSet.size() == 0) {
      continue;
    }
    if (!toBeForwardedHeader(messageSet.header(0)->GetData())) {
      continue;
    }
    cachedForwardingChoices.clear();

    for (size_t pi = 0; pi < currentSetOfInputs[ii].size(); ++pi) {
      auto& messageSet = currentSetOfInputs[ii];
      auto& header = messageSet.header(pi);
      auto& payload = messageSet.payload(pi);
      auto total = messageSet.getNumberOfPayloads(pi);

      if (!toBeforwardedMessageSet(cachedForwardingChoices, proxy, header, payload, total, consume)) {
        continue;
      }

      // In case of more than one forward route, we need to copy the message.
      // This will eventually use the same mamory if running with the same backend.
      if (cachedForwardingChoices.size() > 1) {
        copy = true;
      }
      if (copy) {
        for (auto& cachedForwardingChoice : cachedForwardingChoices) {
          auto&& newHeader = header->GetTransport()->CreateMessage();
          newHeader->Copy(*header);
          forwardedParts[cachedForwardingChoice.value].AddPart(std::move(newHeader));

          for (size_t payloadIndex = 0; payloadIndex < messageSet.getNumberOfPayloads(pi); ++payloadIndex) {
            auto&& newPayload = header->GetTransport()->CreateMessage();
            newPayload->Copy(*messageSet.payload(pi, payloadIndex));
            forwardedParts[cachedForwardingChoice.value].AddPart(std::move(newPayload));
          }
        }
      } else {
        forwardedParts[cachedForwardingChoices.back().value].AddPart(std::move(messageSet.header(pi)));
        for (size_t payloadIndex = 0; payloadIndex < messageSet.getNumberOfPayloads(pi); ++payloadIndex) {
          forwardedParts[cachedForwardingChoices.back().value].AddPart(std::move(messageSet.payload(pi, payloadIndex)));
        }
      }
    }
  }
  LOG(debug) << "Forwarding " << forwardedParts.size() << " messages";
  for (int fi = 0; fi < proxy.getNumForwardChannels(); fi++) {
    if (forwardedParts[fi].Size() == 0) {
      continue;
    }
    auto channel = proxy.getForwardChannel(ChannelIndex{fi});
    LOG(debug) << "Forwarding to " << channel->GetName() << " " << fi;
    // in DPL we are using subchannel 0 only
    auto& parts = forwardedParts[fi];
    int timeout = 30000;
    auto res = channel->Send(parts, timeout);
    if (res == (size_t)fair::mq::TransferCode::timeout) {
      LOGP(warning, "Timed out sending after {}s. Downstream backpressure detected on {}.", timeout / 1000, channel->GetName());
      channel->Send(parts);
      LOGP(info, "Downstream backpressure on {} recovered.", channel->GetName());
    } else if (res == (size_t)fair::mq::TransferCode::error) {
      LOGP(fatal, "Error while sending on channel {}", channel->GetName());
    }
  }

  auto& asyncQueue = registry.get<AsyncQueue>();
  auto& decongestion = registry.get<DecongestionService>();
  LOG(debug) << "Queuing forwarding oldestPossible " << oldestTimeslice.timeslice.value;
  AsyncQueueHelpers::post(
    asyncQueue, decongestion.oldestPossibleTimesliceTask, [&proxy, &decongestion, registry, oldestTimeslice]() {
      // DataProcessingHelpers::broadcastOldestPossibleTimeslice(proxy, oldestTimeslice.timeslice.value);
      if (oldestTimeslice.timeslice.value <= decongestion.lastTimeslice) {
        LOG(debug) << "Not sending already sent oldest possible timeslice " << oldestTimeslice.timeslice.value;
        return;
      }
      for (int fi = 0; fi < proxy.getNumForwardChannels(); fi++) {
        auto& info = proxy.getForwardChannelInfo(ChannelIndex{fi});
        auto& state = proxy.getForwardChannelState(ChannelIndex{fi});
        // TODO: this we could cache in the proxy at the bind moment.
        if (info.channelType != ChannelAccountingType::DPL) {
          LOG(debug) << "Skipping channel";
          continue;
        }
        if (DataProcessingHelpers::sendOldestPossibleTimeframe(info, state, oldestTimeslice.timeslice.value)) {
          LOGP(debug, "Forwarding to channel {} oldest possible timeslice {}, prio 20", info.name, oldestTimeslice.timeslice.value);
        }
      }
    },
    oldestTimeslice.timeslice, -1);
  LOG(debug) << "Forwarding done";
};
extern volatile int region_read_global_dummy_variable;
volatile int region_read_global_dummy_variable;

/// Invoke the callbacks for the mPendingRegionInfos
void handleRegionCallbacks(ServiceRegistryRef registry, std::vector<fair::mq::RegionInfo>& infos)
{
  if (infos.empty() == false) {
    std::vector<fair::mq::RegionInfo> toBeNotified;
    toBeNotified.swap(infos); // avoid any MT issue.
    static bool dummyRead = getenv("DPL_DEBUG_MAP_ALL_SHM_REGIONS") && atoi(getenv("DPL_DEBUG_MAP_ALL_SHM_REGIONS"));
    for (auto const& info : toBeNotified) {
      if (dummyRead) {
        for (size_t i = 0; i < info.size / sizeof(region_read_global_dummy_variable); i += 4096 / sizeof(region_read_global_dummy_variable)) {
          region_read_global_dummy_variable = ((int*)info.ptr)[i];
        }
      }
      registry.get<CallbackService>().call<CallbackService::Id::RegionInfoCallback>(info);
    }
  }
}

namespace
{
void on_awake_main_thread(uv_async_t* handle)
{
  auto* state = (DeviceState*)handle->data;
  state->loopReason |= DeviceState::ASYNC_NOTIFICATION;
}
} // namespace

void DataProcessingDevice::initPollers()
{
  auto ref = ServiceRegistryRef{mServiceRegistry};
  auto& deviceContext = ref.get<DeviceContext>();
  auto& context = ref.get<DataProcessorContext>();
  auto& spec = ref.get<DeviceSpec const>();
  auto& state = ref.get<DeviceState>();
  // We add a timer only in case a channel poller is not there.
  if ((context.statefulProcess != nullptr) || (context.statelessProcess != nullptr)) {
    for (auto& [channelName, channel] : GetChannels()) {
      InputChannelInfo* channelInfo;
      for (size_t ci = 0; ci < spec.inputChannels.size(); ++ci) {
        auto& channelSpec = spec.inputChannels[ci];
        channelInfo = &state.inputChannelInfos[ci];
        if (channelSpec.name != channelName) {
          continue;
        }
        channelInfo->channel = &this->GetChannel(channelName, 0);
        break;
      }
      if ((channelName.rfind("from_internal-dpl", 0) == 0) &&
          (channelName.rfind("from_internal-dpl-aod", 0) != 0) &&
          (channelName.rfind("from_internal-dpl-ccdb-backend", 0) != 0) &&
          (channelName.rfind("from_internal-dpl-injected", 0)) != 0) {
        LOGP(detail, "{} is an internal channel. Skipping as no input will come from there.", channelName);
        continue;
      }
      // We only watch receiving sockets.
      if (channelName.rfind("from_" + spec.name + "_", 0) == 0) {
        LOGP(detail, "{} is to send data. Not polling.", channelName);
        continue;
      }

      if (channelName.rfind("from_", 0) != 0) {
        LOGP(detail, "{} is not a DPL socket. Not polling.", channelName);
        continue;
      }

      // We assume there is always a ZeroMQ socket behind.
      int zmq_fd = 0;
      size_t zmq_fd_len = sizeof(zmq_fd);
      // FIXME: I should probably save those somewhere... ;-)
      auto* poller = (uv_poll_t*)malloc(sizeof(uv_poll_t));
      channel[0].GetSocket().GetOption("fd", &zmq_fd, &zmq_fd_len);
      if (zmq_fd == 0) {
        LOG(error) << "Cannot get file descriptor for channel." << channelName;
        continue;
      }
      LOGP(detail, "Polling socket for {}", channelName);
      auto* pCtx = (PollerContext*)malloc(sizeof(PollerContext));
      pCtx->name = strdup(channelName.c_str());
      pCtx->loop = state.loop;
      pCtx->device = this;
      pCtx->state = &state;
      pCtx->fd = zmq_fd;
      assert(channelInfo != nullptr);
      pCtx->channelInfo = channelInfo;
      pCtx->socket = &channel[0].GetSocket();
      pCtx->read = true;
      poller->data = pCtx;
      uv_poll_init(state.loop, poller, zmq_fd);
      if (channelName.rfind("from_", 0) != 0) {
        LOGP(detail, "{} is an out of band channel.", channelName);
        state.activeOutOfBandPollers.push_back(poller);
      } else {
        channelInfo->pollerIndex = state.activeInputPollers.size();
        state.activeInputPollers.push_back(poller);
      }
    }
    // In case we do not have any input channel and we do not have
    // any timers or signal watchers we still wake up whenever we can send data to downstream
    // devices to allow for enumerations.
    if (state.activeInputPollers.empty() &&
        state.activeOutOfBandPollers.empty() &&
        state.activeTimers.empty() &&
        state.activeSignals.empty()) {
      // FIXME: this is to make sure we do not reset the output timer
      // for readout proxies or similar. In principle this should go once
      // we move to OutOfBand InputSpec.
      if (state.inputChannelInfos.empty()) {
        LOGP(detail, "No input channels. Setting exit transition timeout to 0.");
        deviceContext.exitTransitionTimeout = 0;
      }
      for (auto& [channelName, channel] : GetChannels()) {
        if (channelName.rfind(spec.channelPrefix + "from_internal-dpl", 0) == 0) {
          LOGP(detail, "{} is an internal channel. Not polling.", channelName);
          continue;
        }
        if (channelName.rfind(spec.channelPrefix + "from_" + spec.name + "_", 0) == 0) {
          LOGP(detail, "{} is an out of band channel. Not polling for output.", channelName);
          continue;
        }
        // We assume there is always a ZeroMQ socket behind.
        int zmq_fd = 0;
        size_t zmq_fd_len = sizeof(zmq_fd);
        // FIXME: I should probably save those somewhere... ;-)
        auto* poller = (uv_poll_t*)malloc(sizeof(uv_poll_t));
        channel[0].GetSocket().GetOption("fd", &zmq_fd, &zmq_fd_len);
        if (zmq_fd == 0) {
          LOGP(error, "Cannot get file descriptor for channel {}", channelName);
          continue;
        }
        LOG(detail) << "Polling socket for " << channel[0].GetName();
        // FIXME: leak
        auto* pCtx = (PollerContext*)malloc(sizeof(PollerContext));
        pCtx->name = strdup(channelName.c_str());
        pCtx->loop = state.loop;
        pCtx->device = this;
        pCtx->state = &state;
        pCtx->fd = zmq_fd;
        pCtx->read = false;
        poller->data = pCtx;
        uv_poll_init(state.loop, poller, zmq_fd);
        state.activeOutputPollers.push_back(poller);
      }
    }
  } else {
    LOGP(detail, "This is a fake device so we exit after the first iteration.");
    deviceContext.exitTransitionTimeout = 0;
    // This is a fake device, so we can request to exit immediately
    ServiceRegistryRef ref{mServiceRegistry};
    ref.get<ControlService>().readyToQuit(QuitRequest::Me);
    // A two second timer to stop internal devices which do not want to
    auto* timer = (uv_timer_t*)malloc(sizeof(uv_timer_t));
    uv_timer_init(state.loop, timer);
    timer->data = &state;
    uv_update_time(state.loop);
    uv_timer_start(timer, on_idle_timer, 2000, 2000);
    state.activeTimers.push_back(timer);
  }
}

void DataProcessingDevice::startPollers()
{
  auto ref = ServiceRegistryRef{mServiceRegistry};
  auto& deviceContext = ref.get<DeviceContext>();
  auto& state = ref.get<DeviceState>();

  for (auto& poller : state.activeInputPollers) {
    uv_poll_start(poller, UV_WRITABLE, &on_socket_polled);
    ((PollerContext*)poller->data)->pollerState = PollerContext::PollerState::Disconnected;
  }
  for (auto& poller : state.activeOutOfBandPollers) {
    uv_poll_start(poller, UV_WRITABLE, &on_out_of_band_polled);
    ((PollerContext*)poller->data)->pollerState = PollerContext::PollerState::Disconnected;
  }
  for (auto& poller : state.activeOutputPollers) {
    uv_poll_start(poller, UV_WRITABLE, &on_socket_polled);
    ((PollerContext*)poller->data)->pollerState = PollerContext::PollerState::Disconnected;
  }

  deviceContext.gracePeriodTimer = (uv_timer_t*)malloc(sizeof(uv_timer_t));
  deviceContext.gracePeriodTimer->data = &state;
  uv_timer_init(state.loop, deviceContext.gracePeriodTimer);
}

void DataProcessingDevice::stopPollers()
{
  auto ref = ServiceRegistryRef{mServiceRegistry};
  auto& deviceContext = ref.get<DeviceContext>();
  auto& state = ref.get<DeviceState>();
  LOGP(detail, "Stopping {} input pollers", state.activeInputPollers.size());
  for (auto& poller : state.activeInputPollers) {
    uv_poll_stop(poller);
    ((PollerContext*)poller->data)->pollerState = PollerContext::PollerState::Stopped;
  }
  LOGP(detail, "Stopping {} out of band pollers", state.activeOutOfBandPollers.size());
  for (auto& poller : state.activeOutOfBandPollers) {
    uv_poll_stop(poller);
    ((PollerContext*)poller->data)->pollerState = PollerContext::PollerState::Stopped;
  }
  LOGP(detail, "Stopping {} output pollers", state.activeOutOfBandPollers.size());
  for (auto& poller : state.activeOutputPollers) {
    uv_poll_stop(poller);
    ((PollerContext*)poller->data)->pollerState = PollerContext::PollerState::Stopped;
  }

  uv_timer_stop(deviceContext.gracePeriodTimer);
  free(deviceContext.gracePeriodTimer);
  deviceContext.gracePeriodTimer = nullptr;
}

void DataProcessingDevice::InitTask()
{
  auto ref = ServiceRegistryRef{mServiceRegistry};
  auto& deviceContext = ref.get<DeviceContext>();
  auto& context = ref.get<DataProcessorContext>();
  auto& spec = getRunningDevice(mRunningDevice, mServiceRegistry);
  auto distinct = DataRelayerHelpers::createDistinctRouteIndex(spec.inputs);
  auto& state = ref.get<DeviceState>();
  int i = 0;
  for (auto& di : distinct) {
    auto& route = spec.inputs[di];
    if (route.configurator.has_value() == false) {
      i++;
      continue;
    }
    ExpirationHandler handler{
      .name = route.configurator->name,
      .routeIndex = RouteIndex{i++},
      .lifetime = route.matcher.lifetime,
      .creator = route.configurator->creatorConfigurator(state, mServiceRegistry, *mConfigRegistry),
      .checker = route.configurator->danglingConfigurator(state, *mConfigRegistry),
      .handler = route.configurator->expirationConfigurator(state, *mConfigRegistry)};
    context.expirationHandlers.emplace_back(std::move(handler));
  }

  if (state.awakeMainThread == nullptr) {
    state.awakeMainThread = (uv_async_t*)malloc(sizeof(uv_async_t));
    state.awakeMainThread->data = &state;
    uv_async_init(state.loop, state.awakeMainThread, on_awake_main_thread);
  }

  deviceContext.expectedRegionCallbacks = std::stoi(fConfig->GetValue<std::string>("expected-region-callbacks"));
  deviceContext.exitTransitionTimeout = std::stoi(fConfig->GetValue<std::string>("exit-transition-timeout"));

  for (auto& channel : GetChannels()) {
    channel.second.at(0).Transport()->SubscribeToRegionEvents([&context = deviceContext,
                                                               &registry = mServiceRegistry,
                                                               &pendingRegionInfos = mPendingRegionInfos,
                                                               &regionInfoMutex = mRegionInfoMutex](fair::mq::RegionInfo info) {
      std::lock_guard<std::mutex> lock(regionInfoMutex);
      LOG(detail) << ">>> Region info event" << info.event;
      LOG(detail) << "id: " << info.id;
      LOG(detail) << "ptr: " << info.ptr;
      LOG(detail) << "size: " << info.size;
      LOG(detail) << "flags: " << info.flags;
      // Now we check for pending events with the mutex,
      // so the lines below are atomic.
      pendingRegionInfos.push_back(info);
      context.expectedRegionCallbacks -= 1;
      // We always want to handle these on the main loop,
      // so we awake it.
      ServiceRegistryRef ref{registry};
      uv_async_send(ref.get<DeviceState>().awakeMainThread);
    });
  }

  // Add a signal manager for SIGUSR1 so that we can force
  // an event from the outside, making sure that the event loop can
  // be unblocked (e.g. by a quitting DPL driver) even when there
  // is no data pending to be processed.
  if (deviceContext.sigusr1Handle == nullptr) {
    deviceContext.sigusr1Handle = (uv_signal_t*)malloc(sizeof(uv_signal_t));
    deviceContext.sigusr1Handle->data = &mServiceRegistry;
    uv_signal_init(state.loop, deviceContext.sigusr1Handle);
    uv_signal_start(deviceContext.sigusr1Handle, on_signal_callback, SIGUSR1);
  }
  // If there is any signal, we want to make sure they are active
  for (auto& handle : state.activeSignals) {
    handle->data = &state;
  }
  // When we start, we must make sure that we do listen to the signal
  deviceContext.sigusr1Handle->data = &mServiceRegistry;

  /// Initialise the pollers
  DataProcessingDevice::initPollers();

  // Whenever we InitTask, we consider as if the previous iteration
  // was successful, so that even if there is no timer or receiving
  // channel, we can still start an enumeration.
  mWasActive = true;

  // We should be ready to run here. Therefore we copy all the
  // required parts in the DataProcessorContext. Eventually we should
  // do so on a per thread basis, with fine grained locks.
  // FIXME: this should not use ServiceRegistry::threadSalt, but
  // more a ServiceRegistry::globalDataProcessorSalt(N) where
  // N is the number of the multiplexed data processor.
  // We will get there.
  this->fillContext(mServiceRegistry.get<DataProcessorContext>(ServiceRegistry::globalDeviceSalt()), deviceContext);

  auto hasPendingEvents = [&mutex = mRegionInfoMutex, &pendingRegionInfos = mPendingRegionInfos](DeviceContext& deviceContext) {
    std::lock_guard<std::mutex> lock(mutex);
    return (pendingRegionInfos.empty() == false) || deviceContext.expectedRegionCallbacks > 0;
  };
  /// We now run an event loop also in InitTask. This is needed to:
  /// * Make sure region registration callbacks are invoked
  /// on the main thread.
  /// * Wait for enough callbacks to be delivered before moving to START
  while (hasPendingEvents(deviceContext)) {
    // Wait for the callback to signal its done, so that we do not busy wait.
    uv_run(state.loop, UV_RUN_ONCE);
    // Handle callbacks if any
    {
      std::lock_guard<std::mutex> lock(mRegionInfoMutex);
      handleRegionCallbacks(mServiceRegistry, mPendingRegionInfos);
    }
  }
}

void DataProcessingDevice::fillContext(DataProcessorContext& context, DeviceContext& deviceContext)
{
  context.wasActive = &mWasActive;

  context.isSink = false;
  // If nothing is a sink, the rate limiting simply does not trigger.
  bool enableRateLimiting = std::stoi(fConfig->GetValue<std::string>("timeframes-rate-limit"));

  auto ref = ServiceRegistryRef{mServiceRegistry};
  auto& spec = ref.get<DeviceSpec const>();

  // The policy is now allowed to state the default.
  context.balancingInputs = spec.completionPolicy.balanceChannels;
  // This is needed because the internal injected dummy sink should not
  // try to balance inputs unless the rate limiting is requested.
  if (enableRateLimiting == false && spec.name == "internal-dpl-injected-dummy-sink") {
    context.balancingInputs = false;
  }
  if (enableRateLimiting) {
    for (auto& spec : spec.outputs) {
      if (spec.matcher.binding.value == "dpl-summary") {
        context.isSink = true;
        break;
      }
    }
  }

  context.registry = &mServiceRegistry;
  /// Callback for the error handling
  /// FIXME: move erro handling to a service?
  if (context.error != nullptr) {
    context.errorHandling = [&errorCallback = context.error,
                             &serviceRegistry = mServiceRegistry](RuntimeErrorRef e, InputRecord& record) {
      ZoneScopedN("Error handling");
      /// FIXME: we should pass the salt in, so that the message
      ///        can access information which were stored in the stream.
      ServiceRegistryRef ref{serviceRegistry, ServiceRegistry::globalDeviceSalt()};
      auto& err = error_from_ref(e);
      LOGP(error, "Exception caught: {} ", err.what);
      demangled_backtrace_symbols(err.backtrace, err.maxBacktrace, STDERR_FILENO);
      auto& stats = ref.get<DataProcessingStats>();
      stats.updateStats({(int)ProcessingStatsId::EXCEPTION_COUNT, DataProcessingStats::Op::Add, 1});
      ErrorContext errorContext{record, ref, e};
      errorCallback(errorContext);
    };
  } else {
    context.errorHandling = [&errorPolicy = mProcessingPolicies.error,
                             &serviceRegistry = mServiceRegistry](RuntimeErrorRef e, InputRecord& record) {
      ZoneScopedN("Error handling");
      auto& err = error_from_ref(e);
      /// FIXME: we should pass the salt in, so that the message
      ///        can access information which were stored in the stream.
      LOGP(error, "Exception caught: {} ", err.what);
      ServiceRegistryRef ref{serviceRegistry, ServiceRegistry::globalDeviceSalt()};
      demangled_backtrace_symbols(err.backtrace, err.maxBacktrace, STDERR_FILENO);
      auto& stats = ref.get<DataProcessingStats>();
      stats.updateStats({(int)ProcessingStatsId::EXCEPTION_COUNT, DataProcessingStats::Op::Add, 1});
      switch (errorPolicy) {
        case TerminationPolicy::QUIT:
          throw e;
        default:
          break;
      }
    };
  }

  auto decideEarlyForward = [&context, &spec, this]() -> bool {
    /// We must make sure there is no optional
    /// if we want to optimize the forwarding
    bool canForwardEarly = (spec.forwards.empty() == false) && mProcessingPolicies.earlyForward != EarlyForwardPolicy::NEVER;
    bool onlyConditions = true;
    bool overriddenEarlyForward = false;
    for (auto& forwarded : spec.forwards) {
      if (forwarded.matcher.lifetime != Lifetime::Condition) {
        onlyConditions = false;
      }
      if (strncmp(DataSpecUtils::asConcreteOrigin(forwarded.matcher).str, "AOD", 3) == 0) {
        context.canForwardEarly = false;
        overriddenEarlyForward = true;
        LOG(detail) << "Cannot forward early because of AOD input: " << DataSpecUtils::describe(forwarded.matcher);
        break;
      }
      if (DataSpecUtils::partialMatch(forwarded.matcher, o2::header::DataDescription{"RAWDATA"}) && mProcessingPolicies.earlyForward == EarlyForwardPolicy::NORAW) {
        context.canForwardEarly = false;
        overriddenEarlyForward = true;
        LOG(detail) << "Cannot forward early because of RAWDATA input: " << DataSpecUtils::describe(forwarded.matcher);
        break;
      }
      if (forwarded.matcher.lifetime == Lifetime::Optional) {
        context.canForwardEarly = false;
        overriddenEarlyForward = true;
        LOG(detail) << "Cannot forward early because of Optional input: " << DataSpecUtils::describe(forwarded.matcher);
        break;
      }
    }
    if (!overriddenEarlyForward && onlyConditions) {
      context.canForwardEarly = true;
      LOG(detail) << "Enabling early forwarding because only conditions to be forwarded";
    }
    return canForwardEarly;
  };
  context.canForwardEarly = decideEarlyForward();
}

void DataProcessingDevice::PreRun()
{
  auto ref = ServiceRegistryRef{mServiceRegistry};
  auto& state = ref.get<DeviceState>();
  state.quitRequested = false;
  state.streaming = StreamingState::Streaming;
  for (auto& info : state.inputChannelInfos) {
    if (info.state != InputChannelState::Pull) {
      info.state = InputChannelState::Running;
    }
  }

  // Catch callbacks which fail before we start.
  // Notice that when running multiple dataprocessors
  // we should probably allow expendable ones to fail.
  try {
    auto& dpContext = ref.get<DataProcessorContext>();
    dpContext.preStartCallbacks(ref);
    for (size_t i = 0; i < mStreams.size(); ++i) {
      auto streamRef = ServiceRegistryRef{mServiceRegistry, ServiceRegistry::globalStreamSalt(i + 1)};
      auto& context = streamRef.get<StreamContext>();
      context.preStartStreamCallbacks(streamRef);
    }
  } catch (std::exception& e) {
    LOGP(error, "Exception caught: {} ", e.what());
    throw;
  } catch (o2::framework::RuntimeErrorRef& e) {
    auto& err = error_from_ref(e);
    LOGP(error, "Exception caught: {} ", err.what);
    throw;
  } catch (...) {
    throw;
  }

  ref.get<CallbackService>().call<CallbackService::Id::Start>();
  startPollers();

  // Raise to 1 when we are ready to start processing
  using o2::monitoring::Metric;
  using o2::monitoring::Monitoring;
  using o2::monitoring::tags::Key;
  using o2::monitoring::tags::Value;

  auto& monitoring = ref.get<Monitoring>();
  monitoring.send(Metric{(uint64_t)1, "device_state"}.addTag(Key::Subsystem, Value::DPL));
}

void DataProcessingDevice::PostRun()
{
  ServiceRegistryRef ref{mServiceRegistry};
  // Raise to 1 when we are ready to start processing
  using o2::monitoring::Metric;
  using o2::monitoring::Monitoring;
  using o2::monitoring::tags::Key;
  using o2::monitoring::tags::Value;

  auto& monitoring = ref.get<Monitoring>();
  monitoring.send(Metric{(uint64_t)0, "device_state"}.addTag(Key::Subsystem, Value::DPL));

  stopPollers();
  ref.get<CallbackService>().call<CallbackService::Id::Stop>();
  auto& dpContext = ref.get<DataProcessorContext>();
  dpContext.postStopCallbacks(ref);
}

void DataProcessingDevice::Reset()
{
  ServiceRegistryRef ref{mServiceRegistry};
  ref.get<CallbackService>().call<CallbackService::Id::Reset>();
}

void DataProcessingDevice::Run()
{
  ServiceRegistryRef ref{mServiceRegistry};
  auto& state = ref.get<DeviceState>();
  state.loopReason = DeviceState::LoopReason::FIRST_LOOP;
  bool firstLoop = true;
  while (state.transitionHandling != TransitionHandlingState::Expired) {
    if (state.nextFairMQState.empty() == false) {
      (void)this->ChangeState(state.nextFairMQState.back());
      state.nextFairMQState.pop_back();
    }
    // Notify on the main thread the new region callbacks, making sure
    // no callback is issued if there is something still processing.
    {
      std::lock_guard<std::mutex> lock(mRegionInfoMutex);
      handleRegionCallbacks(mServiceRegistry, mPendingRegionInfos);
    }
    // This will block for the correct delay (or until we get data
    // on a socket). We also do not block on the first iteration
    // so that devices which do not have a timer can still start an
    // enumeration.
    {
      ZoneScopedN("uv idle");
      TracyPlot("past activity", (int64_t)mWasActive);
      ServiceRegistryRef ref{mServiceRegistry};
      ref.get<DriverClient>().flushPending(mServiceRegistry);
      auto shouldNotWait = (mWasActive &&
                            (state.streaming != StreamingState::Idle) && (state.activeSignals.empty())) ||
                           (state.streaming == StreamingState::EndOfStreaming);
      if (firstLoop) {
        shouldNotWait = true;
        firstLoop = false;
      }
      if (mWasActive) {
        state.loopReason |= DeviceState::LoopReason::PREVIOUSLY_ACTIVE;
      }
      if (NewStatePending()) {
        shouldNotWait = true;
        state.loopReason |= DeviceState::LoopReason::NEW_STATE_PENDING;
      }
      if (state.transitionHandling == TransitionHandlingState::NoTransition && NewStatePending()) {
        state.transitionHandling = TransitionHandlingState::Requested;
        auto& deviceContext = ref.get<DeviceContext>();
        auto timeout = deviceContext.exitTransitionTimeout;
        // Check if we only have timers
        bool onlyTimers = true;
        auto& spec = ref.get<DeviceSpec const>();
        for (auto& route : spec.inputs) {
          if (route.matcher.lifetime != Lifetime::Timer) {
            onlyTimers = false;
            break;
          }
        }
        if (onlyTimers) {
          state.streaming = StreamingState::EndOfStreaming;
        }
        if (timeout != 0 && state.streaming != StreamingState::Idle) {
          state.transitionHandling = TransitionHandlingState::Requested;
          ref.get<CallbackService>().call<CallbackService::Id::ExitRequested>(ServiceRegistryRef{ref});
          uv_update_time(state.loop);
          uv_timer_start(deviceContext.gracePeriodTimer, on_transition_requested_expired, timeout * 1000, 0);
          if (mProcessingPolicies.termination == TerminationPolicy::QUIT) {
            LOGP(info, "New state requested. Waiting for {} seconds before quitting.", timeout);
          } else {
            LOGP(info, "New state requested. Waiting for {} seconds before switching to READY state.", timeout);
          }
        } else {
          state.transitionHandling = TransitionHandlingState::Expired;
          if (timeout == 0 && mProcessingPolicies.termination == TerminationPolicy::QUIT) {
            LOGP(info, "New state requested. No timeout set, quitting immediately as per --completion-policy");
          } else if (timeout == 0 && mProcessingPolicies.termination != TerminationPolicy::QUIT) {
            LOGP(info, "New state requested. No timeout set, switching to READY state immediately");
          } else if (mProcessingPolicies.termination == TerminationPolicy::QUIT) {
            LOGP(info, "New state pending and we are already idle, quitting immediately as per --completion-policy");
          } else {
            LOGP(info, "New state pending and we are already idle, switching to READY immediately.");
          }
        }
      }
      // If we are Idle, we can then consider the transition to be expired.
      if (state.transitionHandling == TransitionHandlingState::Requested && state.streaming == StreamingState::Idle) {
        state.transitionHandling = TransitionHandlingState::Expired;
      }
      TracyPlot("shouldNotWait", (int)shouldNotWait);
      if (state.severityStack.empty() == false) {
        fair::Logger::SetConsoleSeverity((fair::Severity)state.severityStack.back());
        state.severityStack.pop_back();
      }
      // for (auto &info : mDeviceContext.state->inputChannelInfos)  {
      //   shouldNotWait |= info.readPolled;
      // }
      state.loopReason = DeviceState::NO_REASON;
      state.firedTimers.clear();
      if ((state.tracingFlags & DeviceState::LoopReason::TRACE_CALLBACKS) != 0) {
        state.severityStack.push_back((int)fair::Logger::GetConsoleSeverity());
        fair::Logger::SetConsoleSeverity(fair::Severity::trace);
      }
      // Run the asynchronous queue just before sleeping again, so that:
      // - we can trigger further events from the queue
      // - we can guarantee this is the last thing we do in the loop (
      //   assuming no one else is adding to the queue before this point).
      auto onDrop = [&registry = mServiceRegistry](TimesliceSlot slot, std::vector<MessageSet>& dropped, TimesliceIndex::OldestOutputInfo oldestOutputInfo) {
        LOGP(debug, "Dropping message from slot {}. Forwarding as needed.", slot.index);
        ServiceRegistryRef ref{registry};
        ref.get<AsyncQueue>();
        ref.get<DecongestionService>();
        ref.get<DataRelayer>();
        // Get the current timeslice for the slot.
        auto& variables = ref.get<TimesliceIndex>().getVariablesForSlot(slot);
        VariableContextHelpers::getTimeslice(variables);
        forwardInputs(registry, slot, dropped, oldestOutputInfo, false, true);
      };
      auto& relayer = ref.get<DataRelayer>();
      relayer.prunePending(onDrop);
      auto& queue = ref.get<AsyncQueue>();
      auto oldestPossibleTimeslice = relayer.getOldestPossibleOutput();
      AsyncQueueHelpers::run(queue, {oldestPossibleTimeslice.timeslice.value});
      if (shouldNotWait == false) {
        auto& dpContext = ref.get<DataProcessorContext>();
        dpContext.preLoopCallbacks(ref);
      }
      uv_run(state.loop, shouldNotWait ? UV_RUN_NOWAIT : UV_RUN_ONCE);
      if ((state.loopReason & state.tracingFlags) != 0) {
        state.severityStack.push_back((int)fair::Logger::GetConsoleSeverity());
        fair::Logger::SetConsoleSeverity(fair::Severity::trace);
      } else if (state.severityStack.empty() == false) {
        fair::Logger::SetConsoleSeverity((fair::Severity)state.severityStack.back());
        state.severityStack.pop_back();
      }
      TracyPlot("loopReason", (int64_t)(uint64_t)state.loopReason);
      LOGP(debug, "Loop reason mask {:b} & {:b} = {:b}",
           state.loopReason, state.tracingFlags,
           state.loopReason & state.tracingFlags);

      if ((state.loopReason & DeviceState::LoopReason::OOB_ACTIVITY) != 0) {
        LOGP(debug, "We were awakened by a OOB event. Rescanning everything.");
        relayer.rescan();
      }

      if (!state.pendingOffers.empty()) {
        ref.get<ComputingQuotaEvaluator>().updateOffers(state.pendingOffers, uv_now(state.loop));
      }
    }

    // Notify on the main thread the new region callbacks, making sure
    // no callback is issued if there is something still processing.
    // Notice that we still need to perform callbacks also after
    // the socket epolled, because otherwise we would end up serving
    // the callback after the first data arrives is the system is too
    // fast to transition from Init to Run.
    {
      std::lock_guard<std::mutex> lock(mRegionInfoMutex);
      handleRegionCallbacks(mServiceRegistry, mPendingRegionInfos);
    }

    assert(mStreams.size() == mHandles.size());
    /// Decide which task to use
    TaskStreamRef streamRef{-1};
    for (size_t ti = 0; ti < mStreams.size(); ti++) {
      auto& taskInfo = mStreams[ti];
      if (taskInfo.running) {
        continue;
      }
      // Stream 0 is for when we run in
      streamRef.index = ti;
    }
    using o2::monitoring::Metric;
    using o2::monitoring::Monitoring;
    using o2::monitoring::tags::Key;
    using o2::monitoring::tags::Value;
    // We have an empty stream, let's check if we have enough
    // resources for it to run something
    if (streamRef.index != -1) {
      // Synchronous execution of the callbacks. This will be moved in the
      // moved in the on_socket_polled once we have threading in place.
      uv_work_t& handle = mHandles[streamRef.index];
      TaskStreamInfo& stream = mStreams[streamRef.index];
      handle.data = &mStreams[streamRef.index];

      static std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats const& stats)> reportExpiredOffer = [&registry = mServiceRegistry](ComputingQuotaOffer const& offer, ComputingQuotaStats const& stats) {
        ServiceRegistryRef ref{registry};
        auto& dpStats = ref.get<DataProcessingStats>();
        dpStats.updateStats({static_cast<short>(ProcessingStatsId::RESOURCE_OFFER_EXPIRED), DataProcessingStats::Op::Set, stats.totalExpiredOffers});
        dpStats.updateStats({static_cast<short>(ProcessingStatsId::ARROW_BYTES_EXPIRED), DataProcessingStats::Op::Set, stats.totalExpiredBytes});
        dpStats.processCommandQueue();
      };
      auto ref = ServiceRegistryRef{mServiceRegistry};

      // Deciding wether to run or not can be done by passing a request to
      // the evaluator. In this case, the request is always satisfied and
      // we run on whatever resource is available.
      auto& spec = ref.get<DeviceSpec const>();
      bool enough = ref.get<ComputingQuotaEvaluator>().selectOffer(streamRef.index, spec.resourcePolicy.request, uv_now(state.loop));

      if (enough) {
        stream.id = streamRef;
        stream.running = true;
        stream.registry = &mServiceRegistry;
#ifdef DPL_ENABLE_THREADING
        stream.task.data = &handle;
        uv_queue_work(state.loop, &stream.task, run_callback, run_completion);
#else
        run_callback(&handle);
        run_completion(&handle, 0);
#endif
      } else {
        auto ref = ServiceRegistryRef{mServiceRegistry};
        ref.get<ComputingQuotaEvaluator>().handleExpired(reportExpiredOffer);
        mWasActive = false;
      }
    } else {
      mWasActive = false;
    }
    FrameMark;
  }
  auto& spec = ref.get<DeviceSpec const>();
  /// Cleanup messages which are still pending on exit.
  for (size_t ci = 0; ci < spec.inputChannels.size(); ++ci) {
    auto& info = state.inputChannelInfos[ci];
    info.parts.fParts.clear();
  }
  state.transitionHandling = TransitionHandlingState::NoTransition;
}

/// We drive the state loop ourself so that we will be able to support
/// non-data triggers like those which are time based.
void DataProcessingDevice::doPrepare(ServiceRegistryRef ref)
{
  ZoneScopedN("DataProcessingDevice::doPrepare");
  auto& context = ref.get<DataProcessorContext>();

  *context.wasActive = false;
  {
    ZoneScopedN("CallbackService::Id::ClockTick");
    ref.get<CallbackService>().call<CallbackService::Id::ClockTick>();
  }
  // Whether or not we had something to do.

  // Initialise the value for context.allDone. It will possibly be updated
  // below if any of the channels is not done.
  //
  // Notice that fake input channels (InputChannelState::Pull) cannot possibly
  // expect to receive an EndOfStream signal. Thus we do not wait for these
  // to be completed. In the case of data source devices, as they do not have
  // real data input channels, they have to signal EndOfStream themselves.
  auto& state = ref.get<DeviceState>();
  auto& spec = ref.get<DeviceSpec const>();
  context.allDone = std::any_of(state.inputChannelInfos.begin(), state.inputChannelInfos.end(), [](const auto& info) {
    if (info.channel) {
      LOGP(debug, "Input channel {}{} has {} parts left and is in state {}.", info.channel->GetName(), (info.id.value == ChannelIndex::INVALID ? " (non DPL)" : ""), info.parts.fParts.size(), (int)info.state);
    } else {
      LOGP(debug, "External channel {} is in state {}.", info.id.value, (int)info.state);
    }
    return (info.parts.fParts.empty() == true && info.state != InputChannelState::Pull);
  });

  // Whether or not all the channels are completed
  LOGP(debug, "Processing {} input channels.", spec.inputChannels.size());
  /// Sort channels by oldest possible timeframe and
  /// process them in such order.
  static std::vector<int> pollOrder;
  pollOrder.resize(state.inputChannelInfos.size());
  std::iota(pollOrder.begin(), pollOrder.end(), 0);
  std::sort(pollOrder.begin(), pollOrder.end(), [&infos = state.inputChannelInfos](int a, int b) {
    return infos[a].oldestForChannel.value < infos[b].oldestForChannel.value;
  });

  // Nothing to poll...
  if (pollOrder.empty()) {
    return;
  }
  auto currentOldest = state.inputChannelInfos[pollOrder.front()].oldestForChannel;
  auto currentNewest = state.inputChannelInfos[pollOrder.back()].oldestForChannel;
  auto delta = currentNewest.value - currentOldest.value;
  LOGP(debug, "oldest possible timeframe range {}, {} => {} delta", currentOldest.value, currentNewest.value,
       delta);
  auto& infos = state.inputChannelInfos;

  if (context.balancingInputs) {
    static int pipelineLength = DefaultsHelpers::pipelineLength();
    static uint64_t ahead = getenv("DPL_MAX_CHANNEL_AHEAD") ? std::atoll(getenv("DPL_MAX_CHANNEL_AHEAD")) : std::max(8, std::min(pipelineLength - 48, pipelineLength / 2));
    auto newEnd = std::remove_if(pollOrder.begin(), pollOrder.end(), [&infos, limitNew = currentOldest.value + ahead](int a) -> bool {
      return infos[a].oldestForChannel.value > limitNew;
    });
    for (auto it = pollOrder.begin(); it < pollOrder.end(); it++) {
      const auto& channelInfo = state.inputChannelInfos[*it];
      if (channelInfo.pollerIndex != -1) {
        auto& poller = state.activeInputPollers[channelInfo.pollerIndex];
        auto& pollerContext = *(PollerContext*)(poller->data);
        if (pollerContext.pollerState == PollerContext::PollerState::Connected || pollerContext.pollerState == PollerContext::PollerState::Suspended) {
          bool running = pollerContext.pollerState == PollerContext::PollerState::Connected;
          bool shouldBeRunning = it < newEnd;
          if (running != shouldBeRunning) {
            uv_poll_start(poller, shouldBeRunning ? UV_READABLE | UV_DISCONNECT | UV_PRIORITIZED : 0, &on_socket_polled);
            pollerContext.pollerState = shouldBeRunning ? PollerContext::PollerState::Connected : PollerContext::PollerState::Suspended;
          }
        }
      }
    }
    pollOrder.erase(newEnd, pollOrder.end());
  }
  LOGP(debug, "processing {} channels", pollOrder.size());

  for (auto sci : pollOrder) {
    auto& info = state.inputChannelInfos[sci];
    auto& channelSpec = spec.inputChannels[sci];
    LOGP(debug, "Processing channel {}", channelSpec.name);

    if (info.state != InputChannelState::Completed && info.state != InputChannelState::Pull) {
      context.allDone = false;
    }
    if (info.state != InputChannelState::Running) {
      // Remember to flush data if we are not running
      // and there is some message pending.
      if (info.parts.Size()) {
        DataProcessingDevice::handleData(ref, info);
      }
      LOGP(debug, "Flushing channel {} which is in state {} and has {} parts still pending.", channelSpec.name, (int)info.state, info.parts.Size());
      continue;
    }
    if (info.channel == nullptr) {
      continue;
    }
    // Only poll DPL channels for now.
    if (info.channelType != ChannelAccountingType::DPL) {
      continue;
    }
    auto& socket = info.channel->GetSocket();
    // If we have pending events from a previous iteration,
    // we do receive in any case.
    // Otherwise we check if there is any pending event and skip
    // this channel in case there is none.
    if (info.hasPendingEvents == 0) {
      socket.Events(&info.hasPendingEvents);
      // If we do not read, we can continue.
      if ((info.hasPendingEvents & 1) == 0 && (info.parts.Size() == 0)) {
        LOGP(debug, "No pending events and no remaining parts to process for channel {}", channelSpec.name);
        continue;
      }
    }
    // We can reset this, because it means we have seen at least 1
    // message after the UV_READABLE was raised.
    info.readPolled = false;
    // Notice that there seems to be a difference between the documentation
    // of zeromq and the observed behavior. The fact that ZMQ_POLLIN
    // is raised does not mean that a message is immediately available to
    // read, just that it will be available soon, so the receive can
    // still return -2. To avoid this we keep receiving on the socket until
    // we get a message. In order not to overflow the DPL queue we process
    // one message at the time and we keep track of wether there were more
    // to process.
    bool newMessages = false;
    while (true) {
      LOGP(debug, "Receiving loop called for channel {} ({}) with oldest possible timeslice {}",
           info.channel->GetName(), info.id.value, info.oldestForChannel.value);
      if (info.parts.Size() < 64) {
        fair::mq::Parts parts;
        info.channel->Receive(parts, 0);
        if (parts.Size()) {
          LOGP(debug, "Receiving some parts {}", parts.Size());
        }
        for (auto&& part : parts) {
          info.parts.fParts.emplace_back(std::move(part));
        }
        newMessages |= true;
      }

      if (info.parts.Size() >= 0) {
        DataProcessingDevice::handleData(ref, info);
        // Receiving data counts as activity now, so that
        // We can make sure we process all the pending
        // messages without hanging on the uv_run.
        break;
      }
    }
    // We check once again for pending events, keeping track if this was the
    // case so that we can immediately repeat this loop and avoid remaining
    // stuck in uv_run. This is because we will not get notified on the socket
    // if more events are pending due to zeromq level triggered approach.
    socket.Events(&info.hasPendingEvents);
    if (info.hasPendingEvents) {
      info.readPolled = false;
      *context.wasActive |= newMessages;
    }
  }
}

void DataProcessingDevice::doRun(ServiceRegistryRef ref)
{
  auto& context = ref.get<DataProcessorContext>();
  auto switchState = [ref](StreamingState newState) {
    auto& state = ref.get<DeviceState>();
    LOG(detail) << "New state " << (int)newState << " old state " << (int)state.streaming;
    state.streaming = newState;
    ref.get<ControlService>().notifyStreamingState(state.streaming);
  };
  auto& state = ref.get<DeviceState>();
  auto& spec = ref.get<DeviceSpec const>();

  if (state.streaming == StreamingState::Idle) {
    *context.wasActive = false;
    return;
  }

  context.completed.clear();
  context.completed.reserve(16);
  *context.wasActive |= DataProcessingDevice::tryDispatchComputation(ref, context.completed);
  DanglingContext danglingContext{*context.registry};

  context.preDanglingCallbacks(danglingContext);
  if (*context.wasActive == false) {
    ref.get<CallbackService>().call<CallbackService::Id::Idle>();
  }
  auto activity = ref.get<DataRelayer>().processDanglingInputs(context.expirationHandlers, *context.registry, true);
  *context.wasActive |= activity.expiredSlots > 0;

  context.completed.clear();
  *context.wasActive |= DataProcessingDevice::tryDispatchComputation(ref, context.completed);

  context.postDanglingCallbacks(danglingContext);

  // If we got notified that all the sources are done, we call the EndOfStream
  // callback and return false. Notice that what happens next is actually
  // dependent on the callback, not something which is controlled by the
  // framework itself.
  if (context.allDone == true && state.streaming == StreamingState::Streaming) {
    switchState(StreamingState::EndOfStreaming);
    *context.wasActive = true;
  }

  if (state.streaming == StreamingState::EndOfStreaming) {
    LOGP(detail, "We are in EndOfStreaming. Flushing queues.");
    // We keep processing data until we are Idle.
    // FIXME: not sure this is the correct way to drain the queues, but
    // I guess we will see.
    /// Besides flushing the queues we must make sure we do not have only
    /// timers as they do not need to be further processed.
    bool hasOnlyGenerated = (spec.inputChannels.size() == 1) && (spec.inputs[0].matcher.lifetime == Lifetime::Timer || spec.inputs[0].matcher.lifetime == Lifetime::Enumeration);
    auto& relayer = ref.get<DataRelayer>();
    while (DataProcessingDevice::tryDispatchComputation(ref, context.completed) && hasOnlyGenerated == false) {
      relayer.processDanglingInputs(context.expirationHandlers, *context.registry, false);
    }
    EndOfStreamContext eosContext{*context.registry, ref.get<DataAllocator>()};

    context.preEOSCallbacks(eosContext);
    auto& streamContext = ref.get<StreamContext>();
    streamContext.preEOSCallbacks(eosContext);
    ref.get<CallbackService>().call<CallbackService::Id::EndOfStream>(eosContext);
    streamContext.postEOSCallbacks(eosContext);
    context.postEOSCallbacks(eosContext);

    for (auto& channel : spec.outputChannels) {
      LOGP(detail, "Sending end of stream to {}", channel.name);
      auto& rawDevice = ref.get<RawDeviceService>();
      DataProcessingHelpers::sendEndOfStream(*rawDevice.device(), channel);
    }
    // This is needed because the transport is deleted before the device.
    relayer.clear();
    switchState(StreamingState::Idle);
    if (hasOnlyGenerated) {
      *context.wasActive = false;
    } else {
      *context.wasActive = true;
    }
    // On end of stream we shut down all output pollers.
    LOGP(detail, "Shutting down output pollers");
    for (auto& poller : state.activeOutputPollers) {
      uv_poll_stop(poller);
    }
    return;
  }

  if (state.streaming == StreamingState::Idle) {
    // On end of stream we shut down all output pollers.
    LOGP(detail, "We are in Idle. Shutting down output pollers.");
    for (auto& poller : state.activeOutputPollers) {
      uv_poll_stop(poller);
    }
  }

  return;
}

void DataProcessingDevice::ResetTask()
{
  ServiceRegistryRef ref{mServiceRegistry};
  ref.get<DataRelayer>().clear();
  auto& deviceContext = ref.get<DeviceContext>();
  // If the signal handler is there, we should
  // hide the registry from it, so that we do not
  // end up calling the signal handler on something
  // which might not be there anymore.
  if (deviceContext.sigusr1Handle) {
    deviceContext.sigusr1Handle->data = nullptr;
  }
  // Makes sure we do not have a working context on
  // shutdown.
  for (auto& handle : ref.get<DeviceState>().activeSignals) {
    handle->data = nullptr;
  }
}

struct WaitBackpressurePolicy {
  void backpressure(InputChannelInfo const&)
  {
  }
};

/// This is the inner loop of our framework. The actual implementation
/// is divided in two parts. In the first one we define a set of lambdas
/// which describe what is actually going to happen, hiding all the state
/// boilerplate which the user does not need to care about at top level.
void DataProcessingDevice::handleData(ServiceRegistryRef ref, InputChannelInfo& info)
{
  auto& context = ref.get<DataProcessorContext>();
  ZoneScopedN("DataProcessingDevice::handleData");

  enum struct InputType : int {
    Invalid = 0,
    Data = 1,
    SourceInfo = 2,
    DomainInfo = 3
  };

  struct InputInfo {
    InputInfo(size_t p, size_t s, InputType t)
      : position(p), size(s), type(t)
    {
    }
    size_t position;
    size_t size;
    InputType type;
  };

  // This is how we validate inputs. I.e. we try to enforce the O2 Data model
  // and we do a few stats. We bind parts as a lambda captured variable, rather
  // than an input, because we do not want the outer loop actually be exposed
  // to the implementation details of the messaging layer.
  auto getInputTypes = [&info, &context]() -> std::optional<std::vector<InputInfo>> {
    auto ref = ServiceRegistryRef{*context.registry};
    auto& stats = ref.get<DataProcessingStats>();
    auto& parts = info.parts;
    stats.updateStats({(int)ProcessingStatsId::TOTAL_INPUTS, DataProcessingStats::Op::Set, (int64_t)parts.Size()});

    TracyPlot("messages received", (int64_t)parts.Size());
    std::vector<InputInfo> results;
    // we can reserve the upper limit
    results.reserve(parts.Size() / 2);
    size_t nTotalPayloads = 0;

    auto insertInputInfo = [&results, &nTotalPayloads](size_t position, size_t length, InputType type) {
      results.emplace_back(position, length, type);
      if (type != InputType::Invalid && length > 1) {
        nTotalPayloads += length - 1;
      }
    };

    for (size_t pi = 0; pi < parts.Size(); pi += 2) {
      auto* headerData = parts.At(pi)->GetData();
      auto sih = o2::header::get<SourceInfoHeader*>(headerData);
      if (sih) {
        LOGP(debug, "Got SourceInfoHeader with state {}", (int)sih->state);
        info.state = sih->state;
        insertInputInfo(pi, 2, InputType::SourceInfo);
        *context.wasActive = true;
        continue;
      }
      auto dih = o2::header::get<DomainInfoHeader*>(headerData);
      if (dih) {
        insertInputInfo(pi, 2, InputType::DomainInfo);
        *context.wasActive = true;
        continue;
      }
      auto dh = o2::header::get<DataHeader*>(headerData);
      if (!dh) {
        insertInputInfo(pi, 0, InputType::Invalid);
        LOGP(error, "Header is not a DataHeader?");
        continue;
      }
      if (dh->payloadSize > parts.At(pi + 1)->GetSize()) {
        insertInputInfo(pi, 0, InputType::Invalid);
        LOGP(error, "DataHeader payloadSize mismatch");
        continue;
      }
      TracyPlot("payload size", (int64_t)dh->payloadSize);
      auto dph = o2::header::get<DataProcessingHeader*>(headerData);
      TracyAlloc(parts.At(pi + 1)->GetData(), parts.At(pi + 1)->GetSize());
      if (!dph) {
        insertInputInfo(pi, 2, InputType::Invalid);
        LOGP(error, "Header stack does not contain DataProcessingHeader");
        continue;
      }
      if (dh->splitPayloadParts > 0 && dh->splitPayloadParts == dh->splitPayloadIndex) {
        // this is indicating a sequence of payloads following the header
        // FIXME: we will probably also set the DataHeader version
        insertInputInfo(pi, dh->splitPayloadParts + 1, InputType::Data);
        pi += dh->splitPayloadParts - 1;
      } else {
        // We can set the type for the next splitPayloadParts
        // because we are guaranteed they are all the same.
        // If splitPayloadParts = 0, we assume that means there is only one (header, payload)
        // pair.
        size_t finalSplitPayloadIndex = pi + (dh->splitPayloadParts > 0 ? dh->splitPayloadParts : 1) * 2;
        if (finalSplitPayloadIndex > parts.Size()) {
          LOGP(error, "DataHeader::splitPayloadParts invalid");
          insertInputInfo(pi, 0, InputType::Invalid);
          continue;
        }
        insertInputInfo(pi, 2, InputType::Data);
        for (; pi + 2 < finalSplitPayloadIndex; pi += 2) {
          insertInputInfo(pi + 2, 2, InputType::Data);
        }
      }
    }
    if (results.size() + nTotalPayloads != parts.Size()) {
      LOG(error) << "inconsistent number of inputs extracted";
      return std::nullopt;
    }
    return results;
  };

  auto reportError = [ref](const char* message) {
    auto& stats = ref.get<DataProcessingStats>();
    stats.updateStats({(int)ProcessingStatsId::ERROR_COUNT, DataProcessingStats::Op::Add, 1});
  };

  auto handleValidMessages = [&info, ref, &reportError](std::vector<InputInfo> const& inputInfos) {
    auto& relayer = ref.get<DataRelayer>();
    static WaitBackpressurePolicy policy;
    auto& parts = info.parts;
    // We relay execution to make sure we have a complete set of parts
    // available.
    bool hasBackpressure = false;
    size_t minBackpressureTimeslice = -1;
    bool hasData = false;
    size_t oldestPossibleTimeslice = -1;
    static std::vector<int> ordering;
    // Same as inputInfos but with iota.
    ordering.resize(inputInfos.size());
    std::iota(ordering.begin(), ordering.end(), 0);
    // stable sort orderings by type and position
    std::stable_sort(ordering.begin(), ordering.end(), [&inputInfos](int const& a, int const& b) {
      auto const& ai = inputInfos[a];
      auto const& bi = inputInfos[b];
      if (ai.type != bi.type) {
        return ai.type < bi.type;
      }
      return ai.position < bi.position;
    });
    for (size_t ii = 0; ii < inputInfos.size(); ++ii) {
      auto const& input = inputInfos[ordering[ii]];
      switch (input.type) {
        case InputType::Data: {
          hasData = true;
          auto headerIndex = input.position;
          auto nMessages = 0;
          auto nPayloadsPerHeader = 0;
          if (input.size > 2) {
            // header and multiple payload sequence
            nMessages = input.size;
            nPayloadsPerHeader = nMessages - 1;
          } else {
            // multiple header-payload pairs
            auto dh = o2::header::get<DataHeader*>(parts.At(headerIndex)->GetData());
            nMessages = dh->splitPayloadParts > 0 ? dh->splitPayloadParts * 2 : 2;
            nPayloadsPerHeader = 1;
            ii += (nMessages / 2) - 1;
          }
          auto onDrop = [ref](TimesliceSlot slot, std::vector<MessageSet>& dropped, TimesliceIndex::OldestOutputInfo oldestOutputInfo) {
            LOGP(debug, "Dropping message from slot {}. Forwarding as needed. Timeslice {}", slot.index, oldestOutputInfo.timeslice.value);
            ref.get<AsyncQueue>();
            ref.get<DecongestionService>();
            ref.get<DataRelayer>();
            // Get the current timeslice for the slot.
            auto& variables = ref.get<TimesliceIndex>().getVariablesForSlot(slot);
            VariableContextHelpers::getTimeslice(variables);
            forwardInputs(ref, slot, dropped, oldestOutputInfo, false, true);
          };
          auto relayed = relayer.relay(parts.At(headerIndex)->GetData(),
                                       &parts.At(headerIndex),
                                       nMessages,
                                       nPayloadsPerHeader,
                                       onDrop);
          switch (relayed.type) {
            case DataRelayer::RelayChoice::Type::Backpressured:
              if (info.normalOpsNotified == true && info.backpressureNotified == false) {
                LOGP(alarm, "Backpressure on channel {}. Waiting.", info.channel->GetName());
                auto& monitoring = ref.get<o2::monitoring::Monitoring>();
                monitoring.send(o2::monitoring::Metric{1, fmt::format("backpressure_{}", info.channel->GetName())});
                info.backpressureNotified = true;
                info.normalOpsNotified = false;
              }
              policy.backpressure(info);
              hasBackpressure = true;
              minBackpressureTimeslice = std::min<size_t>(minBackpressureTimeslice, relayed.timeslice.value);
              break;
            case DataRelayer::RelayChoice::Type::Dropped:
            case DataRelayer::RelayChoice::Type::Invalid:
            case DataRelayer::RelayChoice::Type::WillRelay:
              if (info.normalOpsNotified == false && info.backpressureNotified == true) {
                LOGP(info, "Back to normal on channel {}.", info.channel->GetName());
                auto& monitoring = ref.get<o2::monitoring::Monitoring>();
                monitoring.send(o2::monitoring::Metric{0, fmt::format("backpressure_{}", info.channel->GetName())});
                info.normalOpsNotified = true;
                info.backpressureNotified = false;
              }
              break;
          }
        } break;
        case InputType::SourceInfo: {
          LOGP(detail, "Received SourceInfo");
          auto& context = ref.get<DataProcessorContext>();
          *context.wasActive = true;
          auto headerIndex = input.position;
          auto payloadIndex = input.position + 1;
          assert(payloadIndex < parts.Size());
          // FIXME: the message with the end of stream cannot contain
          //        split parts.
          parts.At(headerIndex).reset(nullptr);
          parts.At(payloadIndex).reset(nullptr);
          // for (size_t i = 0; i < dh->splitPayloadParts > 0 ? dh->splitPayloadParts * 2 - 1 : 1; ++i) {
          //   parts.At(headerIndex + 1 + i).reset(nullptr);
          // }
          // pi += dh->splitPayloadParts > 0 ? dh->splitPayloadParts - 1 : 0;

        } break;
        case InputType::DomainInfo: {
          /// We have back pressure, therefore we do not process DomainInfo anymore.
          /// until the previous message are processed.
          auto& context = ref.get<DataProcessorContext>();
          *context.wasActive = true;
          auto headerIndex = input.position;
          auto payloadIndex = input.position + 1;
          assert(payloadIndex < parts.Size());
          // FIXME: the message with the end of stream cannot contain
          //        split parts.

          auto dih = o2::header::get<DomainInfoHeader*>(parts.At(headerIndex)->GetData());
          if (hasBackpressure && dih->oldestPossibleTimeslice >= minBackpressureTimeslice) {
            break;
          }
          oldestPossibleTimeslice = std::min(oldestPossibleTimeslice, dih->oldestPossibleTimeslice);
          LOGP(debug, "Got DomainInfoHeader, new oldestPossibleTimeslice {} on channel {}", oldestPossibleTimeslice, info.id.value);
          parts.At(headerIndex).reset(nullptr);
          parts.At(payloadIndex).reset(nullptr);
        }
        case InputType::Invalid: {
          reportError("Invalid part found.");
        } break;
      }
    }
    /// The oldest possible timeslice has changed. We can should therefore process it.
    /// Notice we do so only if the incoming data has been fully processed.
    if (oldestPossibleTimeslice != (size_t)-1) {
      info.oldestForChannel = {oldestPossibleTimeslice};
      auto& context = ref.get<DataProcessorContext>();
      context.domainInfoUpdatedCallback(*context.registry, oldestPossibleTimeslice, info.id);
      ref.get<CallbackService>().call<CallbackService::Id::DomainInfoUpdated>((ServiceRegistryRef)*context.registry, (size_t)oldestPossibleTimeslice, (ChannelIndex)info.id);
      *context.wasActive = true;
    }
    auto it = std::remove_if(parts.fParts.begin(), parts.fParts.end(), [](auto& msg) -> bool { return msg.get() == nullptr; });
    parts.fParts.erase(it, parts.end());
    if (parts.fParts.size()) {
      LOG(debug) << parts.fParts.size() << " messages backpressured";
    }
  };

  // Second part. This is the actual outer loop we want to obtain, with
  // implementation details which can be read. Notice how most of the state
  // is actually hidden. For example we do not expose what "input" is. This
  // will allow us to keep the same toplevel logic even if the actual meaning
  // of input is changed (for example we might move away from multipart
  // messages). Notice also that we need to act diffently depending on the
  // actual CompletionOp we want to perform. In particular forwarding inputs
  // also gets rid of them from the cache.
  auto inputTypes = getInputTypes();
  if (bool(inputTypes) == false) {
    reportError("Parts should come in couples. Dropping it.");
    return;
  }
  handleValidMessages(*inputTypes);
  return;
}

namespace
{
struct InputLatency {
  uint64_t minLatency = std::numeric_limits<uint64_t>::max();
  uint64_t maxLatency = std::numeric_limits<uint64_t>::min();
};

auto calculateInputRecordLatency(InputRecord const& record, uint64_t currentTime) -> InputLatency
{
  InputLatency result;

  for (auto& item : record) {
    auto* header = o2::header::get<DataProcessingHeader*>(item.header);
    if (header == nullptr) {
      continue;
    }
    int64_t partLatency = (0x7fffffffffffffff & currentTime) - (0x7fffffffffffffff & header->creation);
    if (partLatency < 0) {
      partLatency = 0;
    }
    result.minLatency = std::min(result.minLatency, (uint64_t)partLatency);
    result.maxLatency = std::max(result.maxLatency, (uint64_t)partLatency);
  }
  return result;
};

auto calculateTotalInputRecordSize(InputRecord const& record) -> int
{
  size_t totalInputSize = 0;
  for (auto& item : record) {
    auto* header = o2::header::get<DataHeader*>(item.header);
    if (header == nullptr) {
      continue;
    }
    totalInputSize += header->payloadSize;
  }
  return totalInputSize;
};

template <typename T>
void update_maximum(std::atomic<T>& maximum_value, T const& value) noexcept
{
  T prev_value = maximum_value;
  while (prev_value < value &&
         !maximum_value.compare_exchange_weak(prev_value, value)) {
  }
}
} // namespace

bool DataProcessingDevice::tryDispatchComputation(ServiceRegistryRef ref, std::vector<DataRelayer::RecordAction>& completed)
{
  auto& context = ref.get<DataProcessorContext>();
  ZoneScopedN("DataProcessingDevice::tryDispatchComputation");
  LOGP(debug, "DataProcessingDevice::tryDispatchComputation");
  // This is the actual hidden state for the outer loop. In case we decide we
  // want to support multithreaded dispatching of operations, I can simply
  // move these to some thread local store and the rest of the lambdas
  // should work just fine.
  std::vector<MessageSet> currentSetOfInputs;

  // For the moment we have a simple "immediately dispatch" policy for stuff
  // in the cache. This could be controlled from the outside e.g. by waiting
  // for a few sets of inputs to arrive before we actually dispatch the
  // computation, however this can be defined at a later stage.
  auto canDispatchSomeComputation = [&completed, ref]() -> bool {
    ref.get<DataRelayer>().getReadyToProcess(completed);
    return completed.empty() == false;
  };

  // We use this to get a list with the actual indexes in the cache which
  // indicate a complete set of inputs. Notice how I fill the completed
  // vector and return it, so that I can have a nice for loop iteration later
  // on.
  auto getReadyActions = [&completed, ref]() -> std::vector<DataRelayer::RecordAction> {
    auto& stats = ref.get<DataProcessingStats>();
    auto& relayer = ref.get<DataRelayer>();
    using namespace o2::framework;
    stats.updateStats({(int)ProcessingStatsId::PENDING_INPUTS, DataProcessingStats::Op::Set, static_cast<int64_t>(relayer.getParallelTimeslices() - completed.size())});
    stats.updateStats({(int)ProcessingStatsId::INCOMPLETE_INPUTS, DataProcessingStats::Op::Set, completed.empty() ? 1 : 0});
    return completed;
  };

  //
  auto getInputSpan = [ref, &currentSetOfInputs](TimesliceSlot slot, bool consume = true) {
    auto& relayer = ref.get<DataRelayer>();
    if (consume) {
      currentSetOfInputs = relayer.consumeAllInputsForTimeslice(slot);
    } else {
      currentSetOfInputs = relayer.consumeExistingInputsForTimeslice(slot);
    }
    auto getter = [&currentSetOfInputs](size_t i, size_t partindex) -> DataRef {
      if (currentSetOfInputs[i].getNumberOfPairs() > partindex) {
        const char* headerptr = nullptr;
        const char* payloadptr = nullptr;
        size_t payloadSize = 0;
        // - each input can have multiple parts
        // - "part" denotes a sequence of messages belonging together, the first message of the
        //   sequence is the header message
        // - each part has one or more payload messages
        // - InputRecord provides all payloads as header-payload pair
        auto const& headerMsg = currentSetOfInputs[i].associatedHeader(partindex);
        auto const& payloadMsg = currentSetOfInputs[i].associatedPayload(partindex);
        headerptr = static_cast<char const*>(headerMsg->GetData());
        payloadptr = payloadMsg ? static_cast<char const*>(payloadMsg->GetData()) : nullptr;
        payloadSize = payloadMsg ? payloadMsg->GetSize() : 0;
        return DataRef{nullptr, headerptr, payloadptr, payloadSize};
      }
      return DataRef{};
    };
    auto nofPartsGetter = [&currentSetOfInputs](size_t i) -> size_t {
      return currentSetOfInputs[i].getNumberOfPairs();
    };
    return InputSpan{getter, nofPartsGetter, currentSetOfInputs.size()};
  };

  auto markInputsAsDone = [ref](TimesliceSlot slot) -> void {
    auto& relayer = ref.get<DataRelayer>();
    relayer.updateCacheStatus(slot, CacheEntryStatus::RUNNING, CacheEntryStatus::DONE);
  };

  // I need a preparation step which gets the current timeslice id and
  // propagates it to the various contextes (i.e. the actual entities which
  // create messages) because the messages need to have the timeslice id into
  // it.
  auto prepareAllocatorForCurrentTimeSlice = [ref](TimesliceSlot i) -> void {
    auto& dataProcessorContext = ref.get<DataProcessorContext>();
    auto& relayer = ref.get<DataRelayer>();
    auto& timingInfo = ref.get<TimingInfo>();
    ZoneScopedN("DataProcessingDevice::prepareForCurrentTimeslice");
    auto timeslice = relayer.getTimesliceForSlot(i);

    timingInfo.timeslice = timeslice.value;
    timingInfo.tfCounter = relayer.getFirstTFCounterForSlot(i);
    timingInfo.firstTForbit = relayer.getFirstTFOrbitForSlot(i);
    timingInfo.runNumber = relayer.getRunNumberForSlot(i);
    timingInfo.creation = relayer.getCreationTimeForSlot(i);
    timingInfo.globalRunNumberChanged = !TimingInfo::timesliceIsTimer(timeslice.value) && dataProcessorContext.lastRunNumberProcessed != timingInfo.runNumber;
    // A switch to runNumber=0 should not appear and thus does not set globalRunNumberChanged, unless it is seen in the first processed timeslice
    timingInfo.globalRunNumberChanged &= (dataProcessorContext.lastRunNumberProcessed == -1 || timingInfo.runNumber != 0);
    // We report wether or not this timing info refers to a new Run.
    if (timingInfo.globalRunNumberChanged) {
      dataProcessorContext.lastRunNumberProcessed = timingInfo.runNumber;
    }
    // FIXME: for now there is only one stream, however we
    //        should calculate this correctly once we finally get the
    //        the StreamContext in.
    timingInfo.streamRunNumberChanged = timingInfo.globalRunNumberChanged;
  };

  // When processing them, timers will have to be cleaned up
  // to avoid double counting them.
  // This was actually the easiest solution we could find for
  // O2-646.
  auto cleanTimers = [&currentSetOfInputs](TimesliceSlot slot, InputRecord& record) {
    assert(record.size() == currentSetOfInputs.size());
    for (size_t ii = 0, ie = record.size(); ii < ie; ++ii) {
      // assuming that for timer inputs we do have exactly one PartRef object
      // in the MessageSet, multiple PartRef Objects are only possible for either
      // split payload messages of wildcard matchers, both for data inputs
      DataRef input = record.getByPos(ii);
      if (input.spec->lifetime != Lifetime::Timer) {
        continue;
      }
      if (input.header == nullptr) {
        continue;
      }
      // This will hopefully delete the message.
      currentSetOfInputs[ii].clear();
    }
  };

  // Function to cleanup record. For the moment we
  // simply use it to keep track of input messages
  // which are not needed, to display them in the GUI.
#ifdef TRACY_ENABLE
  auto cleanupRecord = [](InputRecord& record) {
    for (size_t ii = 0, ie = record.size(); ii < ie; ++ii) {
      DataRef input = record.getByPos(ii);
      if (input.header == nullptr) {
        continue;
      }
      auto sih = o2::header::get<SourceInfoHeader*>(input.header);
      if (sih) {
        continue;
      }

      auto dh = o2::header::get<DataHeader*>(input.header);
      if (!dh) {
        continue;
      }
      TracyFree(input.payload);
    }
  };
#endif

  auto switchState = [ref](StreamingState newState) {
    auto& control = ref.get<ControlService>();
    auto& state = ref.get<DeviceState>();
    state.streaming = newState;
    control.notifyStreamingState(state.streaming);
  };

  if (canDispatchSomeComputation() == false) {
    LOGP(debug, "No computations available for dispatching.");
    return false;
  }

  auto postUpdateStats = [ref](DataRelayer::RecordAction const& action, InputRecord const& record, uint64_t tStart, uint64_t tStartMilli) {
    auto& stats = ref.get<DataProcessingStats>();
    auto& states = ref.get<DataProcessingStates>();
    std::atomic_thread_fence(std::memory_order_release);
    char relayerSlotState[1024];
    int written = snprintf(relayerSlotState, 1024, "%d ", DefaultsHelpers::pipelineLength());
    char* buffer = relayerSlotState + written;
    for (size_t ai = 0; ai != record.size(); ai++) {
      buffer[ai] = record.isValid(ai) ? '3' : '0';
    }
    buffer[record.size()] = 0;
    states.updateState({.id = short((int)ProcessingStateId::DATA_RELAYER_BASE + action.slot.index),
                        .size = (int)(record.size() + buffer - relayerSlotState),
                        .data = relayerSlotState});
    uint64_t tEnd = uv_hrtime();
    // tEnd and tStart are in nanoseconds according to https://docs.libuv.org/en/v1.x/misc.html#c.uv_hrtime
    int64_t wallTimeMs = (tEnd - tStart) / 1000000;
    stats.updateStats({(int)ProcessingStatsId::LAST_ELAPSED_TIME_MS, DataProcessingStats::Op::Set, wallTimeMs});
    // Sum up the total wall time, in milliseconds.
    stats.updateStats({(int)ProcessingStatsId::TOTAL_WALL_TIME_MS, DataProcessingStats::Op::Add, wallTimeMs});
    // The time interval is in seconds while tEnd - tStart is in nanoseconds, so we divide by 1000000 to get the fraction in ms/s.
    stats.updateStats({(short)ProcessingStatsId::CPU_USAGE_FRACTION, DataProcessingStats::Op::CumulativeRate, wallTimeMs});
    stats.updateStats({(int)ProcessingStatsId::LAST_PROCESSED_SIZE, DataProcessingStats::Op::Set, calculateTotalInputRecordSize(record)});
    stats.updateStats({(int)ProcessingStatsId::TOTAL_PROCESSED_SIZE, DataProcessingStats::Op::Add, calculateTotalInputRecordSize(record)});
    auto latency = calculateInputRecordLatency(record, tStartMilli);
    stats.updateStats({(int)ProcessingStatsId::LAST_MIN_LATENCY, DataProcessingStats::Op::Set, (int)latency.minLatency});
    stats.updateStats({(int)ProcessingStatsId::LAST_MAX_LATENCY, DataProcessingStats::Op::Set, (int)latency.maxLatency});
    static int count = 0;
    stats.updateStats({(int)ProcessingStatsId::PROCESSING_RATE_HZ, DataProcessingStats::Op::CumulativeRate, 1});
    count++;
  };

  auto preUpdateStats = [ref](DataRelayer::RecordAction const& action, InputRecord const& record, uint64_t) {
    auto& states = ref.get<DataProcessingStates>();
    std::atomic_thread_fence(std::memory_order_release);
    char relayerSlotState[1024];
    snprintf(relayerSlotState, 1024, "%d ", DefaultsHelpers::pipelineLength());
    char* buffer = strchr(relayerSlotState, ' ') + 1;
    for (size_t ai = 0; ai != record.size(); ai++) {
      buffer[ai] = record.isValid(ai) ? '2' : '0';
    }
    buffer[record.size()] = 0;
    states.updateState({.id = short((int)ProcessingStateId::DATA_RELAYER_BASE + action.slot.index), .size = (int)(record.size() + buffer - relayerSlotState), .data = relayerSlotState});
  };

  // This is the main dispatching loop
  LOGP(debug, "Processing actions:");
  auto& state = ref.get<DeviceState>();
  auto& spec = ref.get<DeviceSpec const>();

  auto& dpContext = ref.get<DataProcessorContext>();
  auto& streamContext = ref.get<StreamContext>();
  for (auto action : getReadyActions()) {
    LOGP(debug, "  Begin action");
    if (action.op == CompletionPolicy::CompletionOp::Wait) {
      LOGP(debug, "  - Action is to Wait");
      continue;
    }

    switch (action.op) {
      case CompletionPolicy::CompletionOp::Consume:
        LOG(debug) << "  - Action is to " << action.op << " " << action.slot.index;
        break;
      default:
        LOG(debug) << "  - Action is to " << action.op << " " << action.slot.index;
        break;
    }

    prepareAllocatorForCurrentTimeSlice(TimesliceSlot{action.slot});
    bool shouldConsume = action.op == CompletionPolicy::CompletionOp::Consume ||
                         action.op == CompletionPolicy::CompletionOp::Discard;
    InputSpan span = getInputSpan(action.slot, shouldConsume);
    auto& spec = ref.get<DeviceSpec const>();
    InputRecord record{spec.inputs,
                       span,
                       *context.registry};
    ProcessingContext processContext{record, ref, ref.get<DataAllocator>()};
    {
      ZoneScopedN("service pre processing");
      // Notice this should be thread safe and reentrant
      // as it is called from many threads.
      streamContext.preProcessingCallbacks(processContext);
      dpContext.preProcessingCallbacks(processContext);
    }
    if (action.op == CompletionPolicy::CompletionOp::Discard) {
      LOGP(debug, "  - Action is to Discard");
      context.postDispatchingCallbacks(processContext);
      if (spec.forwards.empty() == false) {
        auto& timesliceIndex = ref.get<TimesliceIndex>();
        forwardInputs(ref, action.slot, currentSetOfInputs, timesliceIndex.getOldestPossibleOutput(), false);
        continue;
      }
    }
    // If there is no optional inputs we canForwardEarly
    // the messages to that parallel processing can happen.
    // In this case we pass true to indicate that we want to
    // copy the messages to the subsequent data processor.
    bool hasForwards = spec.forwards.empty() == false;
    bool consumeSomething = action.op == CompletionPolicy::CompletionOp::Consume || action.op == CompletionPolicy::CompletionOp::ConsumeExisting;

    if (context.canForwardEarly && hasForwards && consumeSomething) {
      LOGP(debug, "  - Early forwarding");
      auto& timesliceIndex = ref.get<TimesliceIndex>();
      forwardInputs(ref, action.slot, currentSetOfInputs, timesliceIndex.getOldestPossibleOutput(), true, action.op == CompletionPolicy::CompletionOp::Consume);
    }
    markInputsAsDone(action.slot);

    uint64_t tStart = uv_hrtime();
    uint64_t tStartMilli = TimingHelpers::getRealtimeSinceEpochStandalone();
    preUpdateStats(action, record, tStart);

    static bool noCatch = getenv("O2_NO_CATCHALL_EXCEPTIONS") && strcmp(getenv("O2_NO_CATCHALL_EXCEPTIONS"), "0");

    auto runNoCatch = [&context, ref, &processContext](DataRelayer::RecordAction& action) mutable {
      auto& state = ref.get<DeviceState>();
      auto& spec = ref.get<DeviceSpec const>();
      auto& streamContext = ref.get<StreamContext>();
      auto& dpContext = ref.get<DataProcessorContext>();
      if (state.quitRequested == false) {
        {
          ZoneScopedN("service post processing");
          // Callbacks from services
          dpContext.preProcessingCallbacks(processContext);
          streamContext.preProcessingCallbacks(processContext);
          dpContext.preProcessingCallbacks(processContext);
          // Callbacks from users
          ref.get<CallbackService>().call<CallbackService::Id::PreProcessing>(o2::framework::ServiceRegistryRef{ref}, (int)action.op);
        }
        if (context.statefulProcess) {
          ZoneScopedN("statefull process");
          (context.statefulProcess)(processContext);
        } else if (context.statelessProcess) {
          ZoneScopedN("stateless process");
          (context.statelessProcess)(processContext);
        } else {
          state.streaming = StreamingState::Idle;
        }

        // Notify the sink we just consumed some timeframe data
        if (context.isSink && action.op == CompletionPolicy::CompletionOp::Consume) {
          auto& allocator = ref.get<DataAllocator>();
          allocator.make<int>(OutputRef{"dpl-summary", compile_time_hash(spec.name.c_str())}, 1);
        }

        // Extra callback which allows a service to add extra outputs.
        // This is needed e.g. to ensure that injected CCDB outputs are added
        // before an end of stream.
        {
          ref.get<CallbackService>().call<CallbackService::Id::FinaliseOutputs>(o2::framework::ServiceRegistryRef{ref}, (int)action.op);
          dpContext.finaliseOutputsCallbacks(processContext);
          streamContext.finaliseOutputsCallbacks(processContext);
        }

        {
          ZoneScopedN("service post processing");
          ref.get<CallbackService>().call<CallbackService::Id::PostProcessing>(o2::framework::ServiceRegistryRef{ref}, (int)action.op);
          dpContext.postProcessingCallbacks(processContext);
          streamContext.postProcessingCallbacks(processContext);
        }
      }
    };

    if ((state.tracingFlags & DeviceState::LoopReason::TRACE_USERCODE) != 0) {
      state.severityStack.push_back((int)fair::Logger::GetConsoleSeverity());
      fair::Logger::SetConsoleSeverity(fair::Severity::trace);
    }
    if (noCatch) {
      try {
        runNoCatch(action);
      } catch (o2::framework::RuntimeErrorRef e) {
        ZoneScopedN("error handling");
        (context.errorHandling)(e, record);
      }
    } else {
      try {
        runNoCatch(action);
      } catch (std::exception& ex) {
        ZoneScopedN("error handling");
        /// Convert a standard exception to a RuntimeErrorRef
        /// Notice how this will lose the backtrace information
        /// and report the exception coming from here.
        auto e = runtime_error(ex.what());
        (context.errorHandling)(e, record);
      } catch (o2::framework::RuntimeErrorRef e) {
        ZoneScopedN("error handling");
        (context.errorHandling)(e, record);
      }
    }
    if (state.severityStack.empty() == false) {
      fair::Logger::SetConsoleSeverity((fair::Severity)state.severityStack.back());
      state.severityStack.pop_back();
    }

    postUpdateStats(action, record, tStart, tStartMilli);
    // We forward inputs only when we consume them. If we simply Process them,
    // we keep them for next message arriving.
    if (action.op == CompletionPolicy::CompletionOp::Consume) {
      context.postDispatchingCallbacks(processContext);
      ref.get<CallbackService>().call<CallbackService::Id::DataConsumed>(o2::framework::ServiceRegistryRef{ref});
    }
    if ((context.canForwardEarly == false) && hasForwards && consumeSomething) {
      LOGP(debug, "Late forwarding");
      auto& timesliceIndex = ref.get<TimesliceIndex>();
      forwardInputs(ref, action.slot, currentSetOfInputs, timesliceIndex.getOldestPossibleOutput(), false, action.op == CompletionPolicy::CompletionOp::Consume);
    }
    context.postForwardingCallbacks(processContext);
    if (action.op == CompletionPolicy::CompletionOp::Consume) {
#ifdef TRACY_ENABLE
      cleanupRecord(record);
#endif
    } else if (action.op == CompletionPolicy::CompletionOp::Process) {
      cleanTimers(action.slot, record);
    }
  }

  // We now broadcast the end of stream if it was requested
  if (state.streaming == StreamingState::EndOfStreaming) {
    LOGP(detail, "Broadcasting end of stream");
    for (auto& channel : spec.outputChannels) {
      auto& rawDevice = ref.get<RawDeviceService>();
      DataProcessingHelpers::sendEndOfStream(*rawDevice.device(), channel);
    }
    switchState(StreamingState::Idle);
  }

  return true;
}

void DataProcessingDevice::error(const char* msg)
{
  LOG(error) << msg;
  ServiceRegistryRef ref{mServiceRegistry};
  auto& stats = ref.get<DataProcessingStats>();
  stats.updateStats({(int)ProcessingStatsId::ERROR_COUNT, DataProcessingStats::Op::Add});
}

std::unique_ptr<ConfigParamStore> DeviceConfigurationHelpers::getConfiguration(ServiceRegistryRef registry, const char* name, std::vector<ConfigParamSpec> const& options)
{

  if (registry.active<ConfigurationInterface>()) {
    auto& cfg = registry.get<ConfigurationInterface>();
    try {
      cfg.getRecursive(name);
      std::vector<std::unique_ptr<ParamRetriever>> retrievers;
      retrievers.emplace_back(std::make_unique<ConfigurationOptionsRetriever>(&cfg, name));
      auto configStore = std::make_unique<ConfigParamStore>(options, std::move(retrievers));
      configStore->preload();
      configStore->activate();
      return configStore;
    } catch (...) {
      // No overrides...
    }
  }
  return {nullptr};
}

} // namespace o2::framework
