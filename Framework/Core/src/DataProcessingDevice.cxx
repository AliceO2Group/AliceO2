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
#include "Framework/DataProcessingDevice.h"
#include "Framework/ChannelMatching.h"
#include "Framework/ControlService.h"
#include "Framework/ComputingQuotaEvaluator.h"
#include "Framework/DataProcessingHeader.h"
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
#include "Framework/SourceInfoHeader.h"
#include "Framework/Logger.h"
#include "Framework/DriverClient.h"
#include "Framework/Monitoring.h"
#include "PropertyTreeHelpers.h"
#include "DataProcessingStatus.h"
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

DataProcessingDevice::DataProcessingDevice(RunningDeviceRef ref, ServiceRegistry& registry, ProcessingPolicies& policies)
  : mSpec{registry.get<RunningWorkflowInfo const>().devices[ref.index]},
    mState{registry.get<DeviceState>()},
    mInit{mSpec.algorithm.onInit},
    mStatefulProcess{nullptr},
    mStatelessProcess{mSpec.algorithm.onProcess},
    mError{mSpec.algorithm.onError},
    mConfigRegistry{nullptr},
    mServiceRegistry{registry},
    mAllocator{&registry, mSpec.outputs},
    mProcessingPolicies{policies},
    mQuotaEvaluator{registry.get<ComputingQuotaEvaluator>()}
{
  /// FIXME: move erro handling to a service?
  if (mError != nullptr) {
    mErrorHandling = [&errorCallback = mError,
                      &serviceRegistry = mServiceRegistry](RuntimeErrorRef e, InputRecord& record) {
      ZoneScopedN("Error handling");
      auto& err = error_from_ref(e);
      LOGP(error, "Exception caught: {} ", err.what);
      demangled_backtrace_symbols(err.backtrace, err.maxBacktrace, STDERR_FILENO);
      serviceRegistry.get<DataProcessingStats>().exceptionCount++;
      ErrorContext errorContext{record, serviceRegistry, e};
      errorCallback(errorContext);
    };
  } else {
    mErrorHandling = [&errorPolicy = mProcessingPolicies.error,
                      &serviceRegistry = mServiceRegistry](RuntimeErrorRef e, InputRecord& record) {
      ZoneScopedN("Error handling");
      auto& err = error_from_ref(e);
      LOGP(error, "Exception caught: {} ", err.what);
      demangled_backtrace_symbols(err.backtrace, err.maxBacktrace, STDERR_FILENO);
      serviceRegistry.get<DataProcessingStats>().exceptionCount++;
      switch (errorPolicy) {
        case TerminationPolicy::QUIT:
          throw e;
        default:
          break;
      }
    };
  }

  std::function<void(const fair::mq::State)> stateWatcher = [this, &registry = mServiceRegistry](const fair::mq::State state) -> void {
    auto& deviceState = registry.get<DeviceState>();
    auto& control = registry.get<ControlService>();
    control.notifyDeviceState(fair::mq::GetStateName(state));
    if (deviceState.nextFairMQState.empty() == false) {
      auto state = deviceState.nextFairMQState.back();
      this->ChangeState(state);
      deviceState.nextFairMQState.pop_back();
    }
  };

  this->SubscribeToStateChange("dpl", stateWatcher);

  // One task for now.
  mStreams.resize(1);
  mHandles.resize(1);

  mDeviceContext.device = this;
  mDeviceContext.spec = &mSpec;
  mDeviceContext.state = &mState;
  mDeviceContext.quotaEvaluator = &mQuotaEvaluator;
  mDeviceContext.stats = &mStats;

  mAwakeHandle = (uv_async_t*)malloc(sizeof(uv_async_t));
  assert(mState.loop);
  int res = uv_async_init(mState.loop, mAwakeHandle, on_communication_requested);
  mAwakeHandle->data = &mState;
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
  TaskStreamInfo* task = (TaskStreamInfo*)handle->data;
  DataProcessorContext& context = *task->context;
  DataProcessingDevice::doPrepare(context);
  DataProcessingDevice::doRun(context);
  //  FrameMark;
}

// Once the processing in a thread is done, this is executed on the main thread.
void run_completion(uv_work_t* handle, int status)
{
  TaskStreamInfo* task = (TaskStreamInfo*)handle->data;
  DataProcessorContext& context = *task->context;

  using o2::monitoring::Metric;
  using o2::monitoring::Monitoring;
  using o2::monitoring::tags::Key;
  using o2::monitoring::tags::Value;

  static std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats&)> reportConsumedOffer = [&monitoring = context.registry->get<Monitoring>()](ComputingQuotaOffer const& accumulatedConsumed, ComputingQuotaStats& stats) {
    stats.totalConsumedBytes += accumulatedConsumed.sharedMemory;
    monitoring.send(Metric{(uint64_t)stats.totalConsumedBytes, "shm-offer-bytes-consumed"}.addTag(Key::Subsystem, Value::DPL));
    monitoring.flushBuffer();
  };

  static std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats const&)> reportExpiredOffer = [&monitoring = context.registry->get<Monitoring>()](ComputingQuotaOffer const& offer, ComputingQuotaStats const& stats) {
    monitoring.send(Metric{(uint64_t)stats.totalExpiredOffers, "resource-offer-expired"}.addTag(Key::Subsystem, Value::DPL));
    monitoring.send(Metric{(uint64_t)stats.totalExpiredBytes, "arrow-bytes-expired"}.addTag(Key::Subsystem, Value::DPL));
    monitoring.flushBuffer();
  };

  for (auto& consumer : context.deviceContext->state->offerConsumers) {
    context.deviceContext->quotaEvaluator->consume(task->id.index, consumer, reportConsumedOffer);
  }
  context.deviceContext->state->offerConsumers.clear();
  context.deviceContext->quotaEvaluator->handleExpired(reportExpiredOffer);
  context.deviceContext->quotaEvaluator->dispose(task->id.index);
  task->running = false;
  ZoneScopedN("run_completion");
}

// Context for polling
struct PollerContext {
  char const* name = nullptr;
  uv_loop_t* loop = nullptr;
  DataProcessingDevice* device = nullptr;
  DeviceState* state = nullptr;
  fair::mq::Socket* socket = nullptr;
  InputChannelInfo* channelInfo = nullptr;
  int fd = -1;
  bool read = true;
};

void on_socket_polled(uv_poll_t* poller, int status, int events)
{
  PollerContext* context = (PollerContext*)poller->data;
  context->state->loopReason |= DeviceState::DATA_SOCKET_POLLED;
  switch (events) {
    case UV_READABLE: {
      ZoneScopedN("socket readable event");
      LOG(debug) << "socket polled UV_READABLE: " << context->name;
      context->state->loopReason |= DeviceState::DATA_INCOMING;
    } break;
    case UV_WRITABLE: {
      ZoneScopedN("socket writeable");
      LOG(debug) << "socket polled UV_WRITEABLE";
      context->state->loopReason |= DeviceState::DATA_OUTGOING;
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
  TracyAppInfo(mSpec.name.data(), mSpec.name.size());
  ZoneScopedN("DataProcessingDevice::Init");
  mRelayer = &mServiceRegistry.get<DataRelayer>();

  auto configStore = DeviceConfigurationHelpers::getConfiguration(mServiceRegistry, mSpec.name.c_str(), mSpec.options);
  if (configStore == nullptr) {
    std::vector<std::unique_ptr<ParamRetriever>> retrievers;
    retrievers.emplace_back(std::make_unique<FairOptionsRetriever>(GetConfig()));
    configStore = std::make_unique<ConfigParamStore>(mSpec.options, std::move(retrievers));
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
    std::string configString = fmt::format("[CONFIG];{}={};1;{}", entry.first, str, configStore->provenance(entry.first.c_str())).c_str();
    mServiceRegistry.get<DriverClient>().tell(configString.c_str());
  }

  mConfigRegistry = std::make_unique<ConfigParamRegistry>(std::move(configStore));

  mExpirationHandlers.clear();

  if (mInit) {
    InitContext initContext{*mConfigRegistry, mServiceRegistry};
    mStatefulProcess = mInit(initContext);
  }
  mState.inputChannelInfos.resize(mSpec.inputChannels.size());
  /// Internal channels which will never create an actual message
  /// should be considered as in "Pull" mode, since we do not
  /// expect them to create any data.
  int validChannelId = 0;
  for (size_t ci = 0; ci < mSpec.inputChannels.size(); ++ci) {
    auto& name = mSpec.inputChannels[ci].name;
    if (name.find(mSpec.channelPrefix + "from_internal-dpl-clock") == 0) {
      mState.inputChannelInfos[ci].state = InputChannelState::Pull;
      mState.inputChannelInfos[ci].id = {ChannelIndex::INVALID};
      validChannelId++;
    } else {
      mState.inputChannelInfos[ci].id = {validChannelId++};
    }
  }

  // Invoke the callback policy for this device.
  if (mSpec.callbacksPolicy.policy != nullptr) {
    InitContext initContext{*mConfigRegistry, mServiceRegistry};
    mSpec.callbacksPolicy.policy(mServiceRegistry.get<CallbackService>(), initContext);
  }
}

void on_signal_callback(uv_signal_t* handle, int signum)
{
  ZoneScopedN("Signal callaback");
  LOG(debug) << "Signal " << signum << " received.";
  auto* context = (DeviceContext*)handle->data;
  context->state->loopReason |= DeviceState::SIGNAL_ARRIVED;
  size_t ri = 0;
  while (ri != context->quotaEvaluator->mOffers.size()) {
    auto& offer = context->quotaEvaluator->mOffers[ri];
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
  for (auto& offer : context->quotaEvaluator->mOffers) {
    if (offer.valid == false) {
      offer.cpu = 0;
      offer.memory = 0;
      offer.sharedMemory = 1000000000;
      offer.valid = true;
      offer.user = -1;
      break;
    }
  }
  context->stats->totalSigusr1 += 1;
}

/// Invoke the callbacks for the mPendingRegionInfos
void handleRegionCallbacks(ServiceRegistry& registry, std::vector<fair::mq::RegionInfo>& infos)
{
  if (infos.empty() == false) {
    std::vector<fair::mq::RegionInfo> toBeNotified;
    toBeNotified.swap(infos); // avoid any MT issue.
    for (auto const& info : toBeNotified) {
      registry.get<CallbackService>()(CallbackService::Id::RegionInfoCallback, info);
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
  // We add a timer only in case a channel poller is not there.
  if ((mStatefulProcess != nullptr) || (mStatelessProcess != nullptr)) {
    for (auto& [channelName, channel] : fChannels) {
      InputChannelInfo* channelInfo;
      for (size_t ci = 0; ci < mDeviceContext.spec->inputChannels.size(); ++ci) {
        auto& channelSpec = mDeviceContext.spec->inputChannels[ci];
        channelInfo = &mDeviceContext.state->inputChannelInfos[ci];
        if (channelSpec.name != channelName) {
          continue;
        }
        channelInfo->channel = &this->GetChannel(channelName, 0);
      }
      if ((channelName.rfind("from_internal-dpl", 0) == 0) &&
          (channelName.rfind("from_internal-dpl-aod", 0) != 0) &&
          (channelName.rfind("from_internal-dpl-ccdb-backend", 0) != 0) &&
          (channelName.rfind("from_internal-dpl-injected", 0)) != 0) {
        LOGP(detail, "{} is an internal channel. Skipping as no input will come from there.", channelName);
        continue;
      }
      // We only watch receiving sockets.
      if (channelName.rfind("from_" + mSpec.name + "_", 0) == 0) {
        LOGP(detail, "{} is to send data. Not polling.", channelName);
        continue;
      }

      if (channelName.rfind("from_") != 0) {
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
      pCtx->loop = mState.loop;
      pCtx->device = this;
      pCtx->state = &mState;
      pCtx->fd = zmq_fd;
      assert(channelInfo != nullptr);
      pCtx->channelInfo = channelInfo;
      pCtx->socket = &channel[0].GetSocket();
      pCtx->read = true;
      poller->data = pCtx;
      uv_poll_init(mState.loop, poller, zmq_fd);
      if (channelName.rfind("from_") != 0) {
        LOGP(detail, "{} is an out of band channel.", channelName);
        mState.activeOutOfBandPollers.push_back(poller);
      } else {
        mState.activeInputPollers.push_back(poller);
      }
    }
    // In case we do not have any input channel and we do not have
    // any timers or signal watchers we still wake up whenever we can send data to downstream
    // devices to allow for enumerations.
    if (mState.activeInputPollers.empty() &&
        mState.activeOutOfBandPollers.empty() &&
        mState.activeTimers.empty() &&
        mState.activeSignals.empty()) {
      mDeviceContext.exitTransitionTimeout = 0;
      for (auto& [channelName, channel] : fChannels) {
        if (channelName.rfind(mSpec.channelPrefix + "from_internal-dpl", 0) == 0) {
          LOGP(detail, "{} is an internal channel. Not polling.", channelName);
          continue;
        }
        if (channelName.rfind(mSpec.channelPrefix + "from_" + mSpec.name + "_", 0) == 0) {
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
        pCtx->loop = mState.loop;
        pCtx->device = this;
        pCtx->state = &mState;
        pCtx->fd = zmq_fd;
        pCtx->read = false;
        poller->data = pCtx;
        uv_poll_init(mState.loop, poller, zmq_fd);
        mState.activeOutputPollers.push_back(poller);
      }
    }
  } else {
    mDeviceContext.exitTransitionTimeout = 0;
    // This is a fake device, so we can request to exit immediately
    mServiceRegistry.get<ControlService>().readyToQuit(QuitRequest::Me);
    // A two second timer to stop internal devices which do not want to
    auto* timer = (uv_timer_t*)malloc(sizeof(uv_timer_t));
    uv_timer_init(mState.loop, timer);
    timer->data = &mState;
    uv_update_time(mState.loop);
    uv_timer_start(timer, on_idle_timer, 2000, 2000);
    mState.activeTimers.push_back(timer);
  }
}

void DataProcessingDevice::startPollers()
{
  for (auto& poller : mState.activeInputPollers) {
    uv_poll_start(poller, UV_READABLE | UV_DISCONNECT, &on_socket_polled);
  }
  for (auto& poller : mState.activeOutOfBandPollers) {
    uv_poll_start(poller, UV_WRITABLE, &on_out_of_band_polled);
  }
  for (auto& poller : mState.activeOutputPollers) {
    uv_poll_start(poller, UV_WRITABLE, &on_socket_polled);
  }

  mDeviceContext.gracePeriodTimer = (uv_timer_t*)malloc(sizeof(uv_timer_t));
  mDeviceContext.gracePeriodTimer->data = &mState;
  uv_timer_init(mState.loop, mDeviceContext.gracePeriodTimer);
}

void DataProcessingDevice::stopPollers()
{
  LOGP(detail, "Stopping {} input pollers", mState.activeInputPollers.size());
  for (auto& poller : mState.activeInputPollers) {
    uv_poll_stop(poller);
  }
  LOGP(detail, "Stopping {} out of band pollers", mState.activeOutOfBandPollers.size());
  for (auto& poller : mState.activeOutOfBandPollers) {
    uv_poll_stop(poller);
  }
  LOGP(detail, "Stopping {} output pollers", mState.activeOutOfBandPollers.size());
  for (auto& poller : mState.activeOutputPollers) {
    uv_poll_stop(poller);
  }

  uv_timer_stop(mDeviceContext.gracePeriodTimer);
  free(mDeviceContext.gracePeriodTimer);
  mDeviceContext.gracePeriodTimer = nullptr;
}

void DataProcessingDevice::InitTask()
{
  auto distinct = DataRelayerHelpers::createDistinctRouteIndex(mSpec.inputs);
  int i = 0;
  for (auto& di : distinct) {
    auto& route = mSpec.inputs[di];
    if (route.configurator.has_value() == false) {
      i++;
      continue;
    }
    ExpirationHandler handler{
      .name = route.configurator->name,
      .routeIndex = RouteIndex{i++},
      .lifetime = route.matcher.lifetime,
      .creator = route.configurator->creatorConfigurator(mState, mServiceRegistry, *mConfigRegistry),
      .checker = route.configurator->danglingConfigurator(mState, *mConfigRegistry),
      .handler = route.configurator->expirationConfigurator(mState, *mConfigRegistry)};
    mExpirationHandlers.emplace_back(std::move(handler));
  }

  if (mState.awakeMainThread == nullptr) {
    mState.awakeMainThread = (uv_async_t*)malloc(sizeof(uv_async_t));
    mState.awakeMainThread->data = &mState;
    uv_async_init(mState.loop, mState.awakeMainThread, on_awake_main_thread);
  }

  mDeviceContext.expectedRegionCallbacks = std::stoi(fConfig->GetValue<std::string>("expected-region-callbacks"));
  mDeviceContext.exitTransitionTimeout = std::stoi(fConfig->GetValue<std::string>("exit-transition-timeout"));

  for (auto& channel : fChannels) {
    channel.second.at(0).Transport()->SubscribeToRegionEvents([&context = mDeviceContext,
                                                               &registry = mServiceRegistry,
                                                               &pendingRegionInfos = mPendingRegionInfos,
                                                               &regionInfoMutex = mRegionInfoMutex](fair::mq::RegionInfo info) {
      std::lock_guard<std::mutex> lock(regionInfoMutex);
      LOG(detail) << ">>> Region info event" << info.event;
      LOG(detail) << "id: " << info.id;
      LOG(detail) << "ptr: " << info.ptr;
      LOG(detail) << "size: " << info.size;
      LOG(detail) << "flags: " << info.flags;
      context.expectedRegionCallbacks -= 1;
      pendingRegionInfos.push_back(info);
      // We always want to handle these on the main loop
      uv_async_send(registry.get<DeviceState>().awakeMainThread);
    });
  }

  // Add a signal manager for SIGUSR1 so that we can force
  // an event from the outside, making sure that the event loop can
  // be unblocked (e.g. by a quitting DPL driver) even when there
  // is no data pending to be processed.
  uv_signal_t* sigusr1Handle = (uv_signal_t*)malloc(sizeof(uv_signal_t));
  uv_signal_init(mState.loop, sigusr1Handle);
  sigusr1Handle->data = &mDeviceContext;
  uv_signal_start(sigusr1Handle, on_signal_callback, SIGUSR1);

  /// Initialise the pollers
  DataProcessingDevice::initPollers();

  // Whenever we InitTask, we consider as if the previous iteration
  // was successful, so that even if there is no timer or receiving
  // channel, we can still start an enumeration.
  mWasActive = true;

  // We should be ready to run here. Therefore we copy all the
  // required parts in the DataProcessorContext. Eventually we should
  // do so on a per thread basis, with fine grained locks.
  mDataProcessorContexes.resize(1);
  this->fillContext(mDataProcessorContexes.at(0), mDeviceContext);

  /// We now run an event loop also in InitTask. This is needed to:
  /// * Make sure region registration callbacks are invoked
  /// on the main thread.
  /// * Wait for enough callbacks to be delivered before moving to START
  while (mDeviceContext.expectedRegionCallbacks > 0 && uv_run(mState.loop, UV_RUN_ONCE)) {
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

  deviceContext.device = this;
  deviceContext.spec = &mSpec;
  deviceContext.state = &mState;
  deviceContext.quotaEvaluator = &mQuotaEvaluator;
  deviceContext.stats = &mStats;
  context.isSink = false;
  context.balancingInputs = true;
  // If nothing is a sink, the rate limiting simply does not trigger.
  bool enableRateLimiting = std::stoi(fConfig->GetValue<std::string>("timeframes-rate-limit"));

  // This is needed because the internal injected dummy sink should not
  // try to balance inputs unless the rate limiting is requested.
  if (enableRateLimiting == false && deviceContext.spec->name == "internal-dpl-injected-dummy-sink") {
    context.balancingInputs = false;
  }
  if (enableRateLimiting) {
    for (auto& spec : mSpec.outputs) {
      if (spec.matcher.binding.value == "dpl-summary") {
        context.isSink = true;
        break;
      }
    }
  }

  context.relayer = mRelayer;
  context.registry = &mServiceRegistry;
  context.completed = &mCompleted;
  context.expirationHandlers = &mExpirationHandlers;
  context.timingInfo = &mServiceRegistry.get<TimingInfo>();
  context.allocator = &mAllocator;
  context.statefulProcess = &mStatefulProcess;
  context.statelessProcess = &mStatelessProcess;
  context.error = &mError;
  context.deviceContext = &deviceContext;
  /// Callback for the error handling
  context.errorHandling = &mErrorHandling;
  /// We must make sure there is no optional
  /// if we want to optimize the forwarding
  context.canForwardEarly = (mSpec.forwards.empty() == false) && mProcessingPolicies.earlyForward != EarlyForwardPolicy::NEVER;
  for (auto& forwarded : mSpec.forwards) {
    if (strncmp(DataSpecUtils::asConcreteOrigin(forwarded.matcher).str, "AOD", 3) == 0) {
      context.canForwardEarly = false;
      break;
    }
    if (DataSpecUtils::partialMatch(forwarded.matcher, o2::header::DataDescription{"RAWDATA"}) && mProcessingPolicies.earlyForward == EarlyForwardPolicy::NORAW) {
      context.canForwardEarly = false;
      break;
    }
    if (forwarded.matcher.lifetime == Lifetime::Optional) {
      context.canForwardEarly = false;
      break;
    }
  }
}

void DataProcessingDevice::PreRun()
{
  mDeviceContext.state->quitRequested = false;
  mDeviceContext.state->streaming = StreamingState::Streaming;
  for (auto& info : mDeviceContext.state->inputChannelInfos) {
    if (info.state != InputChannelState::Pull) {
      info.state = InputChannelState::Running;
    }
  }
  mServiceRegistry.preStartCallbacks();
  mServiceRegistry.get<CallbackService>()(CallbackService::Id::Start);
  startPollers();
}

void DataProcessingDevice::PostRun()
{
  stopPollers();
  mServiceRegistry.get<CallbackService>()(CallbackService::Id::Stop);
  mServiceRegistry.postStopCallbacks();
}

void DataProcessingDevice::Reset()
{
  mServiceRegistry.get<CallbackService>()(CallbackService::Id::Reset);
}

void DataProcessingDevice::Run()
{
  mState.loopReason = DeviceState::LoopReason::FIRST_LOOP;
  while (mState.transitionHandling != TransitionHandlingState::Expired) {
    if (mState.nextFairMQState.empty() == false) {
      this->ChangeState(mState.nextFairMQState.back());
      mState.nextFairMQState.pop_back();
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
      mServiceRegistry.get<DriverClient>().flushPending();
      auto shouldNotWait = (mWasActive &&
                            (mState.streaming != StreamingState::Idle) && (mState.activeSignals.empty())) ||
                           (mState.streaming == StreamingState::EndOfStreaming);
      if (mWasActive) {
        mState.loopReason |= DeviceState::LoopReason::PREVIOUSLY_ACTIVE;
      }
      if (NewStatePending()) {
        shouldNotWait = true;
        mState.loopReason |= DeviceState::LoopReason::NEW_STATE_PENDING;
      }
      if (mState.transitionHandling == TransitionHandlingState::NoTransition && NewStatePending()) {
        mState.transitionHandling = TransitionHandlingState::Requested;
        auto timeout = mDeviceContext.exitTransitionTimeout;
        if (timeout != 0 && mState.streaming != StreamingState::Idle) {
          mState.transitionHandling = TransitionHandlingState::Requested;
          uv_update_time(mState.loop);
          uv_timer_start(mDeviceContext.gracePeriodTimer, on_transition_requested_expired, timeout * 1000, 0);
          if (mProcessingPolicies.termination == TerminationPolicy::QUIT) {
            LOGP(info, "New state requested. Waiting for {} seconds before quitting.", timeout);
          } else {
            LOGP(info, "New state requested. Waiting for {} seconds before switching to READY state.", timeout);
          }
        } else {
          mState.transitionHandling = TransitionHandlingState::Expired;
          if (mProcessingPolicies.termination == TerminationPolicy::QUIT) {
            LOGP(info, "New state requested. No timeout set, quitting immediately as per --completion-policy");
          } else {
            LOGP(info, "New state requested. No timeout set, switching to READY state immediately");
          }
        }
      }
      TracyPlot("shouldNotWait", (int)shouldNotWait);
      if (mState.severityStack.empty() == false) {
        fair::Logger::SetConsoleSeverity((fair::Severity)mState.severityStack.back());
        mState.severityStack.pop_back();
      }
      // for (auto &info : mDeviceContext.state->inputChannelInfos)  {
      //   shouldNotWait |= info.readPolled;
      // }
      mState.loopReason = DeviceState::NO_REASON;
      if ((mState.tracingFlags & DeviceState::LoopReason::TRACE_CALLBACKS) != 0) {
        mState.severityStack.push_back((int)fair::Logger::GetConsoleSeverity());
        fair::Logger::SetConsoleSeverity(fair::Severity::trace);
      }
      uv_run(mState.loop, shouldNotWait ? UV_RUN_NOWAIT : UV_RUN_ONCE);
      if ((mState.loopReason & mState.tracingFlags) != 0) {
        mState.severityStack.push_back((int)fair::Logger::GetConsoleSeverity());
        fair::Logger::SetConsoleSeverity(fair::Severity::trace);
      } else if (mState.severityStack.empty() == false) {
        fair::Logger::SetConsoleSeverity((fair::Severity)mState.severityStack.back());
        mState.severityStack.pop_back();
      }
      TracyPlot("loopReason", (int64_t)(uint64_t)mState.loopReason);
      LOGP(debug, "Loop reason mask {:b} & {:b} = {:b}",
           mState.loopReason, mState.tracingFlags,
           mState.loopReason & mState.tracingFlags);

      if ((mState.loopReason & DeviceState::LoopReason::OOB_ACTIVITY) != 0) {
        LOGP(debug, "We were awakened by a OOB event. Rescanning everything.");
        mRelayer->rescan();
      }

      if (!mState.pendingOffers.empty()) {
        mQuotaEvaluator.updateOffers(mState.pendingOffers, uv_now(mState.loop));
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
      auto& handle = mHandles[streamRef.index];
      auto& stream = mStreams[streamRef.index];
      handle.data = &mStreams[streamRef.index];

      static std::function<void(ComputingQuotaOffer const&, ComputingQuotaStats const& stats)> reportExpiredOffer = [&monitoring = mServiceRegistry.get<o2::monitoring::Monitoring>()](ComputingQuotaOffer const& offer, ComputingQuotaStats const& stats) {
        monitoring.send(Metric{(uint64_t)stats.totalExpiredOffers, "resource-offer-expired"}.addTag(Key::Subsystem, Value::DPL));
        monitoring.send(Metric{(uint64_t)stats.totalExpiredBytes, "arrow-bytes-expired"}.addTag(Key::Subsystem, Value::DPL));
        monitoring.flushBuffer();
      };

      // Deciding wether to run or not can be done by passing a request to
      // the evaluator. In this case, the request is always satisfied and
      // we run on whatever resource is available.
      bool enough = mQuotaEvaluator.selectOffer(streamRef.index, mSpec.resourcePolicy.request, uv_now(mState.loop));

      if (enough) {
        stream.id = streamRef;
        stream.running = true;
        stream.context = &mDataProcessorContexes.at(0);
#ifdef DPL_ENABLE_THREADING
        stream.task.data = &handle;
        uv_queue_work(mState.loop, &stream.task, run_callback, run_completion);
#else
        run_callback(&handle);
        run_completion(&handle, 0);
#endif
      } else {
        mDataProcessorContexes.at(0).deviceContext->quotaEvaluator->handleExpired(reportExpiredOffer);
        mWasActive = false;
      }
    } else {
      mWasActive = false;
    }
    FrameMark;
  }
  /// Cleanup messages which are still pending on exit.
  for (size_t ci = 0; ci < mDeviceContext.spec->inputChannels.size(); ++ci) {
    auto& info = mDeviceContext.state->inputChannelInfos[ci];
    info.parts.fParts.clear();
  }
  mState.transitionHandling = TransitionHandlingState::NoTransition;
}

/// We drive the state loop ourself so that we will be able to support
/// non-data triggers like those which are time based.
void DataProcessingDevice::doPrepare(DataProcessorContext& context)
{
  ZoneScopedN("DataProcessingDevice::doPrepare");
  context.registry->get<DataProcessingStats>().beginIterationTimestamp = uv_hrtime() / 1000000;

  *context.wasActive = false;
  {
    ZoneScopedN("CallbackService::Id::ClockTick");
    context.registry->get<CallbackService>()(CallbackService::Id::ClockTick);
  }
  // Whether or not we had something to do.

  // Notice that fake input channels (InputChannelState::Pull) cannot possibly
  // expect to receive an EndOfStream signal. Thus we do not wait for these
  // to be completed. In the case of data source devices, as they do not have
  // real data input channels, they have to signal EndOfStream themselves.
  context.allDone = std::any_of(context.deviceContext->state->inputChannelInfos.begin(), context.deviceContext->state->inputChannelInfos.end(), [](const auto& info) {
    return info.parts.fParts.empty() == true && info.state != InputChannelState::Pull;
  });

  // Whether or not all the channels are completed
  LOGP(debug, "Processing {} input channels.", context.deviceContext->spec->inputChannels.size());
  /// Sort channels by oldest possible timeframe and
  /// process them in such order.
  static std::vector<int> pollOrder;
  pollOrder.resize(context.deviceContext->state->inputChannelInfos.size());
  std::iota(pollOrder.begin(), pollOrder.end(), 0);
  std::sort(pollOrder.begin(), pollOrder.end(), [&infos = context.deviceContext->state->inputChannelInfos](int a, int b) {
    return infos[a].oldestForChannel.value < infos[b].oldestForChannel.value;
  });

  // Nothing to poll...
  if (pollOrder.empty()) {
    return;
  }
  auto currentOldest = context.deviceContext->state->inputChannelInfos[pollOrder.front()].oldestForChannel;
  auto currentNewest = context.deviceContext->state->inputChannelInfos[pollOrder.back()].oldestForChannel;
  auto delta = currentNewest.value - currentOldest.value;
  LOGP(debug, "oldest possible timeframe range {}, {} => {} delta", currentOldest.value, currentNewest.value,
       delta);
  auto& infos = context.deviceContext->state->inputChannelInfos;

  if (context.balancingInputs) {
    static uint64_t ahead = getenv("DPL_MAX_CHANNEL_AHEAD") ? std::atoll(getenv("DPL_MAX_CHANNEL_AHEAD")) : 8;
    auto newEnd = std::remove_if(pollOrder.begin(), pollOrder.end(), [&infos, limitNew = currentOldest.value + ahead](int a) -> bool {
      return infos[a].oldestForChannel.value > limitNew;
    });
    pollOrder.erase(newEnd, pollOrder.end());
  }
  LOGP(debug, "processing {} channels", pollOrder.size());

  for (auto sci : pollOrder) {
    auto& info = context.deviceContext->state->inputChannelInfos[sci];
    auto& channelSpec = context.deviceContext->spec->inputChannels[sci];
    LOGP(debug, "Processing channel {}", channelSpec.name);

    if (info.state != InputChannelState::Completed && info.state != InputChannelState::Pull) {
      context.allDone = false;
    }
    if (info.state != InputChannelState::Running) {
      // Remember to flush data if we are not running
      // and there is some message pending.
      if (info.parts.Size()) {
        DataProcessingDevice::handleData(context, info);
      }
      LOGP(debug, "Flushing channel {} which is in state {} and has {} parts still pending.", channelSpec.name, (int)info.state, info.parts.Size());
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
        DataProcessingDevice::handleData(context, info);
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

void DataProcessingDevice::doRun(DataProcessorContext& context)
{
  auto switchState = [&registry = context.registry,
                      &state = context.deviceContext->state](StreamingState newState) {
    LOG(debug) << "New state " << (int)newState << " old state " << (int)state->streaming;
    state->streaming = newState;
    registry->get<ControlService>().notifyStreamingState(state->streaming);
  };

  if (context.deviceContext->state->streaming == StreamingState::Idle) {
    *context.wasActive = false;
    return;
  }

  context.completed->clear();
  context.completed->reserve(16);
  *context.wasActive |= DataProcessingDevice::tryDispatchComputation(context, *context.completed);
  DanglingContext danglingContext{*context.registry};

  context.registry->preDanglingCallbacks(danglingContext);
  if (*context.wasActive == false) {
    context.registry->get<CallbackService>()(CallbackService::Id::Idle);
  }
  auto activity = context.relayer->processDanglingInputs(*context.expirationHandlers, *context.registry, true);
  *context.wasActive |= activity.expiredSlots > 0;

  context.completed->clear();
  *context.wasActive |= DataProcessingDevice::tryDispatchComputation(context, *context.completed);

  context.registry->postDanglingCallbacks(danglingContext);

  // If we got notified that all the sources are done, we call the EndOfStream
  // callback and return false. Notice that what happens next is actually
  // dependent on the callback, not something which is controlled by the
  // framework itself.
  if (context.allDone == true && context.deviceContext->state->streaming == StreamingState::Streaming) {
    switchState(StreamingState::EndOfStreaming);
    *context.wasActive = true;
  }

  if (context.deviceContext->state->streaming == StreamingState::EndOfStreaming) {
    LOGP(debug, "We are in EndOfStreaming. Flushing queues.");
    context.registry->get<DriverClient>().flushPending();
    // We keep processing data until we are Idle.
    // FIXME: not sure this is the correct way to drain the queues, but
    // I guess we will see.
    /// Besides flushing the queues we must make sure we do not have only
    /// timers as they do not need to be further processed.
    bool hasOnlyGenerated = (context.deviceContext->spec->inputChannels.size() == 1) && (context.deviceContext->spec->inputs[0].matcher.lifetime == Lifetime::Timer || context.deviceContext->spec->inputs[0].matcher.lifetime == Lifetime::Enumeration);
    while (DataProcessingDevice::tryDispatchComputation(context, *context.completed) && hasOnlyGenerated == false) {
      context.relayer->processDanglingInputs(*context.expirationHandlers, *context.registry, false);
    }
    EndOfStreamContext eosContext{*context.registry, *context.allocator};

    context.registry->preEOSCallbacks(eosContext);
    context.registry->get<CallbackService>()(CallbackService::Id::EndOfStream, eosContext);
    context.registry->postEOSCallbacks(eosContext);

    for (auto& channel : context.deviceContext->spec->outputChannels) {
      DataProcessingHelpers::sendEndOfStream(*context.deviceContext->device, channel);
    }
    // This is needed because the transport is deleted before the device.
    context.relayer->clear();
    switchState(StreamingState::Idle);
    if (hasOnlyGenerated) {
      *context.wasActive = false;
    } else {
      *context.wasActive = true;
    }
    // On end of stream we shut down all output pollers.
    for (auto& poller : context.deviceContext->state->activeOutputPollers) {
      uv_poll_stop(poller);
    }
    return;
  }

  if (context.deviceContext->state->streaming == StreamingState::Idle) {
    // On end of stream we shut down all output pollers.
    for (auto& poller : context.deviceContext->state->activeOutputPollers) {
      uv_poll_stop(poller);
    }
  }

  return;
}

void DataProcessingDevice::ResetTask()
{
  mRelayer->clear();
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
void DataProcessingDevice::handleData(DataProcessorContext& context, InputChannelInfo& info)
{
  ZoneScopedN("DataProcessingDevice::handleData");
  assert(context.deviceContext->spec->inputChannels.empty() == false);
  assert(info.parts.Size() > 0);

  // Initial part. Let's hide all the unnecessary and have
  // simple lambdas for each of the steps I am planning to have.
  assert(!context.deviceContext->spec->inputs.empty());

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
  auto getInputTypes = [&stats = context.registry->get<DataProcessingStats>(),
                        &info, &context]() -> std::optional<std::vector<InputInfo>> {
    auto& parts = info.parts;
    stats.inputParts = parts.Size();

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
    assert(std::accumulate(results.begin(), results.end(), 0, [](size_t const& count, auto const& element) -> size_t { return count + element.size; }));
    if (results.size() + nTotalPayloads != parts.Size()) {
      LOG(error) << "inconsistent number of inputs extracted";
      return std::nullopt;
    }
    return results;
  };

  auto reportError = [&registry = *context.registry](const char* message) {
    registry.get<DataProcessingStats>().errorCount++;
  };

  auto handleValidMessages = [&info, &context = context, &relayer = *context.relayer, &reportError](std::vector<InputInfo> const& inputInfos) {
    static WaitBackpressurePolicy policy;
    auto& parts = info.parts;
    // We relay execution to make sure we have a complete set of parts
    // available.
    bool hasBackpressure = false;
    bool hasData = false;
    bool hasDomainInfo = false;
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
          auto relayed = relayer.relay(parts.At(headerIndex)->GetData(),
                                       &parts.At(headerIndex),
                                       nMessages,
                                       nPayloadsPerHeader);
          switch (relayed) {
            case DataRelayer::Backpressured:
              if (info.normalOpsNotified == true && info.backpressureNotified == false) {
                LOGP(alarm, "Backpressure on channel {}. Waiting.", info.channel->GetName());
                auto& monitoring = context.registry->get<o2::monitoring::Monitoring>();
                monitoring.send(o2::monitoring::Metric{1, fmt::format("backpressure_{}", info.channel->GetName())});
                info.backpressureNotified = true;
                info.normalOpsNotified = false;
              }
              policy.backpressure(info);
              hasBackpressure = true;
              break;
            case DataRelayer::Dropped:
            case DataRelayer::Invalid:
            case DataRelayer::WillRelay:
              if (info.normalOpsNotified == false && info.backpressureNotified == true) {
                LOGP(info, "Back to normal on channel {}.", info.channel->GetName());
                auto& monitoring = context.registry->get<o2::monitoring::Monitoring>();
                monitoring.send(o2::monitoring::Metric{0, fmt::format("backpressure_{}", info.channel->GetName())});
                info.normalOpsNotified = true;
                info.backpressureNotified = false;
              }
              break;
          }
        } break;
        case InputType::SourceInfo: {
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
          if (hasBackpressure) {
            break;
          }
          *context.wasActive = true;
          auto headerIndex = input.position;
          auto payloadIndex = input.position + 1;
          assert(payloadIndex < parts.Size());
          // FIXME: the message with the end of stream cannot contain
          //        split parts.

          auto dih = o2::header::get<DomainInfoHeader*>(parts.At(headerIndex)->GetData());
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
      context.registry->domainInfoUpdatedCallback(*context.registry, oldestPossibleTimeslice, info.id);
      context.registry->get<CallbackService>()(CallbackService::Id::DomainInfoUpdated, (ServiceRegistry&)*context.registry, (size_t)oldestPossibleTimeslice, (ChannelIndex)info.id);
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
auto calculateInputRecordLatency(InputRecord const& record, uint64_t currentTime) -> DataProcessingStats::InputLatency
{
  DataProcessingStats::InputLatency result{static_cast<int>(-1), 0};

  for (auto& item : record) {
    auto* header = o2::header::get<DataProcessingHeader*>(item.header);
    if (header == nullptr) {
      continue;
    }
    int partLatency = currentTime - header->creation;
    result.minLatency = std::min(result.minLatency, partLatency);
    result.maxLatency = std::max(result.maxLatency, partLatency);
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

bool DataProcessingDevice::tryDispatchComputation(DataProcessorContext& context, std::vector<DataRelayer::RecordAction>& completed)
{
  ZoneScopedN("DataProcessingDevice::tryDispatchComputation");
  LOGP(debug, "DataProcessingDevice::tryDispatchComputation");
  // This is the actual hidden state for the outer loop. In case we decide we
  // want to support multithreaded dispatching of operations, I can simply
  // move these to some thread local store and the rest of the lambdas
  // should work just fine.
  std::vector<MessageSet> currentSetOfInputs;

  auto reportError = [&registry = *context.registry](const char* message) {
    registry.get<DataProcessingStats>().errorCount++;
  };

  // For the moment we have a simple "immediately dispatch" policy for stuff
  // in the cache. This could be controlled from the outside e.g. by waiting
  // for a few sets of inputs to arrive before we actually dispatch the
  // computation, however this can be defined at a later stage.
  auto canDispatchSomeComputation = [&completed,
                                     &relayer = context.relayer]() -> bool {
    relayer->getReadyToProcess(completed);
    return completed.empty() == false;
  };

  // We use this to get a list with the actual indexes in the cache which
  // indicate a complete set of inputs. Notice how I fill the completed
  // vector and return it, so that I can have a nice for loop iteration later
  // on.
  auto getReadyActions = [&relayer = context.relayer,
                          &completed,
                          &stats = context.registry->get<DataProcessingStats>()]() -> std::vector<DataRelayer::RecordAction> {
    stats.pendingInputs = (int)relayer->getParallelTimeslices() - completed.size();
    stats.incomplete = completed.empty() ? 1 : 0;
    return completed;
  };

  //
  auto getInputSpan = [&relayer = context.relayer,
                       &currentSetOfInputs](TimesliceSlot slot, bool consume = true) {
    if (consume) {
      currentSetOfInputs = relayer->consumeAllInputsForTimeslice(slot);
    } else {
      currentSetOfInputs = relayer->consumeExistingInputsForTimeslice(slot);
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
        // - InputRecord provides all payloads as header-payload pairs
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

  auto markInputsAsDone = [&relayer = context.relayer](TimesliceSlot slot) -> void {
    relayer->updateCacheStatus(slot, CacheEntryStatus::RUNNING, CacheEntryStatus::DONE);
  };

  // I need a preparation step which gets the current timeslice id and
  // propagates it to the various contextes (i.e. the actual entities which
  // create messages) because the messages need to have the timeslice id into
  // it.
  auto prepareAllocatorForCurrentTimeSlice = [&timingInfo = context.timingInfo,
                                              &relayer = context.relayer](TimesliceSlot i) {
    ZoneScopedN("DataProcessingDevice::prepareForCurrentTimeslice");
    auto timeslice = relayer->getTimesliceForSlot(i);
    timingInfo->timeslice = timeslice.value;
    timingInfo->tfCounter = relayer->getFirstTFCounterForSlot(i);
    timingInfo->firstTFOrbit = relayer->getFirstTFOrbitForSlot(i);
    timingInfo->runNumber = relayer->getRunNumberForSlot(i);
    timingInfo->creation = relayer->getCreationTimeForSlot(i);
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

  // This is how we do the forwarding, i.e. we push
  // the inputs which are shared between this device and others
  // to the next one in the daisy chain.
  // FIXME: do it in a smarter way than O(N^2)
  auto forwardInputs = [&reportError,
                        &spec = context.deviceContext->spec,
                        &timesliceIndex = context.registry->get<TimesliceIndex>(),
                        &device = context.deviceContext->device, &currentSetOfInputs](TimesliceSlot slot, InputRecord& record, bool copy, bool consume = true) {
    ZoneScopedN("forward inputs");
    LOGP(debug, "DataProcessingDevice::tryDispatchComputation::forwardInputs");
    assert(record.size() == currentSetOfInputs.size());
    // we collect all messages per forward in a map and send them together
    std::vector<fair::mq::Parts> forwardedParts;
    forwardedParts.resize(spec->forwards.size());

    std::vector<size_t> forwardMap;
    forwardMap.resize(spec->forwards.size());
    std::unordered_map<std::string, size_t> tmpMap;
    for (size_t fi = 0; fi < spec->forwards.size(); fi++) {
      forwardMap[fi] = tmpMap.try_emplace(spec->forwards[fi].channel, fi).first->second;
    }

    for (size_t ii = 0, ie = record.size(); ii < ie; ++ii) {
      DataRef input = record.getByPos(ii);

      // If is now possible that the record is not complete when
      // we forward it, because of a custom completion policy.
      // this means that we need to skip the empty entries in the
      // record for being forwarded.
      if (input.header == nullptr) {
        continue;
      }
      auto sih = o2::header::get<SourceInfoHeader*>(input.header);
      if (sih) {
        continue;
      }

      auto dih = o2::header::get<DomainInfoHeader*>(input.header);
      if (dih) {
        continue;
      }

      auto dh = o2::header::get<DataHeader*>(input.header);
      if (!dh) {
        reportError("Forwarding a non-DataHeader?");
        continue;
      }
      auto dph = o2::header::get<DataProcessingHeader*>(input.header);
      if (!dph) {
        reportError("Header stack does not contain DataProcessingHeader");
        continue;
      }

      int cachedForwardingChoice = -1;

      for (size_t pi = 0; pi < currentSetOfInputs[ii].size(); ++pi) {
        auto& messageSet = currentSetOfInputs[ii];
        auto& header = messageSet.header(pi);
        auto& payload = messageSet.payload(pi);

        if (header.get() == nullptr) {
          // Missing an header is not an error anymore.
          // it simply means that we did not receive the
          // given input, but we were asked to
          // consume existing, so we skip it.
          continue;
        }
        if (payload.get() == nullptr && consume == true) {
          // If the payload is not there, it means we already
          // processed it with ConsumeExisiting. Therefore we
          // need to do something only if this is the last consume.
          header.reset(nullptr);
          continue;
        }

        auto fdph = o2::header::get<DataProcessingHeader*>(header->GetData());
        if (fdph == nullptr) {
          LOG(error) << "Data is missing DataProcessingHeader";
          continue;
        }
        auto fdh = o2::header::get<DataHeader*>(header->GetData());
        if (fdh == nullptr) {
          LOG(error) << "Data is missing DataHeader";
          continue;
        }
        // We need to find the forward route only for the first
        // part of a split payload. All the others will use the same.
        // but always check if we have a sequence of multiple payloads
        if (fdh->splitPayloadIndex == 0 || fdh->splitPayloadParts <= 1 || messageSet.getNumberOfPayloads(pi) > 1) {
          cachedForwardingChoice = -1;
          for (size_t fi = 0; fi < spec->forwards.size(); fi++) {
            auto& forward = spec->forwards[fi];
            if (DataSpecUtils::match(forward.matcher, fdh->dataOrigin, fdh->dataDescription, fdh->subSpecification) == false || (fdph->startTime % forward.maxTimeslices) != forward.timeslice) {
              continue;
            }
            cachedForwardingChoice = forwardMap[fi];
            break;
          }
        }
        /// We did not find a match. Skip it.
        if (cachedForwardingChoice == -1) {
          continue;
        }

        if (copy) {
          auto&& newHeader = header->GetTransport()->CreateMessage();
          newHeader->Copy(*header);
          forwardedParts[cachedForwardingChoice].AddPart(std::move(newHeader));

          for (size_t payloadIndex = 0; payloadIndex < messageSet.getNumberOfPayloads(pi); ++payloadIndex) {
            auto&& newPayload = header->GetTransport()->CreateMessage();
            newPayload->Copy(*messageSet.payload(pi, payloadIndex));
            forwardedParts[cachedForwardingChoice].AddPart(std::move(newPayload));
          }
        } else {
          forwardedParts[cachedForwardingChoice].AddPart(std::move(messageSet.header(pi)));
          for (size_t payloadIndex = 0; payloadIndex < messageSet.getNumberOfPayloads(pi); ++payloadIndex) {
            forwardedParts[cachedForwardingChoice].AddPart(std::move(messageSet.payload(pi, payloadIndex)));
          }
        }
      }
    }
    for (size_t fi = 0; fi < spec->forwards.size(); fi++) {
      auto& channel = device->GetChannel(spec->forwards[fi].channel, 0);
      if (forwardedParts[fi].Size() != 0) {
        // in DPL we are using subchannel 0 only
        channel.Send(forwardedParts[fi]);
      }
    }
    for (size_t fi = 0; fi < spec->forwards.size(); fi++) {
      auto& channel = device->GetChannel(spec->forwards[fi].channel, 0);
      // The oldest possible timeslice for a forwarded message
      // is conservatively the one of the device doing the forwarding.
      if (spec->forwards[fi].channel.rfind("from_", 0) == 0) {
        auto oldestTimeslice = timesliceIndex.getOldestPossibleOutput();
        DataProcessingHelpers::sendOldestPossibleTimeframe(channel, oldestTimeslice.timeslice.value);
        LOGP(debug, "Forwarding to channel {} oldest possible timeslice {}", spec->forwards[fi].channel, oldestTimeslice.timeslice.value);
      }
    }
  };

  auto switchState = [&control = context.registry->get<ControlService>(),
                      &state = context.deviceContext->state](StreamingState newState) {
    state->streaming = newState;
    control.notifyStreamingState(state->streaming);
  };

  if (canDispatchSomeComputation() == false) {
    LOGP(debug, "No computations available for dispatching.");
    return false;
  }

  auto postUpdateStats = [&stats = context.registry->get<DataProcessingStats>()](DataRelayer::RecordAction const& action, InputRecord const& record, uint64_t tStart) {
    std::atomic_thread_fence(std::memory_order_release);
    for (size_t ai = 0; ai != record.size(); ai++) {
      auto cacheId = action.slot.index * record.size() + ai;
      auto state = record.isValid(ai) ? 3 : 0;
      update_maximum(stats.statesSize, cacheId + 1);
      assert(cacheId < DataProcessingStats::MAX_RELAYER_STATES);
      stats.relayerState[cacheId].store(state);
    }
    uint64_t tEnd = uv_hrtime();
    stats.lastElapsedTimeMs = tEnd - tStart;
    stats.lastProcessedSize = calculateTotalInputRecordSize(record);
    stats.totalProcessedSize += stats.lastProcessedSize;
    stats.lastLatency = calculateInputRecordLatency(record, tStart);
  };

  auto preUpdateStats = [&stats = context.registry->get<DataProcessingStats>()](DataRelayer::RecordAction const& action, InputRecord const& record, uint64_t) {
    std::atomic_thread_fence(std::memory_order_release);
    for (size_t ai = 0; ai != record.size(); ai++) {
      auto cacheId = action.slot.index * record.size() + ai;
      auto state = record.isValid(ai) ? 2 : 0;
      update_maximum(stats.statesSize, cacheId + 1);
      assert(cacheId < DataProcessingStats::MAX_RELAYER_STATES);
      stats.relayerState[cacheId].store(state);
    }
  };

  // This is the main dispatching loop
  LOGP(debug, "Processing actions:");
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
    InputRecord record{context.deviceContext->spec->inputs,
                       span,
                       *context.registry};
    ProcessingContext processContext{record, *context.registry, *context.allocator};
    {
      ZoneScopedN("service pre processing");
      context.registry->preProcessingCallbacks(processContext);
    }
    if (action.op == CompletionPolicy::CompletionOp::Discard) {
      LOGP(debug, "  - Action is to Discard");
      context.registry->postDispatchingCallbacks(processContext);
      if (context.deviceContext->spec->forwards.empty() == false) {
        forwardInputs(action.slot, record, false);
        continue;
      }
    }
    // If there is no optional inputs we canForwardEarly
    // the messages to that parallel processing can happen.
    // In this case we pass true to indicate that we want to
    // copy the messages to the subsequent data processor.
    bool hasForwards = context.deviceContext->spec->forwards.empty() == false;
    bool consumeSomething = action.op == CompletionPolicy::CompletionOp::Consume || action.op == CompletionPolicy::CompletionOp::ConsumeExisting;

    if (context.canForwardEarly && hasForwards && consumeSomething) {
      LOGP(debug, "  - Early forwarding");
      forwardInputs(action.slot, record, true, action.op == CompletionPolicy::CompletionOp::Consume);
    }
    markInputsAsDone(action.slot);

    uint64_t tStart = uv_hrtime();
    preUpdateStats(action, record, tStart);

    static bool noCatch = getenv("O2_NO_CATCHALL_EXCEPTIONS") && strcmp(getenv("O2_NO_CATCHALL_EXCEPTIONS"), "0");

    auto runNoCatch = [&context, &processContext](DataRelayer::RecordAction& action) {
      if (context.deviceContext->state->quitRequested == false) {
        {
          ZoneScopedN("service post processing");
          // Callbacks from services
          context.registry->preProcessingCallbacks(processContext);
          // Callbacks from users
          context.registry->get<CallbackService>()(CallbackService::Id::PreProcessing, *(context.registry), (int)action.op);
        }
        if (*context.statefulProcess) {
          ZoneScopedN("statefull process");
          (*context.statefulProcess)(processContext);
        } else if (*context.statelessProcess) {
          ZoneScopedN("stateless process");
          (*context.statelessProcess)(processContext);
        } else {
          context.deviceContext->state->streaming = StreamingState::Idle;
        }

        // Notify the sink we just consumed some timeframe data
        if (context.isSink && action.op == CompletionPolicy::CompletionOp::Consume) {
          context.allocator->make<int>(OutputRef{"dpl-summary", compile_time_hash(context.deviceContext->spec->name.c_str())}, 1);
        }

        {
          ZoneScopedN("service post processing");
          context.registry->get<CallbackService>()(CallbackService::Id::PostProcessing, *(context.registry), (int)action.op);
          context.registry->postProcessingCallbacks(processContext);
        }
      }
    };

    if ((context.deviceContext->state->tracingFlags & DeviceState::LoopReason::TRACE_USERCODE) != 0) {
      context.deviceContext->state->severityStack.push_back((int)fair::Logger::GetConsoleSeverity());
      fair::Logger::SetConsoleSeverity(fair::Severity::trace);
    }
    if (noCatch) {
      runNoCatch(action);
    } else {
      try {
        runNoCatch(action);
      } catch (std::exception& ex) {
        ZoneScopedN("error handling");
        /// Convert a standard exception to a RuntimeErrorRef
        /// Notice how this will lose the backtrace information
        /// and report the exception coming from here.
        auto e = runtime_error(ex.what());
        (*context.errorHandling)(e, record);
      } catch (o2::framework::RuntimeErrorRef e) {
        ZoneScopedN("error handling");
        (*context.errorHandling)(e, record);
      }
    }
    if (context.deviceContext->state->severityStack.empty() == false) {
      fair::Logger::SetConsoleSeverity((fair::Severity)context.deviceContext->state->severityStack.back());
      context.deviceContext->state->severityStack.pop_back();
    }

    postUpdateStats(action, record, tStart);
    // We forward inputs only when we consume them. If we simply Process them,
    // we keep them for next message arriving.
    if (action.op == CompletionPolicy::CompletionOp::Consume) {
      context.registry->postDispatchingCallbacks(processContext);
      context.registry->get<CallbackService>()(CallbackService::Id::DataConsumed, *(context.registry));
    }
    if ((context.canForwardEarly == false) && hasForwards && consumeSomething) {
      LOGP(debug, "Late forwarding");
      forwardInputs(action.slot, record, false, action.op == CompletionPolicy::CompletionOp::Consume);
    }
    context.registry->postForwardingCallbacks(processContext);
    if (action.op == CompletionPolicy::CompletionOp::Consume) {
#ifdef TRACY_ENABLE
      cleanupRecord(record);
#endif
    } else if (action.op == CompletionPolicy::CompletionOp::Process) {
      cleanTimers(action.slot, record);
    }
  }
  // We now broadcast the end of stream if it was requested
  if (context.deviceContext->state->streaming == StreamingState::EndOfStreaming) {
    LOGP(debug, "Broadcasting end of stream");
    for (auto& channel : context.deviceContext->spec->outputChannels) {
      DataProcessingHelpers::sendEndOfStream(*context.deviceContext->device, channel);
    }
    switchState(StreamingState::Idle);
  }

  return true;
}

void DataProcessingDevice::error(const char* msg)
{
  LOG(error) << msg;
  mServiceRegistry.get<DataProcessingStats>().errorCount++;
}

std::unique_ptr<ConfigParamStore> DeviceConfigurationHelpers::getConfiguration(ServiceRegistry& registry, const char* name, std::vector<ConfigParamSpec> const& options)
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
