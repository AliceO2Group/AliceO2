// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "Framework/DriverClient.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/FairOptionsRetriever.h"
#include "ConfigurationOptionsRetriever.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Framework/CallbackService.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/InputRecord.h"
#include "Framework/Signpost.h"
#include "Framework/SourceInfoHeader.h"
#include "Framework/Logger.h"
#include "Framework/DriverClient.h"
#include "Framework/Monitoring.h"
#include "PropertyTreeHelpers.h"
#include "DataProcessingStatus.h"
#include "DataProcessingHelpers.h"
#include "DataRelayerHelpers.h"

#include "ScopedExit.h"

#include <Framework/Tracing.h>

#include <fairmq/FairMQParts.h>
#include <fairmq/FairMQSocket.h>
#include <options/FairMQProgOptions.h>
#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>
#include <TMessage.h>
#include <TClonesArray.h>

#include <algorithm>
#include <vector>
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
  DeviceState* state = (DeviceState*)handle->data;
  state->loopReason |= DeviceState::TIMER_EXPIRED;
}

DataProcessingDevice::DataProcessingDevice(RunningWorkflowInfo const& runningWorkflow, RunningDeviceRef ref, ServiceRegistry& registry, DeviceState& state)
  : mSpec{runningWorkflow.devices[ref.index]},
    mState{state},
    mInit{mSpec.algorithm.onInit},
    mStatefulProcess{nullptr},
    mStatelessProcess{mSpec.algorithm.onProcess},
    mError{mSpec.algorithm.onError},
    mConfigRegistry{nullptr},
    mAllocator{&mTimingInfo, &registry, mSpec.outputs},
    mServiceRegistry{registry},
    mQuotaEvaluator{state.loop}
{
  /// FIXME: move erro handling to a service?
  if (mError != nullptr) {
    mErrorHandling = [&errorCallback = mError,
                      &serviceRegistry = mServiceRegistry](RuntimeErrorRef e, InputRecord& record) {
      ZoneScopedN("Error handling");
      auto& err = error_from_ref(e);
      LOGP(ERROR, "Exception caught: {} ", err.what);
      backtrace_symbols_fd(err.backtrace, err.maxBacktrace, STDERR_FILENO);
      serviceRegistry.get<DataProcessingStats>().exceptionCount++;
      ErrorContext errorContext{record, serviceRegistry, e};
      errorCallback(errorContext);
    };
  } else {
    mErrorHandling = [&errorPolicy = mErrorPolicy,
                      &serviceRegistry = mServiceRegistry](RuntimeErrorRef e, InputRecord& record) {
      ZoneScopedN("Error handling");
      auto& err = error_from_ref(e);
      LOGP(ERROR, "Exception caught: {} ", err.what);
      backtrace_symbols_fd(err.backtrace, err.maxBacktrace, STDERR_FILENO);
      serviceRegistry.get<DataProcessingStats>().exceptionCount++;
      switch (errorPolicy) {
        case TerminationPolicy::QUIT:
          throw e;
        default:
          break;
      }
    };
  }
  // One task for now.
  mStreams.resize(1);
  mHandles.resize(1);
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
  context.deviceContext->quotaEvaluator->dispose(task->offer);
  task->running = false;
  ZoneScopedN("run_completion");
}

// Context for polling
struct PollerContext {
  char const* name = nullptr;
  uv_loop_t* loop = nullptr;
  DataProcessingDevice* device = nullptr;
  DeviceState* state = nullptr;
  int fd;
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

void on_communication_requested(uv_async_t* s)
{
  DeviceState* state = (DeviceState*)s->data;
  state->loopReason |= DeviceState::METRICS_MUST_FLUSH;
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
  // If available use the ConfigurationInterface, otherwise go for
  // the command line options.
  bool hasConfiguration = false;
  bool hasOverrides = false;
  if (mServiceRegistry.active<ConfigurationInterface>()) {
    auto& cfg = mServiceRegistry.get<ConfigurationInterface>();
    hasConfiguration = true;
    try {
      cfg.getRecursive(mSpec.name);
      hasOverrides = true;
    } catch (...) {
      // No overrides...
    }
  }
  // We only use the configuration file if we have a stanza for the given
  // dataprocessor
  std::vector<std::unique_ptr<ParamRetriever>> retrievers;
  if (hasConfiguration && hasOverrides) {
    auto& cfg = mServiceRegistry.get<ConfigurationInterface>();
    retrievers.emplace_back(std::make_unique<ConfigurationOptionsRetriever>(&cfg, mSpec.name));
  } else {
    retrievers.emplace_back(std::make_unique<FairOptionsRetriever>(GetConfig()));
  }
  auto configStore = std::make_unique<ConfigParamStore>(mSpec.options, std::move(retrievers));
  configStore->preload();
  configStore->activate();
  using boost::property_tree::ptree;

  /// Dump the configuration so that we can get it from the driver.
  for (auto& entry : configStore->store()) {
    std::stringstream ss;
    std::string str;
    if (entry.second.empty() == false) {
      boost::property_tree::json_parser::write_json(ss, entry.second, false);
      str = ss.str();
      str.pop_back(); //remove EoL
    } else {
      str = entry.second.get_value<std::string>();
    }
    std::string configString = fmt::format("[CONFIG];{}={};1;{}", entry.first, str, configStore->provenance(entry.first.c_str())).c_str();
    mServiceRegistry.get<DriverClient>().tell(configString.c_str());
  }

  mConfigRegistry = std::make_unique<ConfigParamRegistry>(std::move(configStore));

  mExpirationHandlers.clear();

  auto distinct = DataRelayerHelpers::createDistinctRouteIndex(mSpec.inputs);
  int i = 0;
  for (auto& di : distinct) {
    auto& route = mSpec.inputs[di];
    if (route.configurator.has_value() == false) {
      i++;
      continue;
    }
    ExpirationHandler handler{
      RouteIndex{i++},
      route.matcher.lifetime,
      route.configurator->creatorConfigurator(mState, *mConfigRegistry),
      route.configurator->danglingConfigurator(mState, *mConfigRegistry),
      route.configurator->expirationConfigurator(mState, *mConfigRegistry)};
    mExpirationHandlers.emplace_back(std::move(handler));
  }

  if (mInit) {
    InitContext initContext{*mConfigRegistry, mServiceRegistry};
    mStatefulProcess = mInit(initContext);
  }
  mState.inputChannelInfos.resize(mSpec.inputChannels.size());
  /// Internal channels which will never create an actual message
  /// should be considered as in "Pull" mode, since we do not
  /// expect them to create any data.
  for (size_t ci = 0; ci < mSpec.inputChannels.size(); ++ci) {
    auto& name = mSpec.inputChannels[ci].name;
    if (name.find(mSpec.channelPrefix + "from_internal-dpl-clock") == 0) {
      mState.inputChannelInfos[ci].state = InputChannelState::Pull;
    } else if (name.find(mSpec.channelPrefix + "from_internal-dpl-ccdb-backend") == 0) {
      mState.inputChannelInfos[ci].state = InputChannelState::Pull;
    }
  }
  uv_async_t* wakeHandle = (uv_async_t*)malloc(sizeof(uv_async_t));
  assert(mState.loop);
  int res = uv_async_init(mState.loop, wakeHandle, on_communication_requested);
  wakeHandle->data = &mState;
  if (res < 0) {
    LOG(ERROR) << "Unable to initialise subscription";
  }

  /// This should post a message on the queue...
  SubscribeToNewTransition("dpl", [wakeHandle](fair::mq::Transition t) {
    int res = uv_async_send(wakeHandle);
    if (res < 0) {
      LOG(ERROR) << "Unable to notify subscription";
    }
    LOG(debug) << "State transition requested";
  });
}

void on_signal_callback(uv_signal_t* handle, int signum)
{
  ZoneScopedN("Signal callaback");
  LOG(debug) << "Signal " << signum << " received.";
  DeviceState* state = (DeviceState*)handle->data;
  state->loopReason |= DeviceState::SIGNAL_ARRIVED;
}

void DataProcessingDevice::InitTask()
{
  for (auto& channel : fChannels) {
    channel.second.at(0).Transport()->SubscribeToRegionEvents([& pendingRegionInfos = mPendingRegionInfos, &regionInfoMutex = mRegionInfoMutex](FairMQRegionInfo info) {
      std::lock_guard<std::mutex> lock(regionInfoMutex);
      LOG(debug) << ">>> Region info event" << info.event;
      LOG(debug) << "id: " << info.id;
      LOG(debug) << "ptr: " << info.ptr;
      LOG(debug) << "size: " << info.size;
      LOG(debug) << "flags: " << info.flags;
      pendingRegionInfos.push_back(info);
    });
  }

  // Add a signal manager for SIGUSR1 so that we can force
  // an event from the outside, making sure that the event loop can
  // be unblocked (e.g. by a quitting DPL driver) even when there
  // is no data pending to be processed.
  uv_signal_t* sigusr1Handle = (uv_signal_t*)malloc(sizeof(uv_signal_t));
  uv_signal_init(mState.loop, sigusr1Handle);
  sigusr1Handle->data = &mState;
  uv_signal_start(sigusr1Handle, on_signal_callback, SIGUSR1);

  // We add a timer only in case a channel poller is not there.
  if ((mStatefulProcess != nullptr) || (mStatelessProcess != nullptr)) {
    for (auto& x : fChannels) {
      if ((x.first.rfind("from_internal-dpl", 0) == 0) && (x.first.rfind("from_internal-dpl-aod", 0) != 0) && (x.first.rfind("from_internal-dpl-injected", 0)) != 0) {
        LOG(debug) << x.first << " is an internal channel. Skipping as no input will come from there." << std::endl;
        continue;
      }
      // We only watch receiving sockets.
      if (x.first.rfind("from_" + mSpec.name + "_", 0) == 0) {
        LOG(debug) << x.first << " is to send data. Not polling." << std::endl;
        continue;
      }
      // We assume there is always a ZeroMQ socket behind.
      int zmq_fd = 0;
      size_t zmq_fd_len = sizeof(zmq_fd);
      // FIXME: I should probably save those somewhere... ;-)
      uv_poll_t* poller = (uv_poll_t*)malloc(sizeof(uv_poll_t));
      x.second[0].GetSocket().GetOption("fd", &zmq_fd, &zmq_fd_len);
      if (zmq_fd == 0) {
        LOG(error) << "Cannot get file descriptor for channel." << x.first;
        continue;
      }
      LOG(debug) << "Polling socket for " << x.second[0].GetName();
      // FIXME: leak
      PollerContext* pCtx = (PollerContext*)malloc(sizeof(PollerContext));
      pCtx->name = strdup(x.first.c_str());
      pCtx->loop = mState.loop;
      pCtx->device = this;
      pCtx->state = &mState;
      pCtx->fd = zmq_fd;
      poller->data = pCtx;
      uv_poll_init(mState.loop, poller, zmq_fd);
      uv_poll_start(poller, UV_READABLE | UV_DISCONNECT, &on_socket_polled);
      mState.activeInputPollers.push_back(poller);
    }
    // In case we do not have any input channel and we do not have
    // any timers or signal watchers we still wake up whenever we can send data to downstream
    // devices to allow for enumerations.
    if (mState.activeInputPollers.empty() && mState.activeTimers.empty() && mState.activeSignals.empty()) {
      for (auto& x : fChannels) {
        if (x.first.rfind(mSpec.channelPrefix + "from_internal-dpl", 0) == 0) {
          LOG(debug) << x.first << " is an internal channel. Not polling." << std::endl;
          continue;
        }
        assert(x.first.rfind(mSpec.channelPrefix + "from_" + mSpec.name + "_", 0) == 0);
        // We assume there is always a ZeroMQ socket behind.
        int zmq_fd = 0;
        size_t zmq_fd_len = sizeof(zmq_fd);
        // FIXME: I should probably save those somewhere... ;-)
        uv_poll_t* poller = (uv_poll_t*)malloc(sizeof(uv_poll_t));
        x.second[0].GetSocket().GetOption("fd", &zmq_fd, &zmq_fd_len);
        if (zmq_fd == 0) {
          LOG(error) << "Cannot get file descriptor for channel." << x.first;
          continue;
        }
        LOG(debug) << "Polling socket for " << x.second[0].GetName();
        // FIXME: leak
        PollerContext* pCtx = (PollerContext*)malloc(sizeof(PollerContext));
        pCtx->name = strdup(x.first.c_str());
        pCtx->loop = mState.loop;
        pCtx->device = this;
        pCtx->state = &mState;
        pCtx->fd = zmq_fd;
        poller->data = pCtx;
        uv_poll_init(mState.loop, poller, zmq_fd);
        uv_poll_start(poller, UV_WRITABLE, &on_socket_polled);
        mState.activeOutputPollers.push_back(poller);
      }
    }
  } else {
    // This is a fake device, so we can request to exit immediately
    mServiceRegistry.get<ControlService>().readyToQuit(QuitRequest::Me);
    // A two second timer to stop internal devices which do not want to
    uv_timer_t* timer = (uv_timer_t*)malloc(sizeof(uv_timer_t));
    uv_timer_init(mState.loop, timer);
    timer->data = &mState;
    uv_timer_start(timer, on_idle_timer, 2000, 2000);
    mState.activeTimers.push_back(timer);
  }

  // Whenever we InitTask, we consider as if the previous iteration
  // was successful, so that even if there is no timer or receiving
  // channel, we can still start an enumeration.
  mWasActive = true;

  // We should be ready to run here. Therefore we copy all the
  // required parts in the DataProcessorContext. Eventually we should
  // do so on a per thread basis, with fine grained locks.
  mDataProcessorContexes.resize(1);
  this->fillContext(mDataProcessorContexes.at(0), mDeviceContext);
}

void DataProcessingDevice::fillContext(DataProcessorContext& context, DeviceContext& deviceContext)
{
  context.wasActive = &mWasActive;

  deviceContext.device = this;
  deviceContext.spec = &mSpec;
  deviceContext.state = &mState;
  deviceContext.quotaEvaluator = &mQuotaEvaluator;

  context.relayer = mRelayer;
  context.registry = &mServiceRegistry;
  context.completed = &mCompleted;
  context.expirationHandlers = &mExpirationHandlers;
  context.timingInfo = &mTimingInfo;
  context.allocator = &mAllocator;
  context.statefulProcess = &mStatefulProcess;
  context.statelessProcess = &mStatelessProcess;
  context.error = &mError;
  context.deviceContext = &deviceContext;
  /// Callback for the error handling
  context.errorHandling = &mErrorHandling;
}

void DataProcessingDevice::PreRun()
{
  mServiceRegistry.preStartCallbacks();
  mServiceRegistry.get<CallbackService>()(CallbackService::Id::Start);
}

void DataProcessingDevice::PostRun()
{
  mServiceRegistry.get<CallbackService>()(CallbackService::Id::Stop);
  mServiceRegistry.preExitCallbacks();
}

void DataProcessingDevice::Reset() { mServiceRegistry.get<CallbackService>()(CallbackService::Id::Reset); }

bool DataProcessingDevice::ConditionalRun()
{
  // This will block for the correct delay (or until we get data
  // on a socket). We also do not block on the first iteration
  // so that devices which do not have a timer can still start an
  // enumeration.
  if (mState.loop) {
    ZoneScopedN("uv idle");
    TracyPlot("past activity", (int64_t)mWasActive);
    mServiceRegistry.get<DriverClient>().flushPending();
    auto shouldNotWait = (mWasActive &&
                          (mState.streaming != StreamingState::Idle) && (mState.activeSignals.empty())) ||
                         (mState.streaming == StreamingState::EndOfStreaming);
    if (NewStatePending()) {
      shouldNotWait = true;
    }
    TracyPlot("shouldNotWait", (int)shouldNotWait);
    uv_run(mState.loop, shouldNotWait ? UV_RUN_NOWAIT : UV_RUN_ONCE);
    TracyPlot("loopReason", (int64_t)(uint64_t)mState.loopReason);

    mState.loopReason = DeviceState::NO_REASON;

    // A new state was requested, we exit.
    if (NewStatePending()) {
      return false;
    }
  }

  // Notify on the main thread the new region callbacks, making sure
  // no callback is issued if there is something still processing.
  {
    std::lock_guard<std::mutex> lock(mRegionInfoMutex);
    if (mPendingRegionInfos.empty() == false) {
      std::vector<FairMQRegionInfo> toBeNotified;
      toBeNotified.swap(mPendingRegionInfos); // avoid any MT issue.
      for (auto const& info : toBeNotified) {
        mServiceRegistry.get<CallbackService>()(CallbackService::Id::RegionInfoCallback, info);
      }
    }
  }

  assert(mStreams.size() == mHandles.size());
  /// Decide which task to use
  TaskStreamRef streamRef{-1};
  for (int ti = 0; ti < mStreams.size(); ti++) {
    auto& taskInfo = mStreams[ti];
    if (taskInfo.running) {
      continue;
    }
    streamRef.index = ti;
  }
  // We have an empty stream, let's check if we have enough
  // resources for it to run something
  if (streamRef.index != -1) {
    // Synchronous execution of the callbacks. This will be moved in the
    // moved in the on_socket_polled once we have threading in place.
    auto& handle = mHandles[streamRef.index];
    auto& stream = mStreams[streamRef.index];
    handle.data = &mStreams[streamRef.index];

    // Deciding wether to run or not can be done by passing a request to
    // the evaluator. In this case, the request is always satisfied and
    // we run on whatever resource is available.
    ComputingQuotaOfferRef offer = mQuotaEvaluator.selectOffer(mSpec.resourcePolicy.request);

    if (offer.index != -1) {
      stream.offer = offer;
      stream.running = true;
      stream.context = &mDataProcessorContexes.at(0);
      run_callback(&handle);
      run_completion(&handle, 0);
    } else {
      mWasActive = false;
    }
  } else {
    mWasActive = false;
  }
  FrameMark;
  return true;
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
    return info.state != InputChannelState::Pull;
  });
  // Whether or not all the channels are completed
  for (size_t ci = 0; ci < context.deviceContext->spec->inputChannels.size(); ++ci) {
    auto& channel = context.deviceContext->spec->inputChannels[ci];
    auto& info = context.deviceContext->state->inputChannelInfos[ci];

    if (info.state != InputChannelState::Completed && info.state != InputChannelState::Pull) {
      context.allDone = false;
    }
    if (info.state != InputChannelState::Running) {
      continue;
    }
    int64_t result = -2;
    if (info.channel == nullptr) {
      info.channel = &context.deviceContext->device->GetChannel(channel.name, 0);
    }
    auto& socket = info.channel->GetSocket();
    // If we have pending events from a previous iteration,
    // we do receive in any case.
    // Otherwise we check if there is any pending event and skip
    // this channel in case there is none.
    if (info.hasPendingEvents == 0) {
      socket.Events(&info.hasPendingEvents);
      // If we do not read, we can continue.
      if ((info.hasPendingEvents & 1) == 0) {
        continue;
      }
    }
    // Notice that there seems to be a difference between the documentation
    // of zeromq and the observed behavior. The fact that ZMQ_POLLIN
    // is raised does not mean that a message is immediately available to
    // read, just that it will be available soon, so the receive can
    // still return -2. To avoid this we keep receiving on the socket until
    // we get a message. In order not to overflow the DPL queue we process
    // one message at the time and we keep track of wether there were more
    // to process.
    while (true) {
      FairMQParts parts;
      result = info.channel->Receive(parts, 0);
      if (result >= 0) {
        DataProcessingDevice::handleData(context, parts, info);
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
      *context.wasActive |= true;
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
    // We keep processing data until we are Idle.
    // FIXME: not sure this is the correct way to drain the queues, but
    // I guess we will see.
    while (DataProcessingDevice::tryDispatchComputation(context, *context.completed)) {
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
    *context.wasActive = true;
    return;
  }

  return;
}

void DataProcessingDevice::ResetTask()
{
  mRelayer->clear();
}

/// This is the inner loop of our framework. The actual implementation
/// is divided in two parts. In the first one we define a set of lambdas
/// which describe what is actually going to happen, hiding all the state
/// boilerplate which the user does not need to care about at top level.
void DataProcessingDevice::handleData(DataProcessorContext& context, FairMQParts& parts, InputChannelInfo& info)
{
  ZoneScopedN("DataProcessingDevice::handleData");
  assert(context.deviceContext->spec->inputChannels.empty() == false);
  assert(parts.Size() > 0);

  // Initial part. Let's hide all the unnecessary and have
  // simple lambdas for each of the steps I am planning to have.
  assert(!context.deviceContext->spec->inputs.empty());

  enum struct InputType {
    Invalid,
    Data,
    SourceInfo
  };

  // This is how we validate inputs. I.e. we try to enforce the O2 Data model
  // and we do a few stats. We bind parts as a lambda captured variable, rather
  // than an input, because we do not want the outer loop actually be exposed
  // to the implementation details of the messaging layer.
  auto getInputTypes = [&stats = context.registry->get<DataProcessingStats>(),
                        &parts, &info, &context]() -> std::optional<std::vector<InputType>> {
    stats.inputParts = parts.Size();

    TracyPlot("messages received", (int64_t)parts.Size());
    if (parts.Size() % 2) {
      return std::nullopt;
    }
    std::vector<InputType> results(parts.Size() / 2, InputType::Invalid);

    for (size_t hi = 0; hi < parts.Size() / 2; ++hi) {
      auto pi = hi * 2;
      auto sih = o2::header::get<SourceInfoHeader*>(parts.At(pi)->GetData());
      if (sih) {
        info.state = sih->state;
        results[hi] = InputType::SourceInfo;
        *context.wasActive = true;
        continue;
      }
      auto dh = o2::header::get<DataHeader*>(parts.At(pi)->GetData());
      if (!dh) {
        results[hi] = InputType::Invalid;
        LOGP(error, "Header is not a DataHeader?");
        continue;
      }
      if (dh->payloadSize != parts.At(pi + 1)->GetSize()) {
        results[hi] = InputType::Invalid;
        LOGP(error, "DataHeader payloadSize mismatch");
        continue;
      }
      TracyPlot("payload size", (int64_t)dh->payloadSize);
      auto dph = o2::header::get<DataProcessingHeader*>(parts.At(pi)->GetData());
      TracyAlloc(parts.At(pi + 1)->GetData(), parts.At(pi + 1)->GetSize());
      if (!dph) {
        results[hi] = InputType::Invalid;
        LOGP(error, "Header stack does not contain DataProcessingHeader");
        continue;
      }
      results[hi] = InputType::Data;
    }
    return results;
  };

  auto reportError = [&registry = *context.registry, &context](const char* message) {
    registry.get<DataProcessingStats>().errorCount++;
  };

  auto handleValidMessages = [&parts, &context = context, &relayer = *context.relayer, &reportError](std::vector<InputType> const& types) {
    // We relay execution to make sure we have a complete set of parts
    // available.
    for (size_t pi = 0; pi < (parts.Size() / 2); ++pi) {
      switch (types[pi]) {
        case InputType::Data: {
          auto headerIndex = 2 * pi;
          auto payloadIndex = 2 * pi + 1;
          assert(payloadIndex < parts.Size());
          auto relayed = relayer.relay(std::move(parts.At(headerIndex)),
                                       std::move(parts.At(payloadIndex)));
          if (relayed == DataRelayer::WillNotRelay) {
            reportError("Unable to relay part.");
          }
        } break;
        case InputType::SourceInfo: {
          *context.wasActive = true;

        } break;
        case InputType::Invalid: {
          reportError("Invalid part found.");
        } break;
      }
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
  // This is the actual hidden state for the outer loop. In case we decide we
  // want to support multithreaded dispatching of operations, I can simply
  // move these to some thread local store and the rest of the lambdas
  // should work just fine.
  std::vector<MessageSet> currentSetOfInputs;

  auto reportError = [&registry = *context.registry, &context](const char* message) {
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
  auto getReadyActions = [& relayer = context.relayer,
                          &completed,
                          &stats = context.registry->get<DataProcessingStats>()]() -> std::vector<DataRelayer::RecordAction> {
    stats.pendingInputs = (int)relayer->getParallelTimeslices() - completed.size();
    stats.incomplete = completed.empty() ? 1 : 0;
    return completed;
  };

  // This is needed to convert from a pair of pointers to an actual DataRef
  // and to make sure the ownership is moved from the cache in the relayer to
  // the execution.
  auto fillInputs = [&relayer = context.relayer,
                     &spec = context.deviceContext->spec,
                     &currentSetOfInputs](TimesliceSlot slot) -> InputRecord {
    currentSetOfInputs = std::move(relayer->getInputsForTimeslice(slot));
    auto getter = [&currentSetOfInputs](size_t i, size_t partindex) -> DataRef {
      if (currentSetOfInputs[i].size() > partindex) {
        return DataRef{nullptr,
                       static_cast<char const*>(currentSetOfInputs[i].at(partindex).header->GetData()),
                       static_cast<char const*>(currentSetOfInputs[i].at(partindex).payload->GetData())};
      }
      return DataRef{nullptr, nullptr, nullptr};
    };
    auto nofPartsGetter = [&currentSetOfInputs](size_t i) -> size_t {
      return currentSetOfInputs[i].size();
    };
    InputSpan span{getter, nofPartsGetter, currentSetOfInputs.size()};
    return InputRecord{spec->inputs, std::move(span)};
  };

  auto markInputsAsDone = [&relayer = context.relayer](TimesliceSlot slot) -> void {
    relayer->updateCacheStatus(slot, CacheEntryStatus::RUNNING, CacheEntryStatus::DONE);
  };

  // I need a preparation step which gets the current timeslice id and
  // propagates it to the various contextes (i.e. the actual entities which
  // create messages) because the messages need to have the timeslice id into
  // it.
  auto prepareAllocatorForCurrentTimeSlice = [& timingInfo = context.timingInfo,
                                              &relayer = context.relayer](TimesliceSlot i) {
    ZoneScopedN("DataProcessingDevice::prepareForCurrentTimeslice");
    auto timeslice = relayer->getTimesliceForSlot(i);
    timingInfo->timeslice = timeslice.value;
    timingInfo->tfCounter = relayer->getFirstTFCounterForSlot(i);
    timingInfo->firstTFOrbit = relayer->getFirstTFOrbitForSlot(i);
  };

  // When processing them, timers will have to be cleaned up
  // to avoid double counting them.
  // This was actually the easiest solution we could find for
  // O2-646.
  auto cleanTimers = [&currentSetOfInputs](TimesliceSlot slot, InputRecord& record) {
    assert(record.size() == currentSetOfInputs.size());
    for (size_t ii = 0, ie = record.size(); ii < ie; ++ii) {
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

  // This is how we do the forwarding, i.e. we push
  // the inputs which are shared between this device and others
  // to the next one in the daisy chain.
  // FIXME: do it in a smarter way than O(N^2)
  auto forwardInputs = [&reportError,
                        &spec = context.deviceContext->spec,
                        &device = context.deviceContext->device, &currentSetOfInputs](TimesliceSlot slot, InputRecord& record) {
    ZoneScopedN("forward inputs");
    assert(record.size() == currentSetOfInputs.size());
    // we collect all messages per forward in a map and send them together
    std::unordered_map<std::string, FairMQParts> forwardedParts;
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

      auto dh = o2::header::get<DataHeader*>(input.header);
      if (!dh) {
        reportError("Header is not a DataHeader?");
        continue;
      }
      auto dph = o2::header::get<DataProcessingHeader*>(input.header);
      if (!dph) {
        reportError("Header stack does not contain DataProcessingHeader");
        continue;
      }

      for (auto& part : currentSetOfInputs[ii]) {
        for (auto const& forward : spec->forwards) {
          if (DataSpecUtils::match(forward.matcher, dh->dataOrigin, dh->dataDescription, dh->subSpecification) == false || (dph->startTime % forward.maxTimeslices) != forward.timeslice) {
            continue;
          }
          auto& header = part.header;
          auto& payload = part.payload;

          if (header.get() == nullptr) {
            // FIXME: this should not happen, however it's actually harmless and
            //        we can simply discard it for the moment.
            // LOG(ERROR) << "Missing header! " << dh->dataDescription;
            continue;
          }
          auto fdph = o2::header::get<DataProcessingHeader*>(header.get()->GetData());
          if (fdph == nullptr) {
            LOG(ERROR) << "Forwarded data does not have a DataProcessingHeader";
            continue;
          }
          auto fdh = o2::header::get<DataHeader*>(header.get()->GetData());
          if (fdh == nullptr) {
            LOG(ERROR) << "Forwarded data does not have a DataHeader";
            continue;
          }
     
          forwardedParts[forward.channel].AddPart(std::move(header));
          forwardedParts[forward.channel].AddPart(std::move(payload));
        }
      }
    }
    for (auto& [channelName, channelParts] : forwardedParts) {
      if (channelParts.Size() == 0) {
        continue;
      }
      assert(channelParts.Size() % 2 == 0);
      assert(o2::header::get<DataProcessingHeader*>(channelParts.At(0)->GetData()));
      // in DPL we are using subchannel 0 only
      device->Send(channelParts, channelName, 0);
    }
  };

  auto switchState = [&control = context.registry->get<ControlService>(),
                      &state = context.deviceContext->state](StreamingState newState) {
    state->streaming = newState;
    control.notifyStreamingState(state->streaming);
  };

  if (canDispatchSomeComputation() == false) {
    return false;
  }

  auto postUpdateStats = [& stats = context.registry->get<DataProcessingStats>()](DataRelayer::RecordAction const& action, InputRecord const& record, uint64_t tStart) {
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
    stats.lastTotalProcessedSize = calculateTotalInputRecordSize(record);
    stats.lastLatency = calculateInputRecordLatency(record, tStart);
  };

  auto preUpdateStats = [& stats = context.registry->get<DataProcessingStats>()](DataRelayer::RecordAction const& action, InputRecord const& record, uint64_t tStart) {
    std::atomic_thread_fence(std::memory_order_release);
    for (size_t ai = 0; ai != record.size(); ai++) {
      auto cacheId = action.slot.index * record.size() + ai;
      auto state = record.isValid(ai) ? 2 : 0;
      update_maximum(stats.statesSize, cacheId + 1);
      assert(cacheId < DataProcessingStats::MAX_RELAYER_STATES);
      stats.relayerState[cacheId].store(state);
    }
  };

  for (auto action : getReadyActions()) {
    if (action.op == CompletionPolicy::CompletionOp::Wait) {
      continue;
    }

    prepareAllocatorForCurrentTimeSlice(TimesliceSlot{action.slot});
    InputRecord record = fillInputs(action.slot);
    ProcessingContext processContext{record, *context.registry, *context.allocator};
    {
      ZoneScopedN("service pre processing");
      context.registry->preProcessingCallbacks(processContext);
    }
    if (action.op == CompletionPolicy::CompletionOp::Discard) {
      context.registry->postDispatchingCallbacks(processContext);
      if (context.deviceContext->spec->forwards.empty() == false) {
        forwardInputs(action.slot, record);
        continue;
      }
    }
    markInputsAsDone(action.slot);

    uint64_t tStart = uv_hrtime();
    preUpdateStats(action, record, tStart);
    try {
      if (context.deviceContext->state->quitRequested == false) {

        if (*context.statefulProcess) {
          ZoneScopedN("statefull process");
          (*context.statefulProcess)(processContext);
        }
        if (*context.statelessProcess) {
          ZoneScopedN("stateless process");
          (*context.statelessProcess)(processContext);
        }

        {
          ZoneScopedN("service post processing");
          context.registry->postProcessingCallbacks(processContext);
        }
      }
    } catch (std::exception& ex) {
      ZoneScopedN("error handling");
      /// Convert a standatd exception to a RuntimeErrorRef
      /// Notice how this will lose the backtrace information
      /// and report the exception coming from here.
      auto e = runtime_error(ex.what());
      (*context.errorHandling)(e, record);
    } catch (o2::framework::RuntimeErrorRef e) {
      ZoneScopedN("error handling");
      (*context.errorHandling)(e, record);
    }

    postUpdateStats(action, record, tStart);
    // We forward inputs only when we consume them. If we simply Process them,
    // we keep them for next message arriving.
    if (action.op == CompletionPolicy::CompletionOp::Consume) {
      context.registry->postDispatchingCallbacks(processContext);
      if (context.deviceContext->spec->forwards.empty() == false) {
        forwardInputs(action.slot, record);
      }
#ifdef TRACY_ENABLE
        cleanupRecord(record);
#endif
    } else if (action.op == CompletionPolicy::CompletionOp::Process) {
      cleanTimers(action.slot, record);
    }
  }
  // We now broadcast the end of stream if it was requested
  if (context.deviceContext->state->streaming == StreamingState::EndOfStreaming) {
    for (auto& channel : context.deviceContext->spec->outputChannels) {
      DataProcessingHelpers::sendEndOfStream(*context.deviceContext->device, channel);
    }
    switchState(StreamingState::Idle);
  }

  return true;
}

void DataProcessingDevice::error(const char* msg)
{
  LOG(ERROR) << msg;
  mServiceRegistry.get<DataProcessingStats>().errorCount++;
}

} // namespace o2::framework
