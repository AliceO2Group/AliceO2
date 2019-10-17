// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataProcessingDevice.h"
#include "Framework/ChannelMatching.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataProcessor.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DeviceState.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/FairOptionsRetriever.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Framework/CallbackService.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/InputRecord.h"
#include "Framework/Signpost.h"
#include "Framework/SourceInfoHeader.h"
#include "Framework/Logger.h"
#include "DataProcessingStatus.h"
#include "DataProcessingHelpers.h"

#include "ScopedExit.h"

#include <fairmq/FairMQParts.h>
#include <fairmq/FairMQSocket.h>
#include <options/FairMQProgOptions.h>
#include <Monitoring/Monitoring.h>
#include <TMessage.h>
#include <TClonesArray.h>

#include <vector>
#include <memory>

using namespace o2::framework;
using Key = o2::monitoring::tags::Key;
using Value = o2::monitoring::tags::Value;
using Metric = o2::monitoring::Metric;
using Monitoring = o2::monitoring::Monitoring;
using DataHeader = o2::header::DataHeader;

constexpr unsigned int MONITORING_QUEUE_SIZE = 100;
constexpr unsigned int MIN_RATE_LOGGING = 60;

namespace o2
{
namespace framework
{

DataProcessingDevice::DataProcessingDevice(DeviceSpec const& spec, ServiceRegistry& registry, DeviceState& state)
  : mSpec{spec},
    mState{state},
    mInit{spec.algorithm.onInit},
    mStatefulProcess{nullptr},
    mStatelessProcess{spec.algorithm.onProcess},
    mError{spec.algorithm.onError},
    mConfigRegistry{nullptr},
    mFairMQContext{FairMQDeviceProxy{this}},
    mRootContext{FairMQDeviceProxy{this}},
    mStringContext{FairMQDeviceProxy{this}},
    mDataFrameContext{FairMQDeviceProxy{this}},
    mRawBufferContext{FairMQDeviceProxy{this}},
    mContextRegistry{&mFairMQContext, &mRootContext, &mStringContext, &mDataFrameContext, &mRawBufferContext},
    mAllocator{&mTimingInfo, &mContextRegistry, spec.outputs},
    mRelayer{spec.completionPolicy, spec.inputs, spec.forwards, registry.get<Monitoring>(), registry.get<TimesliceIndex>()},
    mServiceRegistry{registry},
    mErrorCount{0},
    mProcessingCount{0}
{
  StateMonitoring<DataProcessingStatus>::start();
  auto dispatcher = [this](FairMQParts&& parts, std::string const& channel, unsigned int index) {
    DataProcessor::doSend(*this, std::move(parts), channel.c_str(), index);
  };

  if (spec.dispatchPolicy.action == DispatchPolicy::DispatchOp::WhenReady) {
    mFairMQContext.init(dispatcher);
  }
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
  // For some reason passing rateLogging does not work anymore.
  // This makes sure we never have more than one notification per minute.
  for (auto& x : fChannels) {
    for (auto& c : x.second) {
      if (c.GetRateLogging() < MIN_RATE_LOGGING) {
        c.UpdateRateLogging(MIN_RATE_LOGGING);
      }
    }
  }
  auto optionsRetriever(std::make_unique<FairOptionsRetriever>(mSpec.options, GetConfig()));
  mConfigRegistry = std::move(std::make_unique<ConfigParamRegistry>(std::move(optionsRetriever)));

  mExpirationHandlers.clear();
  for (auto& route : mSpec.inputs) {
    ExpirationHandler handler{
      route.matcher.lifetime,
      route.creatorConfigurator(*mConfigRegistry),
      route.danglingConfigurator(*mConfigRegistry),
      route.expirationConfigurator(*mConfigRegistry)};
    mExpirationHandlers.emplace_back(std::move(handler));
  }

  auto& monitoring = mServiceRegistry.get<Monitoring>();
  monitoring.enableBuffering(MONITORING_QUEUE_SIZE);
  static const std::string dataProcessorIdMetric = "dataprocessor_id";
  static const std::string dataProcessorIdValue = mSpec.name;
  monitoring.addGlobalTag("dataprocessor_id", dataProcessorIdValue);

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
    if (name.find("from_internal-dpl-clock") == 0) {
      mState.inputChannelInfos[ci].state = InputChannelState::Pull;
    } else if (name.find("from_internal-dpl-ccdb-backend") == 0) {
      mState.inputChannelInfos[ci].state = InputChannelState::Pull;
    }
  }
}

void DataProcessingDevice::PreRun() { mServiceRegistry.get<CallbackService>()(CallbackService::Id::Start); }

void DataProcessingDevice::PostRun() { mServiceRegistry.get<CallbackService>()(CallbackService::Id::Stop); }

void DataProcessingDevice::Reset() { mServiceRegistry.get<CallbackService>()(CallbackService::Id::Reset); }

/// We drive the state loop ourself so that we will be able to support
/// non-data triggers like those which are time based.
bool DataProcessingDevice::ConditionalRun()
{
  /// This will send metrics for the relayer at regular intervals of
  /// 5 seconds, in order to avoid overloading the system.
  auto sendRelayerMetrics = [relayerStats = mRelayer.getStats(),
                             &stats = mStats,
                             &lastSent = mLastSlowMetricSentTimestamp,
                             &currentTime = mBeginIterationTimestamp,
                             &monitoring = mServiceRegistry.get<Monitoring>()]()
    -> void {
    if (currentTime - lastSent < 5000) {
      return;
    }

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
    monitoring.send(Metric{(stats.lastTotalProcessedSize / (stats.lastElapsedTimeMs ? stats.lastElapsedTimeMs : 1) / 1000),
                           "processing_rate_mb_s"}
                      .addTag(Key::Subsystem, Value::DPL));
    monitoring.send(Metric{stats.lastLatency.minLatency, "min_input_latency_ms"}
                      .addTag(Key::Subsystem, Value::DPL));
    monitoring.send(Metric{stats.lastLatency.maxLatency, "max_input_latency_ms"}
                      .addTag(Key::Subsystem, Value::DPL));
    monitoring.send(Metric{(stats.lastTotalProcessedSize / (stats.lastLatency.maxLatency ? stats.lastLatency.maxLatency : 1) / 1000), "input_rate_mb_s"}
                      .addTag(Key::Subsystem, Value::DPL));

    lastSent = currentTime;
    O2_SIGNPOST_END(MonitoringStatus::ID, MonitoringStatus::SEND, 0, 0, O2_SIGNPOST_BLUE);
  };

  /// This will flush metrics only once every second.
  auto flushMetrics = [& stats = mStats,
                       &relayer = mRelayer,
                       &lastFlushed = mLastMetricFlushedTimestamp,
                       &currentTime = mBeginIterationTimestamp,
                       &monitoring = mServiceRegistry.get<Monitoring>()]()
    -> void {
    if (currentTime - lastFlushed < 1000) {
      return;
    }

    O2_SIGNPOST_START(MonitoringStatus::ID, MonitoringStatus::FLUSH, 0, 0, O2_SIGNPOST_RED);
    // Send all the relevant metrics for the relayer to update the GUI
    // FIXME: do a delta with the previous version if too many metrics are still
    // sent...
    for (size_t si = 0; si < stats.relayerState.size(); ++si) {
      auto state = stats.relayerState[si];
      monitoring.send({state, "data_relayer/" + std::to_string(si)});
    }
    relayer.sendContextState();
    monitoring.flushBuffer();
    lastFlushed = currentTime;
    O2_SIGNPOST_END(MonitoringStatus::ID, MonitoringStatus::FLUSH, 0, 0, O2_SIGNPOST_RED);
  };

  auto switchState = [& control = mServiceRegistry.get<ControlService>(),
                      &state = mState.streaming](StreamingState newState) {
    state = newState;
    control.notifyStreamingState(state);
  };

  auto now = std::chrono::high_resolution_clock::now();
  mBeginIterationTimestamp = (uint64_t)std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();

  mServiceRegistry.get<CallbackService>()(CallbackService::Id::ClockTick);
  // Wether or not we had something to do.
  bool active = false;
  // Notice that in case there are no input channels we should the allDone
  // state will have to be set explicitly in the code, like we do, for example
  // for implicit data sources or timers.  This also implies that whatever is
  // time driven will have to signal EndOfStream itself.
  bool allDone = mSpec.inputChannels.empty() == false;
  // Whether or not all the channels are completed
  for (size_t ci = 0; ci < mSpec.inputChannels.size(); ++ci) {
    auto& channel = mSpec.inputChannels[ci];
    auto& info = mState.inputChannelInfos[ci];
    if (info.state != InputChannelState::Completed) {
      allDone = false;
    }
    if (info.state != InputChannelState::Running) {
      continue;
    }
    FairMQParts parts;
    auto result = this->Receive(parts, channel.name, 0, 0);
    if (result > 0) {
      this->handleData(parts, info);
      active |= this->tryDispatchComputation();
    }
  }
  if (active == false) {
    mServiceRegistry.get<CallbackService>()(CallbackService::Id::Idle);
  }
  mRelayer.processDanglingInputs(mExpirationHandlers, mServiceRegistry);
  this->tryDispatchComputation();

  sendRelayerMetrics();
  flushMetrics();

  // If we got notified that all the sources are done, we call the EndOfStream
  // callback and return false. Notice that what happens next is actually
  // dependent on the callback, not something which is controlled by the
  // framework itself.
  if (allDone == true && mState.streaming == StreamingState::Streaming) {
    switchState(StreamingState::EndOfStreaming);
  }

  if (mState.streaming == StreamingState::EndOfStreaming) {
    // We keep processing data until we are Idle.
    // FIXME: not sure this is the correct way to drain the queues, but
    // I guess we will see.
    while (this->tryDispatchComputation()) {
      mRelayer.processDanglingInputs(mExpirationHandlers, mServiceRegistry);
    }
    sendRelayerMetrics();
    flushMetrics();
    mContextRegistry.get<MessageContext>()->clear();
    mContextRegistry.get<RootObjectContext>()->clear();
    mContextRegistry.get<StringContext>()->clear();
    mContextRegistry.get<ArrowContext>()->clear();
    mContextRegistry.get<RawBufferContext>()->clear();
    EndOfStreamContext eosContext{mServiceRegistry, mAllocator};
    mServiceRegistry.get<CallbackService>()(CallbackService::Id::EndOfStream, eosContext);
    DataProcessor::doSend(*this, *mContextRegistry.get<MessageContext>());
    DataProcessor::doSend(*this, *mContextRegistry.get<RootObjectContext>());
    DataProcessor::doSend(*this, *mContextRegistry.get<StringContext>());
    DataProcessor::doSend(*this, *mContextRegistry.get<ArrowContext>());
    DataProcessor::doSend(*this, *mContextRegistry.get<RawBufferContext>());
    for (auto& channel : mSpec.outputChannels) {
      DataProcessingHelpers::sendEndOfStream(*this, channel);
    }
    switchState(StreamingState::Idle);
    return true;
  }
  return true;
}

/// This is the inner loop of our framework. The actual implementation
/// is divided in two parts. In the first one we define a set of lambdas
/// which describe what is actually going to happen, hiding all the state
/// boilerplate which the user does not need to care about at top level.
bool DataProcessingDevice::handleData(FairMQParts& parts, InputChannelInfo& info)
{
  assert(mSpec.inputChannels.empty() == false);
  assert(parts.Size() > 0);

  // Initial part. Let's hide all the unnecessary and have
  // simple lambdas for each of the steps I am planning to have.
  assert(!mSpec.inputs.empty());

  // These duplicate references are created so that each function
  // does not need to know about the whole class state, but I can
  // fine grain control what is exposed at each state.
  auto& monitoringService = mServiceRegistry.get<Monitoring>();
  StateMonitoring<DataProcessingStatus>::moveTo(DataProcessingStatus::IN_DPL_OVERHEAD);
  ScopedExit metricFlusher([&monitoringService] {
    StateMonitoring<DataProcessingStatus>::moveTo(DataProcessingStatus::IN_DPL_OVERHEAD);
  });

  enum struct InputType {
    Invalid,
    Data,
    SourceInfo
  };

  // This is how we validate inputs. I.e. we try to enforce the O2 Data model
  // and we do a few stats. We bind parts as a lambda captured variable, rather
  // than an input, because we do not want the outer loop actually be exposed
  // to the implementation details of the messaging layer.
  auto getInputTypes = [& stats = mStats, &parts, &info]() -> std::optional<std::vector<InputType>> {
    stats.inputParts = parts.Size();

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
      auto dph = o2::header::get<DataProcessingHeader*>(parts.At(pi)->GetData());
      if (!dph) {
        results[hi] = InputType::Invalid;
        LOGP(error, "Header stack does not contain DataProcessingHeader");
        continue;
      }
      results[hi] = InputType::Data;
    }
    return results;
  };

  auto reportError = [& device = *this](const char* message) {
    device.error(message);
  };

  auto handleValidMessages = [&parts, &relayer = mRelayer, &reportError](std::vector<InputType> const& types) {
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
    return true;
  }
  handleValidMessages(*inputTypes);
  return true;
}

bool DataProcessingDevice::tryDispatchComputation()
{
  // This is the actual hidden state for the outer loop. In case we decide we
  // want to support multithreaded dispatching of operations, I can simply
  // move these to some thread local store and the rest of the lambdas
  // should work just fine.
  std::vector<DataRelayer::RecordAction> completed;
  std::vector<std::unique_ptr<FairMQMessage>> currentSetOfInputs;

  auto& allocator = mAllocator;
  auto& context = *mContextRegistry.get<MessageContext>();
  auto& device = *this;
  auto& errorCallback = mError;
  auto& errorCount = mErrorCount;
  auto& forwards = mSpec.forwards;
  auto& inputsSchema = mSpec.inputs;
  auto& processingCount = mProcessingCount;
  auto& rdfContext = *mContextRegistry.get<ArrowContext>();
  auto& relayer = mRelayer;
  auto& rootContext = *mContextRegistry.get<RootObjectContext>();
  auto& serviceRegistry = mServiceRegistry;
  auto& statefulProcess = mStatefulProcess;
  auto& statelessProcess = mStatelessProcess;
  auto& stringContext = *mContextRegistry.get<StringContext>();
  auto& timingInfo = mTimingInfo;
  auto& timesliceIndex = mServiceRegistry.get<TimesliceIndex>();
  auto& rawContext = *mContextRegistry.get<RawBufferContext>();

  // These duplicate references are created so that each function
  // does not need to know about the whole class state, but I can
  // fine grain control what is exposed at each state.
  // FIXME: I should use a different id for this state.
  auto& monitoringService = mServiceRegistry.get<Monitoring>();
  StateMonitoring<DataProcessingStatus>::moveTo(DataProcessingStatus::IN_DPL_OVERHEAD);
  ScopedExit metricFlusher([&monitoringService] {
    StateMonitoring<DataProcessingStatus>::moveTo(DataProcessingStatus::IN_DPL_OVERHEAD);
  });

  auto reportError = [&device](const char* message) {
    device.error(message);
  };

  // For the moment we have a simple "immediately dispatch" policy for stuff
  // in the cache. This could be controlled from the outside e.g. by waiting
  // for a few sets of inputs to arrive before we actually dispatch the
  // computation, however this can be defined at a later stage.
  auto canDispatchSomeComputation = [&completed, &relayer]() -> bool {
    completed = relayer.getReadyToProcess();
    return completed.empty() == false;
  };

  // We use this to get a list with the actual indexes in the cache which
  // indicate a complete set of inputs. Notice how I fill the completed
  // vector and return it, so that I can have a nice for loop iteration later
  // on.
  auto getReadyActions = [&relayer, &completed, &stats = mStats]() -> std::vector<DataRelayer::RecordAction> {
    stats.pendingInputs = (int)relayer.getParallelTimeslices() - completed.size();
    stats.incomplete = completed.empty() ? 1 : 0;
    return completed;
  };

  // This is needed to convert from a pair of pointers to an actual DataRef
  // and to make sure the ownership is moved from the cache in the relayer to
  // the execution.
  auto fillInputs = [&relayer, &inputsSchema, &currentSetOfInputs](TimesliceSlot slot) -> InputRecord {
    currentSetOfInputs = std::move(relayer.getInputsForTimeslice(slot));
    InputSpan span{[&currentSetOfInputs](size_t i) -> char const* {
                     return currentSetOfInputs.at(i) ? static_cast<char const*>(currentSetOfInputs.at(i)->GetData()) : nullptr;
                   },
                   currentSetOfInputs.size()};
    return InputRecord{inputsSchema, std::move(span)};
  };

  // This is the thing which does the actual computation. No particular reason
  // why we do the stateful processing before the stateless one.
  // PROCESSING:{START,END} is done so that we can trigger on begin / end of processing
  // in the GUI.
  auto dispatchProcessing = [&processingCount, &allocator, &statefulProcess, &statelessProcess, &monitoringService,
                             &context, &rootContext, &stringContext, &rdfContext, &rawContext, &serviceRegistry, &device](TimesliceSlot slot, InputRecord& record) {
    if (statefulProcess) {
      ProcessingContext processContext{record, serviceRegistry, allocator};
      StateMonitoring<DataProcessingStatus>::moveTo(DataProcessingStatus::IN_DPL_USER_CALLBACK);
      statefulProcess(processContext);
      StateMonitoring<DataProcessingStatus>::moveTo(DataProcessingStatus::IN_DPL_OVERHEAD);
      processingCount++;
    }
    if (statelessProcess) {
      ProcessingContext processContext{record, serviceRegistry, allocator};
      StateMonitoring<DataProcessingStatus>::moveTo(DataProcessingStatus::IN_DPL_USER_CALLBACK);
      statelessProcess(processContext);
      StateMonitoring<DataProcessingStatus>::moveTo(DataProcessingStatus::IN_DPL_OVERHEAD);
      processingCount++;
    }

    DataProcessor::doSend(device, context);
    DataProcessor::doSend(device, rootContext);
    DataProcessor::doSend(device, stringContext);
    DataProcessor::doSend(device, rdfContext);
    DataProcessor::doSend(device, rawContext);
  };

  // Error handling means printing the error and updating the metric
  auto errorHandling = [&errorCallback, &monitoringService, &serviceRegistry](std::exception& e, InputRecord& record) {
    StateMonitoring<DataProcessingStatus>::moveTo(DataProcessingStatus::IN_DPL_ERROR_CALLBACK);
    LOG(ERROR) << "Exception caught: " << e.what() << std::endl;
    if (errorCallback) {
      monitoringService.send({1, "error"});
      ErrorContext errorContext{record, serviceRegistry, e};
      errorCallback(errorContext);
    }
    StateMonitoring<DataProcessingStatus>::moveTo(DataProcessingStatus::IN_DPL_OVERHEAD);
  };

  // I need a preparation step which gets the current timeslice id and
  // propagates it to the various contextes (i.e. the actual entities which
  // create messages) because the messages need to have the timeslice id into
  // it.
  auto prepareAllocatorForCurrentTimeSlice = [&timingInfo, &rootContext, &stringContext, &rdfContext, &rawContext, &context, &relayer, &timesliceIndex](TimesliceSlot i) {
    auto timeslice = timesliceIndex.getTimesliceForSlot(i);
    timingInfo.timeslice = timeslice.value;
    rootContext.clear();
    context.clear();
    stringContext.clear();
    rdfContext.clear();
    rawContext.clear();
  };

  // When processing them, timers will have to be cleaned up
  // to avoid double counting them.
  // This was actually the easiest solution we could find for
  // O2-646.
  auto cleanTimers = [&currentSetOfInputs](TimesliceSlot slot, InputRecord& record) {
    assert(record.size() * 2 == currentSetOfInputs.size());
    for (size_t ii = 0, ie = record.size(); ii < ie; ++ii) {
      DataRef input = record.getByPos(ii);
      if (input.spec->lifetime != Lifetime::Timer) {
        continue;
      }
      if (input.header == nullptr || input.payload == nullptr) {
        continue;
      }
      // This will hopefully delete the message.
      std::unique_ptr<FairMQMessage> header = std::move(currentSetOfInputs[ii * 2]);
      std::unique_ptr<FairMQMessage> payload = std::move(currentSetOfInputs[ii * 2 + 1]);
    }
  };

  // This is how we do the forwarding, i.e. we push
  // the inputs which are shared between this device and others
  // to the next one in the daisy chain.
  // FIXME: do it in a smarter way than O(N^2)
  auto forwardInputs = [&reportError, &forwards, &device, &currentSetOfInputs](TimesliceSlot slot, InputRecord& record) {
    assert(record.size() * 2 == currentSetOfInputs.size());
    for (size_t ii = 0, ie = record.size(); ii < ie; ++ii) {
      DataRef input = record.getByPos(ii);

      // If is now possible that the record is not complete when
      // we forward it, because of a custom completion policy.
      // this means that we need to skip the empty entries in the
      // record for being forwarded.
      if (input.header == nullptr || input.payload == nullptr) {
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

      auto& header = currentSetOfInputs[ii * 2];
      auto& payload = currentSetOfInputs[ii * 2 + 1];

      for (auto forward : forwards) {
        if (DataSpecUtils::match(forward.matcher, dh->dataOrigin, dh->dataDescription, dh->subSpecification) && (dph->startTime % forward.maxTimeslices) == forward.timeslice) {

          if (header.get() == nullptr) {
            LOG(ERROR) << "Missing header!";
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
          FairMQParts forwardedParts;
          forwardedParts.AddPart(std::move(header));
          forwardedParts.AddPart(std::move(payload));
          assert(forwardedParts.Size() == 2);
          assert(o2::header::get<DataProcessingHeader*>(forwardedParts.At(0)->GetData()));
          // FIXME: this should use a correct subchannel
          device.Send(forwardedParts, forward.channel, 0);
        }
      }
    }
  };

  auto calculateInputRecordLatency = [](InputRecord const& record, auto now) -> DataProcessingStats::InputLatency {
    DataProcessingStats::InputLatency result{static_cast<int>(-1), 0};

    auto currentTime = (uint64_t)std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
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

  auto calculateTotalInputRecordSize = [](InputRecord const& record) -> int {
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

  auto switchState = [& control = mServiceRegistry.get<ControlService>(),
                      &state = mState.streaming](StreamingState newState) {
    state = newState;
    control.notifyStreamingState(state);
  };

  if (canDispatchSomeComputation() == false) {
    return false;
  }

  for (auto action : getReadyActions()) {
    if (action.op == CompletionPolicy::CompletionOp::Wait) {
      continue;
    }

    prepareAllocatorForCurrentTimeSlice(TimesliceSlot{action.slot});
    InputRecord record = fillInputs(action.slot);
    if (action.op == CompletionPolicy::CompletionOp::Discard) {
      if (forwards.empty() == false) {
        forwardInputs(action.slot, record);
        continue;
      }
    }
    auto tStart = std::chrono::high_resolution_clock::now();
    for (size_t ai = 0; ai != record.size(); ai++) {
      auto cacheId = action.slot.index * record.size() + ai;
      auto state = record.isValid(ai) ? 2 : 0;
      mStats.relayerState.resize(std::max(cacheId + 1, mStats.relayerState.size()), 0);
      mStats.relayerState[cacheId] = state;
    }
    try {
      if (mState.quitRequested == false) {
        dispatchProcessing(action.slot, record);
      }
    } catch (std::exception& e) {
      errorHandling(e, record);
    }
    for (size_t ai = 0; ai != record.size(); ai++) {
      auto cacheId = action.slot.index * record.size() + ai;
      auto state = record.isValid(ai) ? 3 : 0;
      mStats.relayerState.resize(std::max(cacheId + 1, mStats.relayerState.size()), 0);
      mStats.relayerState[cacheId] = state;
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    mStats.lastElapsedTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    mStats.lastTotalProcessedSize = calculateTotalInputRecordSize(record);
    mStats.lastLatency = calculateInputRecordLatency(record, tStart);
    // We forward inputs only when we consume them. If we simply Process them,
    // we keep them for next message arriving.
    if (action.op == CompletionPolicy::CompletionOp::Consume) {
      if (forwards.empty() == false) {
        forwardInputs(action.slot, record);
      }
    } else if (action.op == CompletionPolicy::CompletionOp::Process) {
      cleanTimers(action.slot, record);
    }
  }
  // We now broadcast the end of stream if it was requested
  if (mState.streaming == StreamingState::EndOfStreaming) {
    for (auto& channel : mSpec.outputChannels) {
      DataProcessingHelpers::sendEndOfStream(*this, channel);
    }
    switchState(StreamingState::Idle);
  }

  return true;
}

void DataProcessingDevice::error(const char* msg)
{
  LOG(ERROR) << msg;
  mErrorCount++;
  mServiceRegistry.get<Monitoring>().send(Metric{mErrorCount, "errors"}.addTag(Key::Subsystem, Value::DPL));
}

} // namespace framework
} // namespace o2
