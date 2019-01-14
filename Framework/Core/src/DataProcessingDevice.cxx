// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "DataProcessingStatus.h"
#include "Framework/DataProcessingDevice.h"
#include "Framework/ChannelMatching.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataProcessor.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/FairOptionsRetriever.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Framework/CallbackService.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/InputRecord.h"
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
using Monitoring = o2::monitoring::Monitoring;
using DataHeader = o2::header::DataHeader;

constexpr unsigned int MONITORING_QUEUE_SIZE = 100;
constexpr unsigned int MIN_RATE_LOGGING = 60;

namespace o2
{
namespace framework
{

/// Handle the fact that FairMQ deprecated ReceiveAsync and changed the behavior of receive.
namespace
{
struct FairMQDeviceLegacyWrapper {
  // If both APIs are available, this has precendence because of dummy.
  // Only the old API before the deprecation had FairMQSocket::TrySend().
  template <typename S>
  static auto ReceiveAsync(FairMQDevice* device, FairMQParts& parts, std::string const& channel, int dummy) -> typename std::enable_if<(sizeof(decltype(std::declval<S>().TrySend(parts.At(0)))) > 0), int>::type
  {
    return device->ReceiveAsync(parts, channel);
  }

  // Otherwise if we are here it means that TrySend() is not there anymore
  template <typename S>
  static auto ReceiveAsync(FairMQDevice* device, FairMQParts& parts, std::string const& channel, long dummy) -> int
  {
    return device->Receive(parts, channel, 0, 0);
  }
};
}

DataProcessingDevice::DataProcessingDevice(DeviceSpec const& spec, ServiceRegistry& registry)
  : mSpec{ spec },
    mInit{ spec.algorithm.onInit },
    mStatefulProcess{ nullptr },
    mStatelessProcess{ spec.algorithm.onProcess },
    mError{ spec.algorithm.onError },
    mConfigRegistry{ nullptr },
    mFairMQContext{ FairMQDeviceProxy{ this } },
    mRootContext{ FairMQDeviceProxy{ this } },
    mStringContext{ FairMQDeviceProxy{ this } },
    mDataFrameContext{ FairMQDeviceProxy{ this } },
    mRawBufferContext{ FairMQDeviceProxy{ this } },
    mContextRegistry{ { &mFairMQContext, &mRootContext, &mStringContext, &mDataFrameContext, &mRawBufferContext } },
    mAllocator{ &mTimingInfo, &mContextRegistry, spec.outputs },
    mRelayer{ spec.completionPolicy, spec.inputs, spec.forwards, registry.get<Monitoring>(), registry.get<TimesliceIndex>() },
    mServiceRegistry{ registry },
    mErrorCount{ 0 },
    mProcessingCount{ 0 }
{
}

/// This  takes care  of initialising  the device  from its  specification. In
/// particular it needs to:
///
/// * Fetch parameters from configuration
/// * Materialize the correct callbacks for expiring records. We need to do it
///   here because the configuration is available only at this point.
/// * Invoke the actual init callback, which returns the processing callback.
void DataProcessingDevice::Init() {
  LOG(DEBUG) << "DataProcessingDevice::InitTask::START";
  // For some reason passing rateLogging does not work anymore. 
  // This makes sure we never have more than one notification per minute.
  for (auto& x : fChannels) {
    for (auto& c : x.second) {
      if (c.GetRateLogging() < MIN_RATE_LOGGING) {
        c.UpdateRateLogging(MIN_RATE_LOGGING);
      }
    }
  }
  auto optionsRetriever(std::make_unique<FairOptionsRetriever>(GetConfig()));
  mConfigRegistry = std::move(std::make_unique<ConfigParamRegistry>(std::move(optionsRetriever)));

  mExpirationHandlers.clear();
  for (auto& route : mSpec.inputs) {
    ExpirationHandler handler{
      route.danglingConfigurator(*mConfigRegistry),
      route.expirationConfigurator(*mConfigRegistry)
    };
    mExpirationHandlers.emplace_back(std::move(handler));
  }

  auto& monitoring = mServiceRegistry.get<Monitoring>();
  monitoring.enableBuffering(MONITORING_QUEUE_SIZE);

  if (mInit) {
    InitContext initContext{*mConfigRegistry,mServiceRegistry};
    mStatefulProcess = mInit(initContext);
  }
  LOG(DEBUG) << "DataProcessingDevice::InitTask::END";
}

void DataProcessingDevice::PreRun() { mServiceRegistry.get<CallbackService>()(CallbackService::Id::Start); }

void DataProcessingDevice::PostRun() { mServiceRegistry.get<CallbackService>()(CallbackService::Id::Stop); }

void DataProcessingDevice::Reset() { mServiceRegistry.get<CallbackService>()(CallbackService::Id::Reset); }

/// We drive the state loop ourself so that we will be able to support
/// non-data triggers like those which are time based.
bool DataProcessingDevice::ConditionalRun()
{
  mServiceRegistry.get<CallbackService>()(CallbackService::Id::ClockTick);
  bool active = false;
  for (auto& channel : mSpec.inputChannels) {
    FairMQParts parts;
    auto result = FairMQDeviceLegacyWrapper::ReceiveAsync<FairMQSocket>(this, parts, channel.name, 0);
    if (result > 0) {
      this->handleData(parts);
      active |= this->tryDispatchComputation();
    }
  }
  if (active == false) {
    mServiceRegistry.get<CallbackService>()(CallbackService::Id::Idle);
  }
  mRelayer.processDanglingInputs(mExpirationHandlers, mServiceRegistry);
  this->tryDispatchComputation();
  return true;
}

/// This is the inner loop of our framework. The actual implementation
/// is divided in two parts. In the first one we define a set of lambdas
/// which describe what is actually going to happen, hiding all the state
/// boilerplate which the user does not need to care about at top level.
bool DataProcessingDevice::handleData(FairMQParts& parts)
{
  assert(mSpec.inputChannels.empty() == false);
  assert(parts.Size() > 0);

  static const std::string handleDataMetricName = "dpl/in_handle_data";
  // Initial part. Let's hide all the unnecessary and have
  // simple lambdas for each of the steps I am planning to have.
  assert(!mSpec.inputs.empty());

  // These duplicate references are created so that each function
  // does not need to know about the whole class state, but I can
  // fine grain control what is exposed at each state.
  auto& monitoringService = mServiceRegistry.get<Monitoring>();
  monitoringService.send({ DataProcessingStatus::IN_DPL_WRAPPER, "dpl/in_handle_data" });
  ScopedExit metricFlusher([&monitoringService] {
      monitoringService.send({ DataProcessingStatus::IN_DPL_WRAPPER, "dpl/in_handle_data"});
      monitoringService.send({ DataProcessingStatus::IN_FAIRMQ, "dpl/in_handle_data"});
      monitoringService.flushBuffer(); });

  auto& device = *this;
  auto& errorCount = mErrorCount;
  auto& relayer = mRelayer;
  auto& serviceRegistry = mServiceRegistry;

  // This is how we validate inputs. I.e. we try to enforce the O2 Data model
  // and we do a few stats. We bind parts as a lambda captured variable, rather
  // than an input, because we do not want the outer loop actually be exposed
  // to the implementation details of the messaging layer.
  auto isValidInput = [&monitoringService, &parts]() -> bool {
    // monitoringService.send({ (int)parts.Size(), "inputs/parts/total" });
    monitoringService.send({ (int)parts.Size(), "inputs/parts/total" });

    for (size_t i = 0; i < parts.Size() ; ++i) {
      LOG(DEBUG) << " part " << i << " is " << parts.At(i)->GetSize() << " bytes";
    }
    if (parts.Size() % 2) {
      return false;
    }
    for (size_t hi = 0; hi < parts.Size()/2; ++hi) {
      auto pi = hi*2;
      auto dh = o2::header::get<DataHeader*>(parts.At(pi)->GetData());
      if (!dh) {
        LOG(ERROR) << "Header is not a DataHeader?";
        return false;
      }
      if (dh->payloadSize != parts.At(pi+1)->GetSize()) {
        LOG(ERROR) << "DataHeader payloadSize mismatch";
        return false;
      }
      auto dph = o2::header::get<DataProcessingHeader*>(parts.At(pi)->GetData());
      if (!dph) {
        LOG(ERROR) << "Header stack does not contain DataProcessingHeader";
        return false;
      }
      LOG(DEBUG) << "Timeslice is " << dph->startTime;
      LOG(DEBUG) << " DataOrigin is " << dh->dataOrigin.str;
      LOG(DEBUG) << " DataDescription is " << dh->dataDescription.str;
    }
    return true;
  };

  auto reportError = [&device](const char* message) {
    device.error(message);
  };

  auto putIncomingMessageIntoCache = [&parts,&relayer,&reportError]() {
    // We relay execution to make sure we have a complete set of parts
    // available.
    for (size_t pi = 0; pi < (parts.Size()/2); ++pi) {
      auto headerIndex = 2*pi;
      auto payloadIndex = 2*pi+1;
      assert(payloadIndex < parts.Size());
      auto relayed = relayer.relay(std::move(parts.At(headerIndex)),
                                   std::move(parts.At(payloadIndex)));
      if (relayed == DataRelayer::WillNotRelay) {
        reportError("Unable to relay part.");
        return;
      }
      LOG(DEBUG) << "Relaying part idx: " << headerIndex;
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
  if (isValidInput() == false) {
    reportError("Parts should come in couples. Dropping it.");
    return true;
  }
  putIncomingMessageIntoCache();
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
  auto& context = mFairMQContext;
  auto& device = *this;
  auto& errorCallback = mError;
  auto& errorCount = mErrorCount;
  auto& forwards = mSpec.forwards;
  auto& inputsSchema = mSpec.inputs;
  auto& processingCount = mProcessingCount;
  auto& rdfContext = mDataFrameContext;
  auto& relayer = mRelayer;
  auto& rootContext = mRootContext;
  auto& serviceRegistry = mServiceRegistry;
  auto& statefulProcess = mStatefulProcess;
  auto& statelessProcess = mStatelessProcess;
  auto& stringContext = mStringContext;
  auto& timingInfo = mTimingInfo;
  auto& timesliceIndex = mServiceRegistry.get<TimesliceIndex>();
  auto& rawContext = mRawBufferContext;

  // These duplicate references are created so that each function
  // does not need to know about the whole class state, but I can
  // fine grain control what is exposed at each state.
  // FIXME: I should use a different id for this state.
  auto& monitoringService = mServiceRegistry.get<Monitoring>();
  monitoringService.send({ DataProcessingStatus::IN_DPL_WRAPPER, "dpl/in_handle_data" });
  ScopedExit metricFlusher([&monitoringService] {
      monitoringService.send({ DataProcessingStatus::IN_DPL_WRAPPER, "dpl/in_handle_data"});
      monitoringService.send({ DataProcessingStatus::IN_FAIRMQ, "dpl/in_handle_data"});
      monitoringService.flushBuffer(); });

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
  auto getReadyActions = [&relayer, &completed, &monitoringService]() -> std::vector<DataRelayer::RecordAction> {
    LOG(DEBUG) << "Getting parts to process";
    int pendingInputs = (int)relayer.getParallelTimeslices() - completed.size();
    monitoringService.send({ pendingInputs, "inputs/relayed/pending" });
    if (completed.empty()) {
      monitoringService.send({ 1, "inputs/relayed/incomplete" });
    }
    return completed;
  };

  // This is needed to convert from a pair of pointers to an actual DataRef
  // and to make sure the ownership is moved from the cache in the relayer to
  // the execution.
  auto fillInputs = [&relayer, &inputsSchema, &currentSetOfInputs](TimesliceSlot slot) -> InputRecord {
    currentSetOfInputs = std::move(relayer.getInputsForTimeslice(slot));
    InputSpan span{ [&currentSetOfInputs](size_t i) -> char const* {
                     return currentSetOfInputs.at(i) ? static_cast<char const*>(currentSetOfInputs.at(i)->GetData()) : nullptr;
                   },
                    currentSetOfInputs.size() };
    return InputRecord{ inputsSchema, std::move(span) };
  };

  // This is the thing which does the actual computation. No particular reason
  // why we do the stateful processing before the stateless one.
  // PROCESSING:{START,END} is done so that we can trigger on begin / end of processing
  // in the GUI.
  auto dispatchProcessing = [&processingCount, &allocator, &statefulProcess, &statelessProcess, &monitoringService,
                             &context, &rootContext, &stringContext, &rdfContext, &rawContext, &serviceRegistry, &device](TimesliceSlot slot, InputRecord& record) {
    if (statefulProcess) {
      LOG(DEBUG) << "PROCESSING:START:" << slot.index;
      monitoringService.send({ processingCount++, "dpl/stateful_process_count" });
      ProcessingContext processContext{record, serviceRegistry, allocator};
      monitoringService.send({ DataProcessingStatus::IN_DPL_STATEFUL_CALLBACK, "dpl/in_handle_data" });
      statefulProcess(processContext);
      monitoringService.send({ DataProcessingStatus::IN_DPL_WRAPPER, "dpl/in_handle_data" });
      LOG(DEBUG) << "PROCESSING:END:" << slot.index;
    }
    if (statelessProcess) {
      LOG(DEBUG) << "PROCESSING:START:" << slot.index;
      monitoringService.send({ processingCount++, "dpl/stateless_process_count" });
      ProcessingContext processContext{record, serviceRegistry, allocator};
      monitoringService.send({ DataProcessingStatus::IN_DPL_STATELESS_CALLBACK, "dpl/in_handle_data" });
      statelessProcess(processContext);
      monitoringService.send({ DataProcessingStatus::IN_DPL_WRAPPER, "dpl/in_handle_data" });
      LOG(DEBUG) << "PROCESSING:END:" << slot.index;
    }

    DataProcessor::doSend(device, context);
    DataProcessor::doSend(device, rootContext);
    DataProcessor::doSend(device, stringContext);
    DataProcessor::doSend(device, rdfContext);
    DataProcessor::doSend(device, rawContext);
  };

  // Error handling means printing the error and updating the metric
  auto errorHandling = [&errorCallback, &monitoringService, &serviceRegistry](std::exception& e, InputRecord& record) {
    monitoringService.send({ DataProcessingStatus::IN_DPL_ERROR_CALLBACK, "dpl/in_handle_data" });
    LOG(ERROR) << "Exception caught: " << e.what() << std::endl;
    if (errorCallback) {
      monitoringService.send({ 1, "error" });
      ErrorContext errorContext{record, serviceRegistry, e};
      errorCallback(errorContext);
    }
    monitoringService.send({ DataProcessingStatus::IN_DPL_WRAPPER, "dpl/in_handle_data" });
  };

  // I need a preparation step which gets the current timeslice id and
  // propagates it to the various contextes (i.e. the actual entities which
  // create messages) because the messages need to have the timeslice id into
  // it.
  auto prepareAllocatorForCurrentTimeSlice = [&timingInfo, &rootContext, &stringContext, &rawContext, &context, &relayer, &timesliceIndex](TimesliceSlot i) {
    auto timeslice = timesliceIndex.getTimesliceForSlot(i);
    LOG(DEBUG) << "Timeslice for cacheline is " << timeslice.value;
    timingInfo.timeslice = timeslice.value;
    rootContext.clear();
    context.clear();
    stringContext.clear();
    rawContext.clear();
  };

  // This is how we do the forwarding, i.e. we push
  // the inputs which are shared between this device and others
  // to the next one in the daisy chain.
  // FIXME: do it in a smarter way than O(N^2)
  auto forwardInputs = [&reportError, &forwards, &device, &currentSetOfInputs](TimesliceSlot slot, InputRecord& record) {
    assert(record.size()*2 == currentSetOfInputs.size());
    LOG(DEBUG) << "FORWARDING:START:" << slot.index;
    for (size_t ii = 0, ie = record.size(); ii < ie; ++ii) {
      DataRef input = record.getByPos(ii);

      // If is now possible that the record is not complete when
      // we forward it, because of a custom completion policy.
      // this means that we need to skip the empty entries in the
      // record for being forwarded.
      if (input.header == nullptr || input.payload == nullptr) {
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

      auto &header = currentSetOfInputs[ii*2];
      auto &payload = currentSetOfInputs[ii*2+1];

      for (auto forward : forwards) {
        LOG(DEBUG) << "Input part content";
        LOG(DEBUG) << dh->dataOrigin.str;
        LOG(DEBUG) << dh->dataDescription.str;
        LOG(DEBUG) << dh->subSpecification;
        if (DataSpecUtils::match(forward.matcher, dh->dataOrigin, dh->dataDescription, dh->subSpecification)
            && (dph->startTime % forward.maxTimeslices) == forward.timeslice) {

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
          LOG(DEBUG) << "Forwarding data to " << forward.channel;
          LOG(DEBUG) << "Forwarded timeslice is " << fdph->startTime;
          LOG(DEBUG) << "Forwarded channel is " << forward.channel;
          FairMQParts forwardedParts;
          forwardedParts.AddPart(std::move(header));
          forwardedParts.AddPart(std::move(payload));
          assert(forwardedParts.Size() == 2);
          assert(o2::header::get<DataProcessingHeader*>(forwardedParts.At(0)->GetData()));
          LOG(DEBUG) << o2::header::get<DataProcessingHeader*>(forwardedParts.At(0)->GetData())->startTime;
          LOG(DEBUG) << forwardedParts.At(0)->GetSize();
          // FIXME: this should use a correct subchannel
          device.Send(forwardedParts, forward.channel, 0);
        }
      }
    }
    LOG(DEBUG) << "FORWARDING:END";
  };

  // We use this to keep track of the latency of the first message we get for a given input record
  // and of the last one.
  struct InputLatency {
    uint64_t minLatency;
    uint64_t maxLatency;
  };

  auto calculateInputRecordLatency = [](InputRecord const& record, auto now) -> InputLatency {
    InputLatency result{ static_cast<uint64_t>(-1), 0 };

    auto currentTime = (uint64_t)std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
    for (auto& item : record) {
      auto* header = o2::header::get<DataProcessingHeader*>(item.header);
      if (header == nullptr) {
        continue;
      }
      auto partLatency = currentTime - header->creation;
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

  if (canDispatchSomeComputation() == false) {
    return false;
  }

  for (auto action: getReadyActions()) {
    if (action.op == CompletionPolicy::CompletionOp::Wait) {
      continue;
    }

    prepareAllocatorForCurrentTimeSlice(TimesliceSlot{ action.slot });
    InputRecord record = fillInputs(action.slot);
    if (action.op == CompletionPolicy::CompletionOp::Discard) {
      if (forwards.empty() == false) {
        forwardInputs(action.slot, record);
        continue;
      }
    }
    auto tStart = std::chrono::high_resolution_clock::now();
    try {
      for (size_t ai = 0; ai != record.size(); ai++) {
        auto cacheId = action.slot.index * record.size() + ai;
        auto state = record.isValid(ai) ? 2 : 0;
        monitoringService.send({ state, "data_relayer/" + std::to_string(cacheId) });
      }
      dispatchProcessing(action.slot, record);
      for (size_t ai = 0; ai != record.size(); ai++) {
        auto cacheId = action.slot.index * record.size() + ai;
        auto state = record.isValid(ai) ? 3 : 0;
        monitoringService.send({ state, "data_relayer/" + std::to_string(cacheId) });
      }
    } catch(std::exception &e) {
      errorHandling(e, record);
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    double elapsedTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    auto totalProcessedSize = calculateTotalInputRecordSize(record);
    auto latency = calculateInputRecordLatency(record, tStart);

    /// The size of all the input messages which have been processed in this iteration
    monitoringService.send({ totalProcessedSize, "dpl/processed_input_size_bytes" });
    /// The time to do the processing for this iteration
    monitoringService.send({ elapsedTimeMs, "dpl/elapsed_time_ms" });
    /// The rate at which processing was happening in this iteration
    monitoringService.send({ (int)((totalProcessedSize / elapsedTimeMs) / 1000), "dpl/processing_rate_mb_s" });
    /// The smallest latency between an input message being created and its processing
    /// starting.
    monitoringService.send({ (int)latency.minLatency, "dpl/min_input_latency_ms" });
    /// The largest latency between an input message being created and its processing
    /// starting.
    monitoringService.send({ (int)latency.maxLatency, "dpl/max_input_latency_ms" });
    /// The rate at which we get inputs, i.e. the longest time between one of the inputs being
    /// created and actually reaching the consumer device.
    monitoringService.send({ (int)((totalProcessedSize / latency.maxLatency) / 1000), "dpl/input_rate_mb_s" });

    // We forward inputs only when we consume them. If we simply Process them,
    // we keep them for next message arriving.
    if (action.op == CompletionPolicy::CompletionOp::Consume) {
      if (forwards.empty() == false) {
        forwardInputs(action.slot, record);
      }
    }
  }

  return true;
}

void
DataProcessingDevice::error(const char *msg) {
  LOG(ERROR) << msg;
  mErrorCount++;
  mServiceRegistry.get<Monitoring>().send({ mErrorCount, "dpl/errors" });
}

} // namespace framework
} // namespace o2
