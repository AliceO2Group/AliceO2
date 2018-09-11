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

namespace o2
{
namespace framework
{

DataProcessingDevice::DataProcessingDevice(const DeviceSpec& spec, ServiceRegistry& registry)
  : mInit{ spec.algorithm.onInit },
    mStatefulProcess{ nullptr },
    mStatelessProcess{ spec.algorithm.onProcess },
    mError{ spec.algorithm.onError },
    mConfigRegistry{ nullptr },
    mFairMQContext{ FairMQDeviceProxy{ this } },
    mRootContext{ FairMQDeviceProxy{ this } },
    mStringContext{ FairMQDeviceProxy{ this } },
    mDataFrameContext{ FairMQDeviceProxy{ this } },
    mContextRegistry{ { &mFairMQContext, &mRootContext, &mStringContext, &mDataFrameContext } },
    mAllocator{ &mTimingInfo, &mContextRegistry, spec.outputs },
    mRelayer{ spec.completionPolicy, spec.inputs, spec.forwards, registry.get<Monitoring>() },
    mInputChannels{ spec.inputChannels },
    mOutputChannels{ spec.outputChannels },
    mInputs{ spec.inputs },
    mForwards{ spec.forwards },
    mServiceRegistry{ registry },
    mErrorCount{ 0 },
    mProcessingCount{ 0 }
{
}

/// This  takes care  of initialising  the device  from its  specification. In
/// particular it needs to:
///
/// * Allocate the channels as needed and attach HandleData to each one of them
/// * Invoke the actual init
void DataProcessingDevice::Init() {
  LOG(DEBUG) << "DataProcessingDevice::InitTask::START";
  auto optionsRetriever(std::make_unique<FairOptionsRetriever>(GetConfig()));
  mConfigRegistry = std::move(std::make_unique<ConfigParamRegistry>(std::move(optionsRetriever)));

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
  for (auto& channel : mInputChannels) {
    FairMQParts parts;
    auto result = this->ReceiveAsync(parts, channel.name);
    if (result > 0) {
      this->handleData(parts);
    }
  }
  return true;
}

/// This is the inner loop of our framework. The actual implementation
/// is divided in two parts. In the first one we define a set of lambdas
/// which describe what is actually going to happen, hiding all the state
/// boilerplate which the user does not need to care about at top level.
bool DataProcessingDevice::handleData(FairMQParts& parts)
{
  assert(mInputChannels.empty() == false);
  assert(parts.Size() > 0);

  static const std::string handleDataMetricName = "dpl/in_handle_data";
  // Initial part. Let's hide all the unnecessary and have
  // simple lambdas for each of the steps I am planning to have.
  assert(!mInputs.empty());

  // These duplicate references are created so that each function
  // does not need to know about the whole class state, but I can 
  // fine grain control what is exposed at each state.
  auto& monitoringService = mServiceRegistry.get<Monitoring>();
  monitoringService.send({ 1, "dpl/in_handle_data" });
  ScopedExit metricFlusher([&monitoringService] {
      monitoringService.send({ 1, "dpl/in_handle_data"});
      monitoringService.send({ 0, "dpl/in_handle_data"});
      monitoringService.flushBuffer(); });
  auto &statefulProcess = mStatefulProcess;
  auto &statelessProcess = mStatelessProcess;
  auto &errorCallback = mError;
  auto &serviceRegistry = mServiceRegistry;
  auto &allocator = mAllocator;
  auto &processingCount = mProcessingCount;
  auto &relayer = mRelayer;
  auto &device = *this;
  auto &timingInfo = mTimingInfo;
  auto& context = mFairMQContext;
  auto &rootContext = mRootContext;
  auto& stringContext = mStringContext;
  auto& rdfContext = mDataFrameContext;
  auto &forwards = mForwards;
  auto &inputsSchema = mInputs;
  auto &errorCount = mErrorCount;

  std::vector<std::unique_ptr<FairMQMessage>> currentSetOfInputs;

  // This is the actual hidden state for the outer loop. In case we decide we
  // want to support multithreaded dispatching of operations, I can simply
  // move these to some thread local store and the rest of the lambdas
  // should work just fine.
  std::vector<DataRelayer::RecordAction> completed;

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

  //
  auto reportError = [&errorCount, &monitoringService](const char* message) {
    LOG(ERROR) << message;
    errorCount++;
    monitoringService.send({ errorCount, "dpl/errors" });
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
  auto fillInputs = [&relayer, &inputsSchema, &currentSetOfInputs](int timeslice) -> InputRecord {
    currentSetOfInputs = std::move(relayer.getInputsForTimeslice(timeslice));
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
                             &context, &rootContext, &stringContext, &rdfContext, &serviceRegistry, &device](int i, InputRecord& record) {
    if (statefulProcess) {
      LOG(DEBUG) << "PROCESSING:START:" << i;
      monitoringService.send({ processingCount++, "dpl/stateful_process_count" });
      ProcessingContext processContext{record, serviceRegistry, allocator};
      monitoringService.send({ 2, "dpl/in_handle_data" });
      statefulProcess(processContext);
      monitoringService.send({ 1, "dpl/in_handle_data" });
      LOG(DEBUG) << "PROCESSING:END:" << i;
    }
    if (statelessProcess) {
      LOG(DEBUG) << "PROCESSING:START:" << i;
      monitoringService.send({ processingCount++, "dpl/stateless_process_count" });
      ProcessingContext processContext{record, serviceRegistry, allocator};
      monitoringService.send({ 2, "dpl/in_handle_data" });
      statelessProcess(processContext);
      monitoringService.send({ 1, "dpl/in_handle_data" });
      LOG(DEBUG) << "PROCESSING:END:" << i;
    }

    DataProcessor::doSend(device, context);
    DataProcessor::doSend(device, rootContext);
    DataProcessor::doSend(device, stringContext);
    DataProcessor::doSend(device, rdfContext);
  };

  // Error handling means printing the error and updating the metric
  auto errorHandling = [&errorCallback, &monitoringService, &serviceRegistry](std::exception& e, InputRecord& record) {
    monitoringService.send({ 3, "dpl/in_handle_data" });
    LOG(ERROR) << "Exception caught: " << e.what() << std::endl;
    if (errorCallback) {
      monitoringService.send({ 1, "error" });
      ErrorContext errorContext{record, serviceRegistry, e};
      errorCallback(errorContext);
    }
    monitoringService.send({ 1, "dpl/in_handle_data" });
  };

  // I need a preparation step which gets the current timeslice id and
  // propagates it to the various contextes (i.e. the actual entities which
  // create messages) because the messages need to have the timeslice id into
  // it.
  auto prepareAllocatorForCurrentTimeSlice = [&timingInfo, &rootContext, &stringContext, &context, &relayer](int i) {
    size_t timeslice = relayer.getTimesliceForCacheline(i);
    LOG(DEBUG) << "Timeslice for cacheline is " << timeslice;
    timingInfo.timeslice = timeslice;
    rootContext.clear();
    context.clear();
    stringContext.clear();
  };

  // This is how we do the forwarding, i.e. we push 
  // the inputs which are shared between this device and others
  // to the next one in the daisy chain.
  // FIXME: do it in a smarter way than O(N^2)
  auto forwardInputs = [&reportError, &forwards, &device, &currentSetOfInputs]
                       (int timeslice, InputRecord &record) {
    assert(record.size()*2 == currentSetOfInputs.size());
    LOG(DEBUG) << "FORWARDING:START:" << timeslice;
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
        if (DataSpecUtils::match(forward.matcher, dh->dataOrigin,
                                 dh->dataDescription,
                                 dh->subSpecification)) {

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
  if (canDispatchSomeComputation() == false) {
    return true;
  }

  for (auto action: getReadyActions()) {
    if (action.op == CompletionPolicy::CompletionOp::Wait) {
      continue;
    }

    prepareAllocatorForCurrentTimeSlice(action.cacheLineIdx);
    InputRecord record = fillInputs(action.cacheLineIdx);
    if (action.op == CompletionPolicy::CompletionOp::Discard) {
      if (forwards.empty() == false) {
        forwardInputs(action.cacheLineIdx, record);
        continue;
      }
    }
    try {
      for (size_t ai = 0; ai != record.size(); ai++) {
        auto cacheId = action.cacheLineIdx * record.size() + ai;
        auto state = record.isValid(ai) ? 2 : 0;
        monitoringService.send({ state, "data_relayer/" + std::to_string(cacheId) });
      }
      dispatchProcessing(action.cacheLineIdx, record);
      for (size_t ai = 0; ai != record.size(); ai++) {
        auto cacheId = action.cacheLineIdx * record.size() + ai;
        auto state = record.isValid(ai) ? 3 : 0;
        monitoringService.send({ state, "data_relayer/" + std::to_string(cacheId) });
      }
    } catch(std::exception &e) {
      errorHandling(e, record);
    }
    // We forward inputs only when we consume them. If we simply Process them,
    // we keep them for next message arriving.
    if (action.op == CompletionPolicy::CompletionOp::Consume) {
      if (forwards.empty() == false) {
        forwardInputs(action.cacheLineIdx, record);
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
