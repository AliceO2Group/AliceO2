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
#include "Framework/MetricsService.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/InputRecord.h"
#include <fairmq/FairMQParts.h>
#include <options/FairMQProgOptions.h>
#include <TMessage.h>
#include <TClonesArray.h>

#include <vector>
#include <memory>

using namespace o2::framework;

using DataHeader = o2::header::DataHeader;

namespace o2 {
namespace framework {

DataProcessingDevice::DataProcessingDevice(const DeviceSpec &spec,
                                           ServiceRegistry &registry)
: mInit{spec.algorithm.onInit},
  mStatefulProcess{nullptr},
  mStatelessProcess{spec.algorithm.onProcess},
  mError{spec.algorithm.onError},
  mConfigRegistry{nullptr},
  mAllocator{this, &mContext, &mRootContext, spec.outputs},
  mRelayer{spec.inputs, spec.forwards, registry.get<MetricsService>()},
  mInputChannels{spec.inputChannels},
  mOutputChannels{spec.outputChannels},
  mInputs{spec.inputs},
  mForwards{spec.forwards},
  mServiceRegistry{registry},
  mErrorCount{0},
  mProcessingCount{0}
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
  if (mInputChannels.empty()) {
    LOG(ERROR) << "DataProcessingDevice should have at least one input channel";
  }
  for (auto &channel : mInputChannels) {
    OnData(channel.name.c_str(), &DataProcessingDevice::HandleData);
  }

  if (mInit) {
    InitContext initContext{*mConfigRegistry,mServiceRegistry};
    mStatefulProcess = mInit(initContext);
  }
  LOG(DEBUG) << "DataProcessingDevice::InitTask::END";
}

/// This is the inner loop of our framework. The actual implementation
/// is divided in two parts. In the first one we define a set of lambdas
/// which describe what is actually going to happen, hiding all the state
/// boilerplate which the user does not need to care about at top level.
/// In the second part 
bool
DataProcessingDevice::HandleData(FairMQParts &iParts, int /*index*/) {
  // Initial part. Let's hide all the unnecessary and have
  // simple lambdas for each of the steps I am planning to have.
  assert(!mInputs.empty());

  // These duplicate references are created so that each function
  // does not need to know about the whole class state, but I can 
  // fine grain control what is exposed at each state.
  auto &metricsService = mServiceRegistry.get<MetricsService>();
  auto &statefulProcess = mStatefulProcess;
  auto &statelessProcess = mStatelessProcess;
  auto &errorCallback = mError;
  auto &serviceRegistry = mServiceRegistry;
  auto &allocator = mAllocator;
  auto &processingCount = mProcessingCount;
  auto &relayer = mRelayer;
  auto &device = *this;
  auto &context = mContext;
  auto &rootContext = mRootContext;
  auto &forwards = mForwards;
  auto &inputsSchema = mInputs;
  auto &errorCount = mErrorCount;

  std::vector<std::unique_ptr<FairMQMessage>> currentSetOfInputs;

  // This is the actual hidden state for the outer loop. In case we decide we
  // want to support multithreaded dispatching of operations, I can simply
  // move these to some thread local store and the rest of the lambdas
  // should work just fine.
  FairMQParts &parts = iParts;
  std::vector<int> completed;


  // This is how we validate inputs. I.e. we try to enforce the O2 Data model
  // and we do a few stats. We bind parts as a lambda captured variable, rather
  // than an input, because we do not want the outer loop actually be exposed
  // to the implementation details of the messaging layer.
  auto isValidInput = [&metricsService, &parts]() -> bool {
    metricsService.post("inputs/parts/total", (int)parts.Size());

    for (size_t i = 0; i < parts.Size() ; ++i) {
      LOG(DEBUG) << " part " << i << " is " << parts.At(i)->GetSize() << " bytes";
    }
    if (parts.Size() % 2) {
      return false;
    }
    for (size_t hi = 0; hi < parts.Size()/2; ++hi) {
      auto pi = hi*2;
      auto dh = o2::header::get<DataHeader>(parts.At(pi)->GetData());
      if (!dh) {
        LOG(ERROR) << "Header is not a DataHeader?";
        return false;
      }
      if (dh->payloadSize != parts.At(pi+1)->GetSize()) {
        LOG(ERROR) << "DataHeader payloadSize mismatch";
        return false;
      }
      auto dph = o2::header::get<DataProcessingHeader>(parts.At(pi)->GetData());
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
  auto reportError = [&errorCount, &metricsService](const char *message) {
    LOG(ERROR) << message;
    errorCount++;
    metricsService.post("dataprocessing/errors", errorCount);
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
  auto getCompleteInputSets = [&relayer,&completed,&metricsService]() -> std::vector<int> {
    LOG(DEBUG) << "Getting parts to process";
    completed = relayer.getReadyToProcess();
    int pendingInputs = (int)relayer.getParallelTimeslices() - completed.size();
    metricsService.post("inputs/relayed/pending", pendingInputs);
    if (completed.empty()) {
      metricsService.post("inputs/relayed/incomplete", 1);
    }
    return completed;
  };

  // This is needed to convert from a pair of pointers to an actual DataRef
  // and to make sure the ownership is moved from the cache in the relayer to
  // the execution.
  auto fillInputs = [&relayer, &inputsSchema, &currentSetOfInputs](int timeslice) -> InputRecord {
    currentSetOfInputs = std::move(relayer.getInputsForTimeslice(timeslice));
    InputRecord registry{inputsSchema, currentSetOfInputs};
    return registry;
  };

  // This is the thing which does the actual computation. No particular reason
  // why we do the stateful processing before the stateless one.
  // PROCESSING:{START,END} is done so that we can trigger on begin / end of processing
  // in the GUI.
  auto dispatchProcessing = [&processingCount,
                             &allocator,
                             &statefulProcess,
                             &statelessProcess,
                             &metricsService,
                             &context,
                             &rootContext,
                             &serviceRegistry,
                             &device](int i, InputRecord &record) {
    if (statefulProcess) {
      LOG(DEBUG) << "PROCESSING:START:" << i;
      metricsService.post("dataprocessing/stateful_process", processingCount++);
      ProcessingContext processContext{record, serviceRegistry, allocator};
      statefulProcess(processContext);
      LOG(DEBUG) << "PROCESSING:END:" << i;
    }
    if (statelessProcess) {
      LOG(DEBUG) << "PROCESSING:START:" << i;
      metricsService.post("dataprocessing/stateless_process", processingCount++);
      ProcessingContext processContext{record, serviceRegistry, allocator};
      statelessProcess(processContext);
      LOG(DEBUG) << "PROCESSING:END:" << i;
    }
    DataProcessor::doSend(device, context);
    DataProcessor::doSend(device, rootContext);
  };

  // Error handling means printing the error and updating the metric
  auto errorHandling = [&errorCallback,
                        &metricsService,
                        &serviceRegistry](std::exception &e, InputRecord &record) {
    LOG(ERROR) << "Exception caught: " << e.what() << std::endl;
    if (errorCallback) {
      metricsService.post("error", 1);
      ErrorContext errorContext{record, serviceRegistry, e};
      errorCallback(errorContext);
    }
  };

  // I need a preparation step which gets the current timeslice id and
  // propagates it to the various contextes (i.e. the actual entities which
  // create messages) because the messages need to have the timeslice id into
  // it.
  auto prepareForCurrentTimeSlice = [&rootContext, &context, &relayer](int i) {
    size_t timeslice = relayer.getTimesliceForCacheline(i);
    LOG(DEBUG) << "Timeslice for cacheline is " << timeslice;
    rootContext.prepareForTimeslice(timeslice);
    context.prepareForTimeslice(timeslice);
  };

  // This is how we do the forwarding, i.e. we push 
  // the inputs which are shared between this device and others
  // to the next one in the daisy chain.
  // FIXME: do it in a smarter way than O(N^2)
  auto forwardInputs = [&reportError, &forwards, &device, &currentSetOfInputs]
                       (int timeslice, InputRecord &record) {
    assert(record.size()*2 == currentSetOfInputs.size());
    LOG(DEBUG) << "FORWARDING:START:" << timeslice;
    for (size_t ii = 0, ie = record.size(); ii != ie; ++ii) {
      DataRef input = record.getByPos(ii);
      assert(input.header);
      auto dh = o2::header::get<DataHeader>(input.header);
      if (!dh) {
        reportError("Header is not a DataHeader?");
        continue;
      }
      auto dph = o2::header::get<DataProcessingHeader>(input.header);
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
          auto fdph = o2::header::get<DataProcessingHeader>(header.get()->GetData());
          if (fdph == nullptr) {
            LOG(ERROR) << "Forwarded data does not have a DataProcessingHeader";
            continue;
          }
          auto fdh = o2::header::get<DataHeader>(header.get()->GetData());
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
          assert(o2::header::get<DataProcessingHeader>(forwardedParts.At(0)->GetData()));
          LOG(DEBUG) << o2::header::get<DataProcessingHeader>(forwardedParts.At(0)->GetData())->startTime;
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
  // messages).
  if (isValidInput() == false) {
    reportError("Parts should come in couples. Dropping it.");
    return true;
  }
  putIncomingMessageIntoCache();
  if (canDispatchSomeComputation() == false) {
    return true;
  }

  for (auto cacheline : getCompleteInputSets()) {
    prepareForCurrentTimeSlice(cacheline);
    InputRecord record = fillInputs(cacheline);
    try {
      dispatchProcessing(cacheline, record);
    } catch(std::exception &e) {
      errorHandling(e, record);
    }
    forwardInputs(cacheline, record);
  }

  return true;
}

void
DataProcessingDevice::error(const char *msg) {
  LOG(ERROR) << msg;
  mErrorCount++;
  mServiceRegistry.get<MetricsService>().post("dataprocessing/errors", mErrorCount);
}

} // namespace framework
} // namespace o2
