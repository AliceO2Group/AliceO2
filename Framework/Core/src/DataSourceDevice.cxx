// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/ContextRegistry.h"
#include "Framework/MessageContext.h"
#include "Framework/RootObjectContext.h"
#include "Framework/ArrowContext.h"
#include "Framework/DataSourceDevice.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/DataProcessor.h"
#include "Framework/FairOptionsRetriever.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Framework/DataProcessingHeader.h"
#include "DataProcessingStatus.h"
#include "Framework/CallbackService.h"
#include "ScopedExit.h"
#include <Monitoring/Monitoring.h>

#include <cassert>
#include <chrono>
#include <thread> // this_thread::sleep_for
using TimeScale = std::chrono::microseconds;

using namespace o2::framework;

constexpr unsigned int MONITORING_QUEUE_SIZE = 100;

namespace o2
{
namespace framework
{

DataSourceDevice::DataSourceDevice(const DeviceSpec& spec, ServiceRegistry& registry)
  : mInit{ spec.algorithm.onInit },
    mStatefulProcess{ nullptr },
    mStatelessProcess{ spec.algorithm.onProcess },
    mError{ spec.algorithm.onError },
    mConfigRegistry{ nullptr },
    mFairMQContext{ this },
    mRootContext{ this },
    mStringContext{ this },
    mDataFrameContext{ this },
    mRawBufferContext{ FairMQDeviceProxy{ this } },
    mContextRegistry{ { &mFairMQContext, &mRootContext, &mStringContext, &mDataFrameContext, &mRawBufferContext } },
    mAllocator{ &mTimingInfo, &mContextRegistry, spec.outputs },
    mServiceRegistry{ registry },
    mCurrentTimeslice{ 0 },
    mRate{ 0. },
    mLastTime{ 0 }
{
}

void DataSourceDevice::Init() {
  LOG(DEBUG) << "DataSourceDevice::InitTask::START\n";
  LOG(DEBUG) << "Init thread" << pthread_self();
  // For some reason passing rateLogging does not work anymore.
  // This makes sure the maximum rate is once per minute.
  for (auto& x : fChannels) {
    for (auto& c : x.second) {
      if (c.GetRateLogging() < 60) {
        c.UpdateRateLogging(60);
      }
    }
  }
  std::unique_ptr<ParamRetriever> retriever{new FairOptionsRetriever(GetConfig())};
  mConfigRegistry = std::move(std::make_unique<ConfigParamRegistry>(std::move(retriever)));
  if (mInit) {
    LOG(DEBUG) << "Found onInit method. Executing";
    InitContext initContext{*mConfigRegistry,mServiceRegistry};
    mStatefulProcess = mInit(initContext);
  }
  auto& monitoring = mServiceRegistry.get<o2::monitoring::Monitoring>();
  monitoring.enableBuffering(MONITORING_QUEUE_SIZE);
  LOG(DEBUG) << "DataSourceDevice::InitTask::END";
}

void DataSourceDevice::PreRun() { mServiceRegistry.get<CallbackService>()(CallbackService::Id::Start); }

void DataSourceDevice::PostRun() { mServiceRegistry.get<CallbackService>()(CallbackService::Id::Stop); }

void DataSourceDevice::Reset() { mServiceRegistry.get<CallbackService>()(CallbackService::Id::Reset); }

bool DataSourceDevice::ConditionalRun() {
  auto& monitoring = mServiceRegistry.get<o2::monitoring::Monitoring>();
  monitoring.send({ DataProcessingStatus::IN_DPL_WRAPPER, "dpl/in_handle_data" });
  ScopedExit metricFlusher([&monitoring] {
      monitoring.send({ DataProcessingStatus::IN_DPL_WRAPPER, "dpl/in_handle_data" });
      monitoring.send({ DataProcessingStatus::IN_FAIRMQ, "dpl/in_handle_data" });
      monitoring.flushBuffer(); });
  static const auto reftime = std::chrono::system_clock::now();
  if (mRate > 0.001) {
    auto timeSinceRef = std::chrono::duration_cast<TimeScale>(std::chrono::system_clock::now() - reftime);
    auto timespan = timeSinceRef.count() - mLastTime;
    TimeScale::rep period = (float)TimeScale::period::den / mRate;
    if (timespan < period) {
      TimeScale sleepfor(period - timespan);
      std::this_thread::sleep_for(sleepfor);
    }
    mLastTime = std::chrono::duration_cast<TimeScale>(std::chrono::system_clock::now() - reftime).count();
  }
  LOG(DEBUG) << "DataSourceDevice::Processing::START";
  LOG(DEBUG) << "ConditionalRun thread" << pthread_self();
  // This is dummy because a source does not really have inputs.
  // However, in order to be orthogonal between sources and
  // processing code, we still specify it.
  InputRecord dummyInputs{ {}, { [](size_t) { return nullptr; }, 0 } };
  try {
    mTimingInfo.timeslice = mCurrentTimeslice;
    mContextRegistry.get<MessageContext>()->clear();
    mContextRegistry.get<RootObjectContext>()->clear();
    mContextRegistry.get<StringContext>()->clear();
    mContextRegistry.get<ArrowContext>()->clear();
    mContextRegistry.get<RawBufferContext>()->clear();
    mCurrentTimeslice += 1;

    // Avoid runaway process in case we have nothing to do.
    if ((!mStatefulProcess) && (!mStatelessProcess)) {
      LOG(ERROR) << "No callback defined";
      sleep(1);
    }

    ProcessingContext processingContext{dummyInputs, mServiceRegistry, mAllocator};
    if (mStatelessProcess) {
      LOG(DEBUG) << "Has stateless process callback";
      mStatelessProcess(processingContext);
    }
    if (mStatefulProcess) {
      LOG(DEBUG) << "Has stateful process callback";
      mStatefulProcess(processingContext);
    }
    size_t nMsg = mContextRegistry.get<MessageContext>()->size();
    nMsg += mContextRegistry.get<RootObjectContext>()->size();
    nMsg += mContextRegistry.get<StringContext>()->size();
    nMsg += mContextRegistry.get<ArrowContext>()->size();
    nMsg += mContextRegistry.get<RawBufferContext>()->size();
    monitoring.send({ (int)nMsg, "dpl/output_messages" });
    LOG(DEBUG) << "Process produced " << nMsg << " messages";
    DataProcessor::doSend(*this, *mContextRegistry.get<MessageContext>());
    DataProcessor::doSend(*this, *mContextRegistry.get<RootObjectContext>());
    DataProcessor::doSend(*this, *mContextRegistry.get<StringContext>());
    DataProcessor::doSend(*this, *mContextRegistry.get<ArrowContext>());
    DataProcessor::doSend(*this, *mContextRegistry.get<RawBufferContext>());
  } catch(std::exception &e) {
    if (mError) {
      ErrorContext errorContext{dummyInputs, mServiceRegistry, e};
      mError(errorContext);
    } else {
      LOG(ERROR) << "Uncaught exception: " << e.what();
    }
  } catch(...) {
    LOG(ERROR) << "Unknown exception type.";
    LOG(DEBUG) << "DataSourceDevice::Processing::END";
    return false;
  }
  LOG(DEBUG) << "DataSourceDevice::Processing::END";
  return true;
}

} // namespace framework
} // namespace o2
