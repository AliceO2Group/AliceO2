// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataSourceDevice.h"
#include "Framework/MetricsService.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/DataProcessor.h"
#include "Framework/FairOptionsRetriever.h"
#include "Framework/DataProcessingHeader.h"
#include <cassert>
#include <chrono>
#include <thread> // this_thread::sleep_for
using TimeScale = std::chrono::microseconds;

using namespace o2::framework;

namespace o2 {
namespace framework {

DataSourceDevice::DataSourceDevice(const DeviceSpec &spec, ServiceRegistry &registry)
: mInit{spec.algorithm.onInit},
  mStatefulProcess{nullptr},
  mStatelessProcess{spec.algorithm.onProcess},
  mError{spec.algorithm.onError},
  mConfigRegistry{nullptr},
  mAllocator{this,&mContext, &mRootContext, spec.outputs},
  mServiceRegistry{registry},
  mCurrentTimeslice{0},
  mRate{0.},
  mLastTime{0}
{
}

void DataSourceDevice::Init() {
  LOG(DEBUG) << "DataSourceDevice::InitTask::START\n";
  LOG(DEBUG) << "Init thread" << pthread_self();
  std::unique_ptr<ParamRetriever> retriever{new FairOptionsRetriever(GetConfig())};
  mConfigRegistry = std::move(std::make_unique<ConfigParamRegistry>(std::move(retriever)));
  if (mInit) {
    LOG(DEBUG) << "Found onInit method. Executing";
    InitContext initContext{*mConfigRegistry,mServiceRegistry};
    mStatefulProcess = mInit(initContext);
  }
  LOG(DEBUG) << "DataSourceDevice::InitTask::END";
}

bool DataSourceDevice::ConditionalRun() {
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
  InputRecord dummyInputs{{}, {}};
  try {
    mContext.prepareForTimeslice(mCurrentTimeslice);
    mRootContext.prepareForTimeslice(mCurrentTimeslice);
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
    size_t nMsg = mContext.size() + mRootContext.size();
    LOG(DEBUG) << "Process produced " << nMsg << " messages";
    DataProcessor::doSend(*this, mContext);
    DataProcessor::doSend(*this, mRootContext);
  } catch(std::exception &e) {
    if (mError) {
      ErrorContext errorContext{dummyInputs, mServiceRegistry, e};
      mError(errorContext);
    } else {
      LOG(DEBUG) << "Uncaught exception: " << e.what();
    }
  } catch(...) {
    LOG(DEBUG) << "Unknown exception type.";
    LOG(DEBUG) << "DataSourceDevice::Processing::END";
    return false;
  }
  LOG(DEBUG) << "DataSourceDevice::Processing::END";
  return true;
}

} // namespace framework
} // namespace o2
