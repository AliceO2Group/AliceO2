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
#include <cassert>

using namespace o2::framework;
using DataHeader = o2::Header::DataHeader;

namespace o2 {
namespace framework {

DataSourceDevice::DataSourceDevice(const DeviceSpec &spec, ServiceRegistry &registry)
: mInit{spec.algorithm.onInit},
  mStatefulProcess{nullptr},
  mStatelessProcess{spec.algorithm.onProcess},
  mError{spec.algorithm.onError},
  mConfigRegistry{nullptr},
  mAllocator{this,&mContext,&mRootContext,spec.outputs},
  mServiceRegistry{registry}
{
}

void DataSourceDevice::Init() {
  LOG(DEBUG) << "DataSourceDevice::InitTask::START\n";
  LOG(DEBUG) << "Init thread" << pthread_self();
  std::unique_ptr<ParamRetriever> retriever{new FairOptionsRetriever(GetConfig())};
  mConfigRegistry = std::move(std::make_unique<ConfigParamRegistry>(std::move(retriever)));
  if (mInit) {
    LOG(DEBUG) << "Found onInit method. Executing";
    mStatefulProcess = mInit(*mConfigRegistry, mServiceRegistry);
  }
  LOG(DEBUG) << "DataSourceDevice::InitTask::END";
}

bool DataSourceDevice::ConditionalRun() {
  // We do not have any inputs for a source, by definition,
  // so we simply pass an empty vector.
//  auto &metricsService = mServiceRegistry.get<MetricsService>();
  LOG(DEBUG) << "DataSourceDevice::Processing::START";
  LOG(DEBUG) << "ConditionalRun thread" << pthread_self();
  std::vector<DataRef> dummyInputs;
  try {
    mContext.clear();
    mRootContext.clear();

    // Avoid runaway process in case we have nothing to do.
    if ((!mStatefulProcess) && (!mStatelessProcess)) {
      LOG(ERROR) << "No callback defined";
      sleep(1);
    }

    if (mStatelessProcess) {
      LOG(DEBUG) << "Has stateless process callback";
      mStatelessProcess(dummyInputs, mServiceRegistry, mAllocator);
    }
    if (mStatefulProcess) {
      LOG(DEBUG) << "Has stateful process callback";
      mStatefulProcess(dummyInputs, mServiceRegistry, mAllocator);
    }
    size_t nMsg = mContext.size() + mRootContext.size();
    LOG(DEBUG) << "Process produced " << nMsg << " messages";
    DataProcessor::doSend(*this, mContext);
    DataProcessor::doSend(*this, mRootContext);
  } catch(std::exception &e) {
    if (mError) {
      mError(dummyInputs, mServiceRegistry, e);
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
