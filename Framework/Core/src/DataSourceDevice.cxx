// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataSourceDevice.h"
#include "Framework/MetricsService.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/DataProcessor.h"
#include <cassert>

using namespace o2::framework;
using DataHeader = o2::Header::DataHeader;

namespace o2 {
namespace framework {

DataSourceDevice::DataSourceDevice(const DeviceSpec &spec)
: mInit{spec.init},
  mProcess{spec.process},
  mError{spec.onError},
  mAllocator{this,&mContext,&mRootContext,spec.outputs}
{
}

void DataSourceDevice::InitTask() {
  // We should parse all the 
  LOG(DEBUG) << "DataSourceDevice::InitTask::START\n";
  if (mInit) {
    mInit(mConfigRegistry, mServiceRegistry);
  }
  LOG(DEBUG) << "DataSourceDevice::InitTask::END";
}


bool DataSourceDevice::ConditionalRun() {
  // We do not have any inputs for a source, by definition, 
  // so we simply pass an empty vector.
//  auto &metricsService = mServiceRegistry.get<MetricsService>();
  LOG(DEBUG) << "DataSourceDevice::Processing::START";
  std::vector<DataRef> dummyInputs;
  try {
    mContext.clear();
    mRootContext.clear();
    if (mProcess) {
      LOG(DEBUG) << "Has process callback";
      mProcess(dummyInputs, mServiceRegistry, mAllocator);
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
