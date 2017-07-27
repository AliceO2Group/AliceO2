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
#include <cassert>

using namespace o2::framework;

namespace o2 {
namespace framework {

DataSourceDevice::DataSourceDevice(const DeviceSpec &spec)
: mInit{spec.init},
  mProcess{spec.process},
  mError{spec.onError},
  mAllocator{this,&mContext,spec.outputs}
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
    if (mProcess) {
      LOG(DEBUG) << "Has process callback";
      mProcess(dummyInputs, mServiceRegistry, mAllocator);
    }
    LOG(DEBUG) << "Process produced " << mContext.size() << " messages";
    for (auto &message : mContext) {
 //     metricsService.post("outputs/total", message.parts.Size());
      assert(message.parts.Size() == 2);
      FairMQParts parts = std::move(message.parts);
      assert(message.parts.Size() == 0);
      assert(parts.Size() == 2);
      assert(parts.At(0)->GetSize() == 80);
      this->Send(parts, message.channel, message.index);
      assert(parts.Size() == 2);
    }
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
