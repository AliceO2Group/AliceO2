// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataSender.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/OutputRoute.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Monitoring.h"
#include "Framework/DataSpecUtils.h"

#include <fairmq/Device.h>

using namespace o2::monitoring;

namespace o2::framework
{

namespace
{
std::vector<size_t>
  createDistinctOutputRouteIndex(std::vector<OutputRoute> const& routes)
{
  std::vector<size_t> result;
  for (size_t ri = 0; ri < routes.size(); ++ri) {
    auto& route = routes[ri];
    if (route.timeslice == 0) {
      result.push_back(ri);
    }
  }
  return result;
}
} // namespace

DataSender::DataSender(ServiceRegistry& registry,
                       SendingPolicy const& policy)
  : mContext{registry.get<RawDeviceService>().device()},
    mRegistry{registry},
    mSpec{registry.get<DeviceSpec const>()},
    mDistinctRoutesIndex{createDistinctOutputRouteIndex(mSpec.outputs)},
    mPolicy{policy}
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);

  auto numInputTypes = mDistinctRoutesIndex.size();
  mQueriesMetricsNames.resize(numInputTypes * 1);
  auto& monitoring = mRegistry.get<Monitoring>();
  monitoring.send({(int)numInputTypes, "output_matchers/h", Verbosity::Debug});
  monitoring.send({(int)1, "output_matchers/w", Verbosity::Debug});
  auto& routes = mSpec.outputs;
  for (size_t i = 0; i < numInputTypes; ++i) {
    mQueriesMetricsNames[i] = fmt::format("output_matchers/{}", i);
    char buffer[128];
    assert(mDistinctRoutesIndex[i] < routes.size());
    mOutputs.push_back(routes[mDistinctRoutesIndex[i]].matcher);
    DataSpecUtils::describe(buffer, 127, mOutputs.back());
    monitoring.send({std::string{buffer}, mQueriesMetricsNames[i], Verbosity::Debug});
  }
}

std::unique_ptr<FairMQMessage> DataSender::create()
{
  FairMQDevice* device = (FairMQDevice*)mContext;
  return device->NewMessage();
}

void DataSender::send(FairMQParts& parts, std::string const& channel)
{
  FairMQDevice* device = (FairMQDevice*)mContext;
  mPolicy.send(*device, parts, channel);
}

} // namespace o2::framework
