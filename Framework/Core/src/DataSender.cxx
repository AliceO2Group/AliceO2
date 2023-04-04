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
#include "Framework/LifetimeHelpers.h"
#include "Framework/TimesliceIndex.h"
#include "Framework/DataProcessingHelpers.h"
#include "Framework/CommonServices.h"
#include "Framework/DataProcessingContext.h"
#include "Framework/O2DataModelHelpers.h"
#include "Framework/DataProcessingStates.h"

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

DataSender::DataSender(ServiceRegistryRef registry,
                       SendingPolicy const& policy)
  : mProxy{registry.get<FairMQDeviceProxy>()},
    mRegistry{registry},
    mSpec{registry.get<DeviceSpec const>()},
    mPolicy{policy},
    mDistinctRoutesIndex{createDistinctOutputRouteIndex(mSpec.outputs)}
{
  std::scoped_lock<LockableBase(std::recursive_mutex)> lock(mMutex);

  auto numInputTypes = mDistinctRoutesIndex.size();
  auto& routes = mSpec.outputs;
  auto& states = registry.get<DataProcessingStates>();
  std::string queries = "";
  for (size_t i = 0; i < numInputTypes; ++i) {
    char buffer[128];
    assert(mDistinctRoutesIndex[i] < routes.size());
    mOutputs.push_back(routes[mDistinctRoutesIndex[i]].matcher);
    DataSpecUtils::describe(buffer, 127, mOutputs.back());
    queries += std::string_view(buffer, strlen(buffer));
    queries += ";";
  }

  auto stateId = (short)ProcessingStateId::OUTPUT_MATCHERS;
  states.registerState({.name = "output_matchers", .stateId = stateId, .sendInitialValue = true});
  states.updateState(DataProcessingStates::CommandSpec{.id = stateId, .size = (int)queries.size(), .data = queries.data()});
  states.processCommandQueue();
  /// Fill the mPresents with the outputs which are not timeframes.
  for (size_t i = 0; i < mOutputs.size(); ++i) {
    mPresentDefaults.push_back(mOutputs[i].lifetime != Lifetime::Timeframe);
  }

  /// Check if all the inputs are of kind Timeframe / Optional
  /// and that the completion policy is the default one. If not,
  /// we actually reset the mPresentDefaults to be empty, so that
  /// the check is disabled.
  for (auto& input : mSpec.inputs) {
    if (input.matcher.lifetime != Lifetime::Timeframe && input.matcher.lifetime != Lifetime::Optional) {
      LOGP(detail, "Disabling the Lifetime::timeframe check because not all the inputs are of kind Lifetime::Timeframe");
      mPresentDefaults.resize(0);
      break;
    }
  }
  if (mSpec.completionPolicy.name != "consume-all" && mSpec.completionPolicy.name != "consume-all-ordered") {
    LOGP(detail, "Disabling the Lifetime::timeframe check because the completion policy is not the default one");
    mPresentDefaults.resize(0);
  }
}

std::unique_ptr<fair::mq::Message> DataSender::create(RouteIndex routeIndex)
{
  return mProxy.getOutputTransport(routeIndex)->CreateMessage();
}

void DataSender::send(fair::mq::Parts& parts, ChannelIndex channelIndex)
{
  // In case the vector is empty, it means the check is disabled
  if (mPresentDefaults.empty() == false) {
    O2DataModelHelpers::updateMissingSporadic(parts, mOutputs, mPresent);
  }
  auto& dataProcessorContext = mRegistry.get<DataProcessorContext>();
  dataProcessorContext.preSendingMessagesCallbacks(mRegistry, parts, channelIndex);
  mPolicy.send(mProxy, parts, channelIndex, mRegistry);
}

void DataSender::reset()
{
  mPresent = mPresentDefaults;
}

void DataSender::verifyMissingSporadic() const
{
  for (auto present : mPresent) {
    if (!present) {
      LOGP(debug, O2DataModelHelpers::describeMissingOutputs(mOutputs, mPresent).c_str());
      return;
    }
  }
}

} // namespace o2::framework
