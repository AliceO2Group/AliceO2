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
#ifndef O2_FRAMEWORK_DATASENDER_H_
#define O2_FRAMEWORK_DATASENDER_H_

#include "Framework/RoutingIndices.h"
#include "Framework/SendingPolicy.h"
#include "Framework/Tracing.h"
#include "Framework/OutputSpec.h"
#include <fairmq/FairMQParts.h>
#include <string>

#include <cstddef>
#include <mutex>

namespace o2::framework
{

struct ServiceRegistry;
struct DeviceSpec;

/// Allow injecting policies on send
class DataSender
{
 public:
  DataSender(ServiceRegistry& registry,
             SendingPolicy const& policy);
  void send(FairMQParts&, ChannelIndex index);
  std::unique_ptr<FairMQMessage> create(RouteIndex index);

 private:
  FairMQDeviceProxy& mProxy;
  ServiceRegistry& mRegistry;
  DeviceSpec const& mSpec;
  std::vector<OutputSpec> mOutputs;
  SendingPolicy mPolicy;
  std::vector<size_t> mDistinctRoutesIndex;

  std::vector<std::string> mMetricsNames;
  std::vector<std::string> mVariablesMetricsNames;
  std::vector<std::string> mQueriesMetricsNames;

  TracyLockableN(std::recursive_mutex, mMutex, "data relayer mutex");
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATASENDER_H_
