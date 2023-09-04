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
#ifndef O2_FRAMEWORK_DEVICESPEC_H_
#define O2_FRAMEWORK_DEVICESPEC_H_

#include "CallbacksPolicy.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ComputingResource.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataProcessorLabel.h"
#include "Framework/ChannelSpec.h"
#include "Framework/ChannelInfo.h"
#include "Framework/DeviceControl.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ForwardRoute.h"
#include "Framework/InputRoute.h"
#include "Framework/OutputRoute.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/ResourcePolicy.h"
#include "Framework/SendingPolicy.h"
#include "Framework/ServiceSpec.h"

#include <vector>
#include <string>
#include <map>
#include <utility>

namespace o2::framework
{

/// Concrete description of the device which will actually run
/// a DataProcessor.
///
/// You can get a reference to this by retrieving it from the
/// ServiceRegistry. E.g.:
///
/// auto &deviceSpec = ctx.services().get<DeviceSpec const>();
struct DeviceSpec {
  DeviceSpec(const DeviceSpec&) = delete;
  DeviceSpec(DeviceSpec&&) = default;
  /// The name of the associated DataProcessorSpec
  std::string name;
  /// The id of the device, including time-pipelining and suffix
  std::string id;
  std::string channelPrefix;
  std::vector<InputChannelSpec> inputChannels;
  std::vector<OutputChannelSpec> outputChannels;
  std::vector<std::string> arguments;
  std::vector<ConfigParamSpec> options;
  std::vector<ServiceSpec> services;

  AlgorithmSpec algorithm;

  std::vector<InputRoute> inputs;
  std::vector<OutputRoute> outputs;
  std::vector<ForwardRoute> forwards;
  size_t rank;   // Id of a parallel processing I am part of
  size_t nSlots; // Total number of parallel units I am part of
  /// The time pipelining id of this particular device.
  size_t inputTimesliceId;
  /// The maximum number of time pipelining for this device.
  size_t maxInputTimeslices;
  /// The completion policy to use for this device.
  CompletionPolicy completionPolicy;
  DispatchPolicy dispatchPolicy;
  CallbacksPolicy callbacksPolicy;
  SendingPolicy sendingPolicy;
  /// Policy on when the available resources are enough to run
  /// a computation.
  ResourcePolicy resourcePolicy;
  ComputingResource resource;
  unsigned short resourceMonitoringInterval;
  std::vector<DataProcessorLabel> labels;
  std::vector<DataProcessorMetadata> metadata;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_DEVICESPEC_H_
