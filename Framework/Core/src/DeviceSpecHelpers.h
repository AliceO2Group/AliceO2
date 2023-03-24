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
#ifndef O2_FRAMEWORK_DEVICESPECHELPERS_H_
#define O2_FRAMEWORK_DEVICESPECHELPERS_H_

#include "Framework/WorkflowSpec.h"
#include "Framework/ChannelConfigurationPolicy.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ChannelSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceExecution.h"
#include "Framework/DeviceSpec.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/OutputRoute.h"
#include "Framework/DataProcessorInfo.h"
#include "Framework/ProcessingPolicies.h"
#include "ResourceManager.h"
#include "WorkflowHelpers.h"

#include <boost/program_options.hpp>

#include <vector>
#include <string>
#include <map>
#include <functional>

namespace o2::framework
{
struct InputChannelSpec;
struct OutputChannelSpec;
struct ConfigContext;

struct DeviceSpecHelpers {
  /// Helper to convert from an abstract dataflow specification, @a workflow,
  /// to an actual set of devices which will have to run.
  static void dataProcessorSpecs2DeviceSpecs(
    const WorkflowSpec& workflow,
    std::vector<ChannelConfigurationPolicy> const& channelPolicies,
    std::vector<CompletionPolicy> const& completionPolicies,
    std::vector<DispatchPolicy> const& dispatchPolicies,
    std::vector<ResourcePolicy> const& resourcePolicies,
    std::vector<CallbacksPolicy> const& callbacksPolicies,
    std::vector<SendingPolicy> const& sendingPolicy,
    std::vector<DeviceSpec>& devices,
    ResourceManager& resourceManager,
    std::string const& uniqueWorkflowId,
    ConfigContext const& configContext,
    bool optimizeTopology = false,
    unsigned short resourcesMonitoringInterval = 0,
    std::string const& channelPrefix = "",
    OverrideServiceSpecs const& overrideServices = {});

  static void validate(WorkflowSpec const& workflow);
  static void dataProcessorSpecs2DeviceSpecs(
    const WorkflowSpec& workflow,
    std::vector<ChannelConfigurationPolicy> const& channelPolicies,
    std::vector<CompletionPolicy> const& completionPolicies,
    std::vector<CallbacksPolicy> const& callbacksPolicies,
    std::vector<DeviceSpec>& devices,
    ResourceManager& resourceManager,
    std::string const& uniqueWorkflowId,
    ConfigContext const& configContext,
    bool optimizeTopology = false,
    unsigned short resourcesMonitoringInterval = 0,
    std::string const& channelPrefix = "",
    OverrideServiceSpecs const& overrideServices = {})
  {
    std::vector<DispatchPolicy> dispatchPolicies = DispatchPolicy::createDefaultPolicies();
    std::vector<ResourcePolicy> resourcePolicies = ResourcePolicy::createDefaultPolicies();
    std::vector<SendingPolicy> sendingPolicies = SendingPolicy::createDefaultPolicies();
    dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies,
                                   dispatchPolicies, resourcePolicies, callbacksPolicies,
                                   sendingPolicies, devices,
                                   resourceManager, uniqueWorkflowId, configContext, optimizeTopology,
                                   resourcesMonitoringInterval, channelPrefix, overrideServices);
  }

  /// Helper to provide the channel configuration string for an input channel
  static std::string inputChannel2String(const InputChannelSpec& channel);

  /// Helper to provide the channel configuration string for an output channel
  static std::string outputChannel2String(const OutputChannelSpec& channel);

  /// Rework a given command line option so that all the sub workflows
  /// either have the same value, or they leave it unspecified.
  /// @a infos the DataProcessorInfos to modify
  /// @a name of the option to modify, including --
  /// @a defaultValue the default value for the option. If default is nullptr, not finding the
  ///    option will not not add a default value.
  static void reworkHomogeneousOption(std::vector<DataProcessorInfo>& infos,
                                      char const* name, char const* defaultValue);

  /// Rework a given command line option so that we pick the largest value
  /// which has been specified or a default one.
  /// @a defaultValueCallback a callback which returns the default value, if nullptr, the option
  ///    will not be added.
  /// @a bestValue given to possible values of the option, return the one which should be used.
  static void reworkIntegerOption(std::vector<DataProcessorInfo>& infos,
                                  char const* name,
                                  std::function<long long()> defaultValueCallback,
                                  long long startValue,
                                  std::function<long long(long long, long long)> bestValue);
  /// Rework the infos so that they have a consistent --shm-section-size
  /// which is the maximum of the specified value.
  static void reworkShmSegmentSize(std::vector<DataProcessorInfo>& infos);
  /// Helper to prepare the arguments which will be used to
  /// start the various devices.
  static void prepareArguments(
    bool defaultQuiet,
    bool defaultStopped,
    bool intereactive,
    unsigned short driverPort,
    std::vector<DataProcessorInfo> const& processorInfos,
    std::vector<DeviceSpec> const& deviceSpecs,
    std::vector<DeviceExecution>& deviceExecutions,
    std::vector<DeviceControl>& deviceControls,
    std::string const& uniqueWorkflowId);

  /// This takes the list of preprocessed edges of a graph
  /// and creates Devices and Channels which are related
  /// to the outgoing edges i.e. those which refer
  /// to the act of producing data.
  static void processOutEdgeActions(
    std::vector<DeviceSpec>& devices,
    std::vector<DeviceId>& deviceIndex,
    std::vector<DeviceConnectionId>& connections,
    ResourceManager& resourceManager,
    const std::vector<size_t>& outEdgeIndex,
    const std::vector<DeviceConnectionEdge>& logicalEdges,
    const std::vector<EdgeAction>& actions,
    const WorkflowSpec& workflow,
    const std::vector<OutputSpec>& outputs,
    std::vector<ChannelConfigurationPolicy> const& channelPolicies,
    std::string const& channelPrefix,
    ComputingOffer const& defaultOffer,
    OverrideServiceSpecs const& overrideServices = {});

  /// This takes the list of preprocessed edges of a graph
  /// and creates Devices and Channels which are related
  /// to the incoming edges i.e. those which refer to
  /// the act of consuming data.
  static void processInEdgeActions(
    std::vector<DeviceSpec>& devices,
    std::vector<DeviceId>& deviceIndex,
    const std::vector<DeviceConnectionId>& connections,
    ResourceManager& resourceManager,
    const std::vector<size_t>& inEdgeIndex,
    const std::vector<DeviceConnectionEdge>& logicalEdges,
    const std::vector<EdgeAction>& actions,
    const WorkflowSpec& workflow,
    const std::vector<LogicalForwardInfo>& availableForwardsInfo,
    std::vector<ChannelConfigurationPolicy> const& channelPolicies,
    std::string const& channelPrefix,
    ComputingOffer const& defaultOffer,
    OverrideServiceSpecs const& overrideServices = {});

  /// return a description of all options to be forwarded to the device
  /// by default
  static boost::program_options::options_description getForwardedDeviceOptions();
  /// @return whether a give DeviceSpec @a spec has a label @a label
  static bool hasLabel(DeviceSpec const& spec, char const* label);
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_DEVICESPECHELPERS_H_
