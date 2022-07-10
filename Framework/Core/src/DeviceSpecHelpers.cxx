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
#include "DeviceSpecHelpers.h"
#include "ChannelSpecHelpers.h"
#include <wordexp.h>
#include <algorithm>
#include <boost/program_options.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>
#include "Framework/ChannelConfigurationPolicy.h"
#include "Framework/ChannelMatching.h"
#include "Framework/ConfigParamsHelper.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceState.h"
#include "Framework/Lifetime.h"
#include "Framework/LifetimeHelpers.h"
#include "Framework/ProcessingPolicies.h"
#include "Framework/OutputRoute.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ComputingResource.h"
#include "Framework/Logger.h"
#include "Framework/RuntimeError.h"
#include "Framework/RawDeviceService.h"
#include "ProcessingPoliciesHelpers.h"

#include "WorkflowHelpers.h"

#include <uv.h>
#include <iostream>
#include <fmt/format.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <csignal>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

namespace bpo = boost::program_options;

using namespace o2::framework;

namespace o2::framework
{

namespace detail
{
void timer_callback(uv_timer_t* handle)
{
  // We simply wake up the event loop. Nothing to be done here.
  auto* state = (DeviceState*)handle->data;
  state->loopReason |= DeviceState::TIMER_EXPIRED;
  state->loopReason |= DeviceState::DATA_INCOMING;
}

void signal_callback(uv_signal_t* handle, int)
{
  // We simply wake up the event loop. Nothing to be done here.
  auto* state = (DeviceState*)handle->data;
  state->loopReason |= DeviceState::SIGNAL_ARRIVED;
  state->loopReason |= DeviceState::DATA_INCOMING;
}
} // namespace detail

struct ExpirationHandlerHelpers {
  static RouteConfigurator::CreationConfigurator dataDrivenConfigurator()
  {
    return [](DeviceState&, ServiceRegistry&, ConfigParamRegistry const&) { return LifetimeHelpers::dataDrivenCreation(); };
  }

  static RouteConfigurator::CreationConfigurator timeDrivenConfigurator(InputSpec const& matcher)
  {
    return [matcher](DeviceState& state, ServiceRegistry&, ConfigParamRegistry const& options) {
      std::string rateName = std::string{"period-"} + matcher.binding;
      auto period = options.get<int>(rateName.c_str());
      // We create a timer to wake us up. Notice the actual
      // timeslot creation and record expiration still happens
      // in a synchronous way.
      auto* timer = (uv_timer_t*)(malloc(sizeof(uv_timer_t)));
      timer->data = &state;
      uv_timer_init(state.loop, timer);
      uv_timer_start(timer, detail::timer_callback, period / 1000, period / 1000);
      state.activeTimers.push_back(timer);

      return LifetimeHelpers::timeDrivenCreation(std::chrono::microseconds(period));
    };
  }

  static RouteConfigurator::CreationConfigurator loopEventDrivenConfigurator(InputSpec const& matcher)
  {
    return [matcher](DeviceState& state, ServiceRegistry&, ConfigParamRegistry const&) {
      return LifetimeHelpers::uvDrivenCreation(DeviceState::LoopReason::OOB_ACTIVITY, state);
    };
  }

  static RouteConfigurator::CreationConfigurator signalDrivenConfigurator(InputSpec const& matcher, size_t inputTimeslice, size_t maxInputTimeslices)
  {
    return [matcher, inputTimeslice, maxInputTimeslices](DeviceState& state, ServiceRegistry&, ConfigParamRegistry const& options) {
      std::string startName = std::string{"start-value-"} + matcher.binding;
      std::string endName = std::string{"end-value-"} + matcher.binding;
      std::string stepName = std::string{"step-value-"} + matcher.binding;
      auto start = options.get<int64_t>(startName.c_str());
      auto stop = options.get<int64_t>(endName.c_str());
      auto step = options.get<int64_t>(stepName.c_str());
      // We create a timer to wake us up. Notice the actual
      // timeslot creation and record expiration still happens
      // in a synchronous way.
      auto* sh = (uv_signal_t*)(malloc(sizeof(uv_signal_t)));
      uv_signal_init(state.loop, sh);
      sh->data = &state;
      uv_signal_start(sh, detail::signal_callback, SIGUSR1);
      state.activeSignals.push_back(sh);

      return LifetimeHelpers::enumDrivenCreation(start, stop, step, inputTimeslice, maxInputTimeslices, 1);
    };
  }

  static RouteConfigurator::CreationConfigurator oobDrivenConfigurator()
  {
    return [](DeviceState& state, ServiceRegistry&, ConfigParamRegistry const&) {
      return LifetimeHelpers::uvDrivenCreation(DeviceState::LoopReason::OOB_ACTIVITY, state);
    };
  }

  static RouteConfigurator::CreationConfigurator enumDrivenConfigurator(InputSpec const& matcher, size_t inputTimeslice, size_t maxInputTimeslices)
  {
    return [matcher, inputTimeslice, maxInputTimeslices](DeviceState&, ServiceRegistry&, ConfigParamRegistry const& options) {
      std::string startName = std::string{"start-value-"} + matcher.binding;
      std::string endName = std::string{"end-value-"} + matcher.binding;
      std::string stepName = std::string{"step-value-"} + matcher.binding;
      auto start = options.get<int64_t>(startName.c_str());
      auto stop = options.get<int64_t>(endName.c_str());
      auto step = options.get<int64_t>(stepName.c_str());
      auto repetitions = 1;
      for (auto& meta : matcher.metadata) {
        if (meta.name == "repetitions") {
          repetitions = meta.defaultValue.get<int64_t>();
          break;
        }
      }
      return LifetimeHelpers::enumDrivenCreation(start, stop, step, inputTimeslice, maxInputTimeslices, repetitions);
    };
  }

  static RouteConfigurator::DanglingConfigurator danglingTimeframeConfigurator()
  {
    return [](DeviceState&, ConfigParamRegistry const&) { return LifetimeHelpers::expireNever(); };
  }

  static RouteConfigurator::ExpirationConfigurator expiringTimeframeConfigurator()
  {
    return [](DeviceState&, ConfigParamRegistry const&) { return LifetimeHelpers::doNothing(); };
  }

  static RouteConfigurator::DanglingConfigurator danglingConditionConfigurator()
  {
    return [](DeviceState&, ConfigParamRegistry const& options) {
      auto serverUrl = options.get<std::string>("condition-backend");
      return LifetimeHelpers::expectCTP(serverUrl, true);
    };
  }

  static RouteConfigurator::ExpirationConfigurator expiringConditionConfigurator(InputSpec const& spec, std::string const& sourceChannel)
  {
    return [spec, sourceChannel](DeviceState&, ConfigParamRegistry const& options) {
      auto serverUrl = options.get<std::string>("condition-backend");
      auto forceTimestamp = options.get<std::string>("condition-timestamp");
      return LifetimeHelpers::fetchFromCCDBCache(spec, serverUrl, forceTimestamp, sourceChannel);
    };
  }

  static RouteConfigurator::CreationConfigurator fairmqDrivenConfiguration(InputSpec const& spec, int inputTimeslice, int maxInputTimeslices)
  {
    return [spec, inputTimeslice, maxInputTimeslices](DeviceState& state, ServiceRegistry& services, ConfigParamRegistry const&) {
      // std::string channelNameOption = std::string{"out-of-band-channel-name-"} + spec.binding;
      // auto channelName = options.get<std::string>(channelNameOption.c_str());
      std::string channelName = "upstream";
      for (auto& meta : spec.metadata) {
        if (meta.name != "channel-name") {
          continue;
        }
        channelName = meta.defaultValue.get<std::string>();
      }

      auto device = services.get<RawDeviceService>().device();
      auto& channel = device->fChannels[channelName];

      // We assume there is always a ZeroMQ socket behind.
      int zmq_fd = 0;
      size_t zmq_fd_len = sizeof(zmq_fd);
      auto* poller = (uv_poll_t*)malloc(sizeof(uv_poll_t));
      channel[0].GetSocket().GetOption("fd", &zmq_fd, &zmq_fd_len);
      if (zmq_fd == 0) {
        throw runtime_error_f("Cannot get file descriptor for channel %s", channelName.c_str());
      }
      LOG(debug) << "Polling socket for " << channel[0].GetName();

      state.activeOutOfBandPollers.push_back(poller);

      // We always create entries whenever we get invoked.
      // Notice this works only if we are the only input.
      // Otherwise we should check the channel for new data,
      // before we create an entry.
      return LifetimeHelpers::enumDrivenCreation(0, -1, 1, inputTimeslice, maxInputTimeslices, 1);
    };
  }

  static RouteConfigurator::DanglingConfigurator danglingOutOfBandConfigurator()
  {
    return [](DeviceState&, ConfigParamRegistry const&) {
      return LifetimeHelpers::expireAlways();
    };
  }

  static RouteConfigurator::ExpirationConfigurator expiringOutOfBandConfigurator(InputSpec const& spec)
  {
    return [spec](DeviceState&, ConfigParamRegistry const& options) {
      std::string channelNameOption = std::string{"out-of-band-channel-name-"} + spec.binding;
      auto channelName = options.get<std::string>(channelNameOption.c_str());
      return LifetimeHelpers::fetchFromFairMQ(spec, channelName);
    };
  }

  static RouteConfigurator::DanglingConfigurator danglingQAConfigurator()
  {
    // FIXME: this should really be expireAlways. However, since we do not have
    //        a proper backend for conditions yet, I keep it behaving like it was
    //        before.
    return [](DeviceState&, ConfigParamRegistry const&) { return LifetimeHelpers::expireNever(); };
  }

  static RouteConfigurator::ExpirationConfigurator expiringQAConfigurator()
  {
    return [](DeviceState&, ConfigParamRegistry const&) { return LifetimeHelpers::fetchFromQARegistry(); };
  }

  static RouteConfigurator::DanglingConfigurator danglingTimerConfigurator(InputSpec const& matcher)
  {
    return [matcher](DeviceState&, ConfigParamRegistry const&) {
      return LifetimeHelpers::expireAlways();
    };
  }

  static RouteConfigurator::DanglingConfigurator danglingEnumerationConfigurator(InputSpec const& matcher)
  {
    return [matcher](DeviceState&, ConfigParamRegistry const&) {
      return LifetimeHelpers::expireAlways();
    };
  }

  static RouteConfigurator::ExpirationConfigurator expiringTimerConfigurator(InputSpec const& spec, std::string const& sourceChannel)
  {
    auto m = std::get_if<ConcreteDataMatcher>(&spec.matcher);
    if (m == nullptr) {
      throw runtime_error("InputSpec for Timers must be fully qualified");
    }
    // We copy the matcher to avoid lifetime issues.
    return [matcher = *m, sourceChannel](DeviceState&, ConfigParamRegistry const&) {
      // Timers do not have any orbit associated to them
      return LifetimeHelpers::enumerate(matcher, sourceChannel, 0, 0);
    };
  }

  static RouteConfigurator::ExpirationConfigurator expiringOOBConfigurator(InputSpec const& spec, std::string const& sourceChannel)
  {
    auto m = std::get_if<ConcreteDataMatcher>(&spec.matcher);
    if (m == nullptr) {
      throw runtime_error("InputSpec for OOB must be fully qualified");
    }
    // We copy the matcher to avoid lifetime issues.
    return [matcher = *m, sourceChannel](DeviceState&, ConfigParamRegistry const&) {
      // Timers do not have any orbit associated to them
      return LifetimeHelpers::enumerate(matcher, sourceChannel, 0, 0);
    };
  }

  static RouteConfigurator::ExpirationConfigurator expiringEnumerationConfigurator(InputSpec const& spec, std::string const& sourceChannel)
  {
    auto m = std::get_if<ConcreteDataMatcher>(&spec.matcher);
    if (m == nullptr) {
      throw runtime_error("InputSpec for Enumeration must be fully qualified");
    }
    // We copy the matcher to avoid lifetime issues.
    return [matcher = *m, sourceChannel](DeviceState&, ConfigParamRegistry const& config) {
      size_t orbitOffset = config.get<int64_t>("orbit-offset-enumeration");
      size_t orbitMultiplier = config.get<int64_t>("orbit-multiplier-enumeration");
      return LifetimeHelpers::enumerate(matcher, sourceChannel, orbitOffset, orbitMultiplier);
    };
  }

  static RouteConfigurator::DanglingConfigurator danglingTransientConfigurator()
  {
    // FIXME: this should really be expireAlways. However, since we do not have
    //        a proper backend for conditions yet, I keep it behaving like it was
    //        before.
    return [](DeviceState&, ConfigParamRegistry const&) { return LifetimeHelpers::expireNever(); };
  }

  static RouteConfigurator::ExpirationConfigurator expiringTransientConfigurator(InputSpec const&)
  {
    return [](DeviceState&, ConfigParamRegistry const&) { return LifetimeHelpers::fetchFromObjectRegistry(); };
  }

  /// This behaves as data. I.e. we never create it unless data arrives.
  static RouteConfigurator::CreationConfigurator createOptionalConfigurator()
  {
    return [](DeviceState&, ServiceRegistry&, ConfigParamRegistry const&) { return LifetimeHelpers::dataDrivenCreation(); };
  }

  /// This will always exipire an optional record when no data is received.
  static RouteConfigurator::DanglingConfigurator danglingOptionalConfigurator(std::vector<InputRoute> const& routes)
  {
    return [routes](DeviceState&, ConfigParamRegistry const&) { return LifetimeHelpers::expireIfPresent(routes, ConcreteDataMatcher{"FLP", "DISTSUBTIMEFRAME", 0}); };
  }

  /// When the record expires, simply create a dummy entry.
  static RouteConfigurator::ExpirationConfigurator expiringOptionalConfigurator(InputSpec const& spec, std::string const& sourceChannel)
  {
    try {
      ConcreteDataMatcher concrete = DataSpecUtils::asConcreteDataMatcher(spec);
      return [concrete, sourceChannel](DeviceState&, ConfigParamRegistry const&) {
        return LifetimeHelpers::dummy(concrete, sourceChannel);
      };
    } catch (...) {
      ConcreteDataTypeMatcher dataType = DataSpecUtils::asConcreteDataTypeMatcher(spec);
      ConcreteDataMatcher concrete{dataType.origin, dataType.description, 0xdeadbeef};
      return [concrete, sourceChannel](DeviceState&, ConfigParamRegistry const&) {
        return LifetimeHelpers::dummy(concrete, sourceChannel);
      };
      // We copy the matcher to avoid lifetime issues.
    }
  }
};

/// This creates a string to configure channels of a fair::mq::Device
/// FIXME: support shared memory
std::string DeviceSpecHelpers::inputChannel2String(const InputChannelSpec& channel)
{
  return fmt::format("{}type={},method={},address={},rateLogging={},rcvBufSize={},sndBufSize={}",
                     channel.name.empty() ? "" : "name=" + channel.name + ",",
                     ChannelSpecHelpers::typeAsString(channel.type),
                     ChannelSpecHelpers::methodAsString(channel.method),
                     ChannelSpecHelpers::channelUrl(channel),
                     channel.rateLogging,
                     channel.recvBufferSize,
                     channel.sendBufferSize);
}

std::string DeviceSpecHelpers::outputChannel2String(const OutputChannelSpec& channel)
{
  return fmt::format("{}type={},method={},address={},rateLogging={},rcvBufSize={},sndBufSize={}",
                     channel.name.empty() ? "" : "name=" + channel.name + ",",
                     ChannelSpecHelpers::typeAsString(channel.type),
                     ChannelSpecHelpers::methodAsString(channel.method),
                     ChannelSpecHelpers::channelUrl(channel),
                     channel.rateLogging,
                     channel.recvBufferSize,
                     channel.sendBufferSize);
}

void DeviceSpecHelpers::processOutEdgeActions(std::vector<DeviceSpec>& devices,
                                              std::vector<DeviceId>& deviceIndex,
                                              std::vector<DeviceConnectionId>& connections,
                                              ResourceManager& resourceManager,
                                              const std::vector<size_t>& outEdgeIndex,
                                              const std::vector<DeviceConnectionEdge>& logicalEdges,
                                              const std::vector<EdgeAction>& actions, const WorkflowSpec& workflow,
                                              const std::vector<OutputSpec>& outputsMatchers,
                                              const std::vector<ChannelConfigurationPolicy>& channelPolicies,
                                              std::string const& channelPrefix,
                                              ComputingOffer const& defaultOffer,
                                              OverrideServiceSpecs const& overrideServices)
{
  // The topology cannot be empty or not connected. If that is the case, than
  // something before this went wrong.
  // FIXME: is that really true???
  assert(!workflow.empty());

  // Edges are navigated in order for each device, so the device associaited to
  // an edge is always the last one created.
  auto deviceForEdge = [&actions, &workflow, &devices,
                        &logicalEdges, &resourceManager,
                        &defaultOffer, &channelPrefix, overrideServices](size_t ei, ComputingOffer& acceptedOffer) {
    auto& edge = logicalEdges[ei];
    auto& action = actions[ei];

    if (action.requiresNewDevice == false) {
      assert(devices.empty() == false);
      return devices.size() - 1;
    }
    if (acceptedOffer.hostname != "") {
      resourceManager.notifyAcceptedOffer(acceptedOffer);
    }

    auto processor = workflow[edge.producer];

    acceptedOffer.cpu = defaultOffer.cpu;
    acceptedOffer.memory = defaultOffer.memory;
    for (auto offer : resourceManager.getAvailableOffers()) {
      if (offer.cpu < acceptedOffer.cpu) {
        continue;
      }
      if (offer.memory < acceptedOffer.memory) {
        continue;
      }
      acceptedOffer.hostname = offer.hostname;
      acceptedOffer.startPort = offer.startPort;
      acceptedOffer.rangeSize = 0;
      break;
    }

    DeviceSpec device;
    device.name = processor.name;
    device.id = processor.name;
    device.channelPrefix = channelPrefix;
    if (processor.maxInputTimeslices != 1) {
      device.id = processor.name + "_t" + std::to_string(edge.producerTimeIndex);
    }
    device.algorithm = processor.algorithm;
    device.services = ServiceSpecHelpers::filterDisabled(processor.requiredServices, overrideServices);
    device.options = processor.options;
    device.rank = processor.rank;
    device.nSlots = processor.nSlots;
    device.inputTimesliceId = edge.producerTimeIndex;
    device.maxInputTimeslices = processor.maxInputTimeslices;
    device.resource = {acceptedOffer};
    device.labels = processor.labels;
    /// If any of the inputs or outputs are "Lifetime::OutOfBand"
    /// create the associated channels.
    //
    // for (auto& input : processor.inputs) {
    //  if (input.lifetime != Lifetime::OutOfBand) {
    //    continue;
    //  }
    //  InputChannelSpec extraInputChannelSpec{
    //    .name = "upstream",
    //    .type = ChannelType::Pair,
    //    .method = ChannelMethod::Bind,
    //    .hostname = "localhost",
    //    .port = 33000,
    //    .protocol = ChannelProtocol::IPC,
    //  };
    //  for (auto& meta : input.metadata) {
    //    if (meta.name == "name") {
    //      extraInputChannelSpec.name = meta.defaultValue.get<std::string>();
    //    }
    //    if (meta.name == "port") {
    //      extraInputChannelSpec.port = meta.defaultValue.get<int32_t>();
    //    }
    //    if (meta.name == "address") {
    //      extraInputChannelSpec.hostname = meta.defaultValue.get<std::string>();
    //    }
    //  }
    //  device.inputChannels.push_back(extraInputChannelSpec);
    //}
    for (auto& output : processor.outputs) {
      if (output.lifetime != Lifetime::OutOfBand) {
        continue;
      }
      OutputChannelSpec extraOutputChannelSpec{
        .name = "downstream",
        .type = ChannelType::Pair,
        .method = ChannelMethod::Connect,
        .hostname = "localhost",
        .port = 33000,
        .protocol = ChannelProtocol::IPC};
      for (auto& meta : output.metadata) {
        if (meta.name == "channel-name") {
          extraOutputChannelSpec.name = meta.defaultValue.get<std::string>();
        }
        if (meta.name == "port") {
          extraOutputChannelSpec.port = meta.defaultValue.get<int32_t>();
        }
        if (meta.name == "address") {
          extraOutputChannelSpec.hostname = meta.defaultValue.get<std::string>();
        }
      }
      device.outputChannels.push_back(extraOutputChannelSpec);
    }
    devices.push_back(device);
    return devices.size() - 1;
  };

  auto channelFromDeviceEdgeAndPort = [&connections, &workflow, &channelPolicies](const DeviceSpec& device,
                                                                                  ComputingResource& deviceResource,
                                                                                  ComputingOffer& acceptedOffer,
                                                                                  const DeviceConnectionEdge& edge) {
    OutputChannelSpec channel;
    auto& consumer = workflow[edge.consumer];
    std::string consumerDeviceId = consumer.name;
    if (consumer.maxInputTimeslices != 1) {
      consumerDeviceId += "_t" + std::to_string(edge.timeIndex);
    }
    channel.name = device.channelPrefix + "from_" + device.id + "_to_" + consumerDeviceId;
    channel.port = acceptedOffer.startPort + acceptedOffer.rangeSize;
    channel.hostname = acceptedOffer.hostname;
    deviceResource.usedPorts += 1;
    acceptedOffer.rangeSize += 1;

    for (auto& policy : channelPolicies) {
      if (policy.match(device.id, consumerDeviceId)) {
        policy.modifyOutput(channel);
        break;
      }
    }
    DeviceConnectionId id{edge.producer, edge.consumer, edge.timeIndex, edge.producerTimeIndex, channel.port};
    connections.push_back(id);
    return channel;
  };

  auto isDifferentDestinationDeviceReferredBy = [&actions](size_t ei) { return actions[ei].requiresNewChannel; };

  // This creates a new channel for a given edge, if needed. Notice that we
  // navigate edges in a per device fashion (creating those if they are not
  // alredy there) and create a new channel only if it connects two new
  // devices. Whether or not this is the case was previously computed
  // in the action.requiresNewChannel field.
  auto createChannelForDeviceEdge = [&devices, &logicalEdges, &channelFromDeviceEdgeAndPort,
                                     &deviceIndex](size_t di, size_t ei, ComputingOffer& offer) {
    auto& device = devices[di];
    auto& edge = logicalEdges[ei];

    deviceIndex.emplace_back(DeviceId{edge.producer, edge.producerTimeIndex, di});

    OutputChannelSpec channel = channelFromDeviceEdgeAndPort(device, device.resource, offer, edge);

    device.outputChannels.push_back(channel);
    return device.outputChannels.size() - 1;
  };

  // Notice how we need to behave in two different ways depending
  // whether this is a real OutputRoute or if it's a forward from
  // a previous consumer device.
  // FIXME: where do I find the InputSpec for the forward?
  auto appendOutputRouteToSourceDeviceChannel = [&outputsMatchers, &workflow, &devices, &logicalEdges](
                                                  size_t ei, size_t di, size_t ci) {
    assert(ei < logicalEdges.size());
    assert(di < devices.size());
    assert(ci < devices[di].outputChannels.size());
    auto& edge = logicalEdges[ei];
    auto& device = devices[di];
    assert(edge.consumer < workflow.size());
    auto& consumer = workflow[edge.consumer];
    auto& channel = devices[di].outputChannels[ci];
    assert(edge.outputGlobalIndex < outputsMatchers.size());

    if (edge.isForward == false) {
      OutputRoute route{
        edge.timeIndex,
        consumer.maxInputTimeslices,
        outputsMatchers[edge.outputGlobalIndex],
        channel.name};
      device.outputs.emplace_back(route);
    } else {
      ForwardRoute route{
        edge.timeIndex,
        consumer.maxInputTimeslices,
        workflow[edge.consumer].inputs[edge.consumerInputIndex],
        channel.name};
      device.forwards.emplace_back(route);
    }
  };

  auto sortDeviceIndex = [&deviceIndex]() { std::sort(deviceIndex.begin(), deviceIndex.end()); };

  auto lastChannelFor = [&devices](size_t di) {
    assert(di < devices.size());
    assert(devices[di].outputChannels.empty() == false);
    return devices[di].outputChannels.size() - 1;
  };

  //
  // OUTER LOOP
  //
  // We need to create all the channels going out of a device, and associate
  // routes to them for this reason
  // we iterate over all the edges (which are per-datatype exchanged) and
  // whenever we need to connect to a new device we create the channel. `device`
  // here refers to the source device. This loop will therefore not create the
  // devices which acts as sink, which are done in the preocessInEdgeActions
  // function.
  ComputingOffer acceptedOffer;
  for (auto edge : outEdgeIndex) {
    auto device = deviceForEdge(edge, acceptedOffer);
    size_t channel = -1;
    if (isDifferentDestinationDeviceReferredBy(edge)) {
      channel = createChannelForDeviceEdge(device, edge, acceptedOffer);
    } else {
      channel = lastChannelFor(device);
    }
    appendOutputRouteToSourceDeviceChannel(edge, device, channel);
  }
  if (std::string(acceptedOffer.hostname) != "") {
    resourceManager.notifyAcceptedOffer(acceptedOffer);
  }
  sortDeviceIndex();
}

void DeviceSpecHelpers::processInEdgeActions(std::vector<DeviceSpec>& devices,
                                             std::vector<DeviceId>& deviceIndex,
                                             const std::vector<DeviceConnectionId>& connections,
                                             ResourceManager& resourceManager,
                                             const std::vector<size_t>& inEdgeIndex,
                                             const std::vector<DeviceConnectionEdge>& logicalEdges,
                                             const std::vector<EdgeAction>& actions, const WorkflowSpec& workflow,
                                             std::vector<LogicalForwardInfo> const& availableForwardsInfo,
                                             std::vector<ChannelConfigurationPolicy> const& channelPolicies,
                                             std::string const& channelPrefix,
                                             ComputingOffer const& defaultOffer,
                                             OverrideServiceSpecs const& overrideServices)
{
  auto const& constDeviceIndex = deviceIndex;

  auto findProducerForEdge = [&logicalEdges, &constDeviceIndex](size_t ei) {
    auto& edge = logicalEdges[ei];

    DeviceId pid{edge.producer, edge.producerTimeIndex, 0};
    auto deviceIt = std::lower_bound(constDeviceIndex.cbegin(), constDeviceIndex.cend(), pid);
    // By construction producer should always be there
    assert(deviceIt != constDeviceIndex.end());
    assert(deviceIt->processorIndex == pid.processorIndex && deviceIt->timeslice == pid.timeslice);
    return deviceIt->deviceIndex;
  };

  auto findConsumerForEdge = [&logicalEdges, &constDeviceIndex](size_t ei) {
    auto& edge = logicalEdges[ei];
    if (!std::is_sorted(constDeviceIndex.cbegin(), constDeviceIndex.cend())) {
      throw o2::framework::runtime_error("Needs a sorted vector to be correct");
    }

    DeviceId pid{edge.consumer, edge.timeIndex, 0};
    auto deviceIt = std::lower_bound(constDeviceIndex.cbegin(), constDeviceIndex.cend(), pid);
    // We search for a consumer only if we know it's is already there.
    assert(deviceIt != constDeviceIndex.end());
    assert(deviceIt->processorIndex == pid.processorIndex && deviceIt->timeslice == pid.timeslice);
    return deviceIt->deviceIndex;
  };

  // Notice that to start with, consumer exists only if they also are
  // producers, so we need to create one if it does not exist.  Given this is
  // stateful, we keep an eye on what edge was last searched to make sure we
  // are not screwing up.
  //
  // Notice this is not thread safe.
  decltype(deviceIndex.begin()) lastConsumerSearch;
  size_t lastConsumerSearchEdge;
  auto hasConsumerForEdge = [&lastConsumerSearch, &lastConsumerSearchEdge, &deviceIndex,
                             &logicalEdges](size_t ei) -> int {
    auto& edge = logicalEdges[ei];
    DeviceId cid{edge.consumer, edge.timeIndex, 0};
    lastConsumerSearchEdge = ei; // This will invalidate the cache
    lastConsumerSearch = std::lower_bound(deviceIndex.begin(), deviceIndex.end(), cid);
    return lastConsumerSearch != deviceIndex.end() && cid.processorIndex == lastConsumerSearch->processorIndex &&
           cid.timeslice == lastConsumerSearch->timeslice;
  };

  // The passed argument is there just to check. We do know that the last searched
  // is the one we want.
  auto getConsumerForEdge = [&lastConsumerSearch, &lastConsumerSearchEdge](size_t ei) {
    assert(ei == lastConsumerSearchEdge);
    return lastConsumerSearch->deviceIndex;
  };

  auto createNewDeviceForEdge = [&workflow, &logicalEdges, &devices,
                                 &deviceIndex, &resourceManager, &defaultOffer,
                                 &channelPrefix, &overrideServices](size_t ei, ComputingOffer& acceptedOffer) {
    auto& edge = logicalEdges[ei];

    if (acceptedOffer.hostname != "") {
      resourceManager.notifyAcceptedOffer(acceptedOffer);
    }

    auto& processor = workflow[edge.consumer];

    acceptedOffer.cpu = defaultOffer.cpu;
    acceptedOffer.memory = defaultOffer.memory;
    for (auto offer : resourceManager.getAvailableOffers()) {
      if (offer.cpu < acceptedOffer.cpu) {
        continue;
      }
      if (offer.memory < acceptedOffer.memory) {
        continue;
      }
      acceptedOffer.hostname = offer.hostname;
      acceptedOffer.startPort = offer.startPort;
      acceptedOffer.rangeSize = 0;
      break;
    }

    DeviceSpec device;
    device.name = processor.name;
    device.id = processor.name;
    device.channelPrefix = channelPrefix;
    if (processor.maxInputTimeslices != 1) {
      device.id += "_t" + std::to_string(edge.timeIndex);
    }
    device.algorithm = processor.algorithm;
    device.services = ServiceSpecHelpers::filterDisabled(processor.requiredServices, overrideServices);
    device.options = processor.options;
    device.rank = processor.rank;
    device.nSlots = processor.nSlots;
    device.inputTimesliceId = edge.timeIndex;
    device.maxInputTimeslices = processor.maxInputTimeslices;
    device.resource = {acceptedOffer};
    device.labels = processor.labels;

    // FIXME: maybe I should use an std::map in the end
    //        but this is really not performance critical
    auto id = DeviceId{edge.consumer, edge.timeIndex, devices.size()};
    devices.push_back(device);
    deviceIndex.push_back(id);
    std::sort(deviceIndex.begin(), deviceIndex.end());
    return devices.size() - 1;
  };

  // We search for a preexisting outgoing connection associated to this edge.
  // This is to retrieve the port of the source.
  // This has to exists, because we already created all the outgoing connections
  // so it's just a matter of looking it up.
  auto findMatchingOutgoingPortForEdge = [&logicalEdges, &connections](size_t ei) {
    auto const& edge = logicalEdges[ei];
    DeviceConnectionId connectionId{edge.producer, edge.consumer, edge.timeIndex, edge.producerTimeIndex, 0};

    auto it = std::lower_bound(connections.begin(), connections.end(), connectionId);

    assert(it != connections.end());
    assert(it->producer == connectionId.producer);
    assert(it->consumer == connectionId.consumer);
    assert(it->timeIndex == connectionId.timeIndex);
    assert(it->producerTimeIndex == connectionId.producerTimeIndex);
    return it->port;
  };

  auto checkNoDuplicatesFor = [](std::vector<InputChannelSpec> const& channels, const std::string& name) {
    for (auto const& channel : channels) {
      if (channel.name == name) {
        return false;
      }
    }
    return true;
  };
  auto appendInputChannelForConsumerDevice = [&devices, &checkNoDuplicatesFor, &channelPolicies](
                                               size_t pi, size_t ci, unsigned short port) {
    auto const& producerDevice = devices[pi];
    auto& consumerDevice = devices[ci];
    InputChannelSpec channel;
    channel.name = producerDevice.channelPrefix + "from_" + producerDevice.id + "_to_" + consumerDevice.id;
    channel.hostname = producerDevice.resource.hostname;
    channel.port = port;
    for (auto& policy : channelPolicies) {
      if (policy.match(producerDevice.id, consumerDevice.id)) {
        policy.modifyInput(channel);
        break;
      }
    }
    assert(checkNoDuplicatesFor(consumerDevice.inputChannels, channel.name));
    consumerDevice.inputChannels.push_back(channel);
    return consumerDevice.inputChannels.size() - 1;
  };

  // I think this is trivial, since I think it should always be the last one,
  // in case it's not actually the case, I should probably do an actual lookup
  // here.
  auto getChannelForEdge = [&devices](size_t pi, size_t ci) {
    auto& consumerDevice = devices[ci];
    return consumerDevice.inputChannels.size() - 1;
  };

  // This is always called when adding a new channel, so we can simply refer
  // to back. Notice also that this is the place where it makes sense to
  // assign the forwarding, given that the forwarded stuff comes from some
  // input.
  auto appendInputRouteToDestDeviceChannel = [&devices, &logicalEdges, &workflow](size_t ei, size_t di, size_t ci) {
    auto const& edge = logicalEdges[ei];
    auto const& consumer = workflow[edge.consumer];
    auto& consumerDevice = devices[di];

    auto const& inputSpec = consumer.inputs[edge.consumerInputIndex];
    auto const& sourceChannel = consumerDevice.inputChannels[ci].name;

    InputRoute route{
      inputSpec,
      edge.consumerInputIndex,
      sourceChannel,
      edge.producerTimeIndex,
      std::nullopt};

    // In case we have wildcards, we must make sure that some other edge
    // produced the same route, i.e. has the same matcher.  Without this,
    // otherwise, we would end up with as many input routes as the outputs that
    // can be matched by the wildcard.
    for (size_t iri = 0; iri < consumerDevice.inputs.size(); ++iri) {
      auto& existingRoute = consumerDevice.inputs[iri];
      if (existingRoute.timeslice != edge.producerTimeIndex) {
        continue;
      }
      if (existingRoute.inputSpecIndex == edge.consumerInputIndex) {
        return;
      }
    }

    consumerDevice.inputs.push_back(route);
  };

  // Outer loop. A new device is needed for each
  // of the sink data processors.
  // New InputChannels need to refer to preexisting OutputChannels we create
  // previously.
  ComputingOffer acceptedOffer;
  for (size_t edge : inEdgeIndex) {
    auto& action = actions[edge];

    size_t consumerDevice = -1;

    if (action.requiresNewDevice) {
      if (hasConsumerForEdge(edge)) {
        consumerDevice = getConsumerForEdge(edge);
      } else {
        consumerDevice = createNewDeviceForEdge(edge, acceptedOffer);
      }
    } else {
      consumerDevice = findConsumerForEdge(edge);
    }
    size_t producerDevice = findProducerForEdge(edge);

    size_t channel = -1;
    if (action.requiresNewChannel) {
      int16_t port = findMatchingOutgoingPortForEdge(edge);
      channel = appendInputChannelForConsumerDevice(producerDevice, consumerDevice, port);
    } else {
      channel = getChannelForEdge(producerDevice, consumerDevice);
    }
    appendInputRouteToDestDeviceChannel(edge, consumerDevice, channel);
  }

  // Bind the expiration mechanism to the input routes
  for (auto& device : devices) {
    for (auto& route : device.inputs) {
      switch (route.matcher.lifetime) {
        case Lifetime::OutOfBand:
          route.configurator = {
            .name = "oob",
            .creatorConfigurator = ExpirationHandlerHelpers::loopEventDrivenConfigurator(route.matcher),
            .danglingConfigurator = ExpirationHandlerHelpers::danglingOutOfBandConfigurator(),
            .expirationConfigurator = ExpirationHandlerHelpers::expiringOOBConfigurator(route.matcher, route.sourceChannel)};
          break;
          //      case Lifetime::Condition:
          //        route.configurator = {
          //          ExpirationHandlerHelpers::dataDrivenConfigurator(),
          //          ExpirationHandlerHelpers::danglingConditionConfigurator(),
          //          ExpirationHandlerHelpers::expiringConditionConfigurator(inputSpec, sourceChannel)};
          //        break;
        case Lifetime::QA:
          route.configurator = {
            .name = "qa",
            .creatorConfigurator = ExpirationHandlerHelpers::dataDrivenConfigurator(),
            .danglingConfigurator = ExpirationHandlerHelpers::danglingQAConfigurator(),
            .expirationConfigurator = ExpirationHandlerHelpers::expiringQAConfigurator()};
          break;
        case Lifetime::Timer:
          route.configurator = {
            .name = "timer",
            .creatorConfigurator = ExpirationHandlerHelpers::timeDrivenConfigurator(route.matcher),
            .danglingConfigurator = ExpirationHandlerHelpers::danglingTimerConfigurator(route.matcher),
            .expirationConfigurator = ExpirationHandlerHelpers::expiringTimerConfigurator(route.matcher, route.sourceChannel)};
          break;
        case Lifetime::Enumeration:
          route.configurator = {
            .name = "enumeration",
            .creatorConfigurator = ExpirationHandlerHelpers::enumDrivenConfigurator(route.matcher, device.inputTimesliceId, device.maxInputTimeslices),
            .danglingConfigurator = ExpirationHandlerHelpers::danglingEnumerationConfigurator(route.matcher),
            .expirationConfigurator = ExpirationHandlerHelpers::expiringEnumerationConfigurator(route.matcher, route.sourceChannel)};
          break;
        case Lifetime::Signal:
          route.configurator = {
            .name = "signal",
            .creatorConfigurator = ExpirationHandlerHelpers::signalDrivenConfigurator(route.matcher, device.inputTimesliceId, device.maxInputTimeslices),
            .danglingConfigurator = ExpirationHandlerHelpers::danglingEnumerationConfigurator(route.matcher),
            .expirationConfigurator = ExpirationHandlerHelpers::expiringEnumerationConfigurator(route.matcher, route.sourceChannel)};
          break;
        case Lifetime::Transient:
          route.configurator = {
            .name = "transient",
            .creatorConfigurator = ExpirationHandlerHelpers::dataDrivenConfigurator(),
            .danglingConfigurator = ExpirationHandlerHelpers::danglingTransientConfigurator(),
            .expirationConfigurator = ExpirationHandlerHelpers::expiringTransientConfigurator(route.matcher)};
          break;
        case Lifetime::Optional:
          route.configurator = {
            .name = "optional",
            .creatorConfigurator = ExpirationHandlerHelpers::createOptionalConfigurator(),
            .danglingConfigurator = ExpirationHandlerHelpers::danglingOptionalConfigurator(device.inputs),
            .expirationConfigurator = ExpirationHandlerHelpers::expiringOptionalConfigurator(route.matcher, route.sourceChannel)};
          break;
        default:
          break;
      }
    }
  }

  if (acceptedOffer.hostname != "") {
    resourceManager.notifyAcceptedOffer(acceptedOffer);
  }
}

// Construct the list of actual devices we want, given a workflow.
//
// FIXME: make start port configurable?
void DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(const WorkflowSpec& workflow,
                                                       std::vector<ChannelConfigurationPolicy> const& channelPolicies,
                                                       std::vector<CompletionPolicy> const& completionPolicies,
                                                       std::vector<DispatchPolicy> const& dispatchPolicies,
                                                       std::vector<ResourcePolicy> const& resourcePolicies,
                                                       std::vector<CallbacksPolicy> const& callbacksPolicies,
                                                       std::vector<SendingPolicy> const& sendingPolicies,
                                                       std::vector<DeviceSpec>& devices,
                                                       ResourceManager& resourceManager,
                                                       std::string const& uniqueWorkflowId,
                                                       ConfigContext const& configContext,
                                                       bool optimizeTopology,
                                                       unsigned short resourcesMonitoringInterval,
                                                       std::string const& channelPrefix,
                                                       OverrideServiceSpecs const& overrideServices)
{
  std::vector<LogicalForwardInfo> availableForwardsInfo;
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<DeviceConnectionId> connections;
  std::vector<DeviceId> deviceIndex;

  // This is a temporary store for inputs and outputs,
  // including forwarded channels, so that we can construct
  // them before assigning to a device.
  std::vector<OutputSpec> outputs;

  WorkflowHelpers::constructGraph(workflow, logicalEdges, outputs, availableForwardsInfo);

  // We need to instanciate one device per (me, timeIndex) in the
  // DeviceConnectionEdge. For each device we need one new binding
  // server per (me, other) -> port Moreover for each (me, other,
  // outputGlobalIndex) we need to insert either an output or a
  // forward.
  //
  // We then sort by other. For each (other, me) we need to connect to
  // port (me, other) and add an input.

  // Fill an index to do the sorting
  std::vector<size_t> inEdgeIndex;
  std::vector<size_t> outEdgeIndex;
  WorkflowHelpers::sortEdges(inEdgeIndex, outEdgeIndex, logicalEdges);

  std::vector<EdgeAction> outActions = WorkflowHelpers::computeOutEdgeActions(logicalEdges, outEdgeIndex);
  // Crete the connections on the inverse map for all of them
  // lookup for port and add as input of the current device.
  std::vector<EdgeAction> inActions = WorkflowHelpers::computeInEdgeActions(logicalEdges, inEdgeIndex);
  size_t deviceCount = 0;
  for (auto& action : outActions) {
    deviceCount += action.requiresNewDevice ? 1 : 0;
  }
  for (auto& action : inActions) {
    deviceCount += action.requiresNewDevice ? 1 : 0;
  }

  ComputingOffer defaultOffer;
  for (auto& offer : resourceManager.getAvailableOffers()) {
    defaultOffer.cpu += offer.cpu;
    defaultOffer.memory += offer.memory;
  }

  /// For the moment lets play it safe and underestimate default needed resources.
  defaultOffer.cpu /= deviceCount + 1;
  defaultOffer.memory /= deviceCount + 1;

  processOutEdgeActions(devices, deviceIndex, connections, resourceManager, outEdgeIndex, logicalEdges,
                        outActions, workflow, outputs, channelPolicies, channelPrefix, defaultOffer, overrideServices);

  // FIXME: is this not the case???
  std::sort(connections.begin(), connections.end());

  processInEdgeActions(devices, deviceIndex, connections, resourceManager, inEdgeIndex, logicalEdges,
                       inActions, workflow, availableForwardsInfo, channelPolicies, channelPrefix, defaultOffer, overrideServices);
  // We apply the completion policies here since this is where we have all the
  // devices resolved.
  for (auto& device : devices) {
    for (auto& policy : completionPolicies) {
      if (policy.matcher(device) == true) {
        device.completionPolicy = policy;
        break;
      }
    }
    for (auto& policy : dispatchPolicies) {
      if (policy.deviceMatcher(device) == true) {
        device.dispatchPolicy = policy;
        break;
      }
    }
    for (auto& policy : callbacksPolicies) {
      if (policy.matcher(device, configContext) == true) {
        device.callbacksPolicy = policy;
        break;
      }
    }
    for (auto& policy : sendingPolicies) {
      if (policy.matcher(device, configContext) == true) {
        device.sendingPolicy = policy;
        break;
      }
    }
    bool hasPolicy = false;
    for (auto& policy : resourcePolicies) {
      if (policy.matcher(device) == true) {
        device.resourcePolicy = policy;
        hasPolicy = true;
        break;
      }
    }
    if (hasPolicy == false) {
      throw runtime_error_f("Unable to find a resource policy for %s", device.id.c_str());
    }
  }

  for (auto& device : devices) {
    device.resourceMonitoringInterval = resourcesMonitoringInterval;
  }

  auto findDeviceIndex = [&deviceIndex](size_t processorIndex, size_t timeslice) {
    for (auto& deviceEdge : deviceIndex) {
      if (deviceEdge.processorIndex != processorIndex) {
        continue;
      }
      if (deviceEdge.timeslice != timeslice) {
        continue;
      }
      return deviceEdge.deviceIndex;
    }
    throw runtime_error("Unable to find device.");
  };

  // Optimize the topology when two devices are
  // running on the same node.
  if (optimizeTopology) {
    for (auto& connection : connections) {
      auto& device1 = devices[findDeviceIndex(connection.consumer, connection.timeIndex)];
      auto& device2 = devices[findDeviceIndex(connection.producer, connection.producerTimeIndex)];
      // No need to do anything if they are not on the same host
      if (device1.resource.hostname != device2.resource.hostname) {
        continue;
      }
      for (auto& input : device1.inputChannels) {
        for (auto& output : device2.outputChannels) {
          if (input.hostname == output.hostname && input.port == output.port) {
            input.protocol = ChannelProtocol::IPC;
            output.protocol = ChannelProtocol::IPC;
            input.hostname += uniqueWorkflowId;
            output.hostname += uniqueWorkflowId;
          }
        }
      }
    }
  }
}

void DeviceSpecHelpers::reworkHomogeneousOption(std::vector<DataProcessorInfo>& infos, char const* name, char const* defaultValue)
{
  std::string finalValue;
  for (auto& info : infos) {
    auto it = std::find(info.cmdLineArgs.begin(), info.cmdLineArgs.end(), name);
    if (it == info.cmdLineArgs.end()) {
      continue;
    }
    auto value = it + 1;
    if (value == info.cmdLineArgs.end()) {
      throw runtime_error_f("%s requires an argument", name);
    }
    if (!finalValue.empty() && finalValue != *value) {
      throw runtime_error_f("Found incompatible %s values: %s amd %s", name, finalValue.c_str(), value->c_str());
    }
    finalValue = *value;
    info.cmdLineArgs.erase(it, it + 2);
  }
  if (finalValue.empty() && defaultValue == nullptr) {
    return;
  }
  if (finalValue.empty()) {
    finalValue = defaultValue;
  }
  for (auto& info : infos) {
    info.cmdLineArgs.push_back(name);
    info.cmdLineArgs.push_back(finalValue);
  }
}

void DeviceSpecHelpers::reworkIntegerOption(std::vector<DataProcessorInfo>& infos, char const* name, std::function<long long()> defaultValueCallback, long long startValue, std::function<long long(long long, long long)> bestValue)
{
  int64_t finalValue = startValue;
  bool wasModified = false;
  for (auto& info : infos) {
    auto it = std::find(info.cmdLineArgs.begin(), info.cmdLineArgs.end(), name);
    if (it == info.cmdLineArgs.end()) {
      continue;
    }
    auto valueS = it + 1;
    if (valueS == info.cmdLineArgs.end()) {
      throw runtime_error_f("%s requires an integer argument", name);
    }
    char* err = nullptr;
    long long value = strtoll(valueS->c_str(), &err, 10);
    finalValue = bestValue(value, finalValue);
    wasModified = true;
    info.cmdLineArgs.erase(it, it + 2);
  }
  if (!wasModified && defaultValueCallback == nullptr) {
    return;
  }
  if (!wasModified) {
    finalValue = defaultValueCallback();
  }
  for (auto& info : infos) {
    info.cmdLineArgs.push_back(name);
    info.cmdLineArgs.push_back(std::to_string(finalValue));
  }
}

void DeviceSpecHelpers::reworkShmSegmentSize(std::vector<DataProcessorInfo>& infos)
{
  int64_t segmentSize = 0;
  for (auto& info : infos) {
    auto it = std::find(info.cmdLineArgs.begin(), info.cmdLineArgs.end(), "--shm-segment-size");
    if (it == info.cmdLineArgs.end()) {
      continue;
    }
    auto value = it + 1;
    if (value == info.cmdLineArgs.end()) {
      throw runtime_error("--shm-segment-size requires an argument");
    }
    char* err = nullptr;
    int64_t size = strtoll(value->c_str(), &err, 10);
    if (size > segmentSize) {
      segmentSize = size;
    }
    info.cmdLineArgs.erase(it, it + 2);
  }
  /// If no segment size is set, make it max VSIZE - 1GB or 90% max VSIZE.
  if (segmentSize == 0) {
    struct rlimit limits;
    getrlimit(RLIMIT_AS, &limits);
    if (limits.rlim_cur != RLIM_INFINITY) {
      segmentSize = std::min(limits.rlim_cur - 1000000000LL, (limits.rlim_cur * 90LL) / 100LL);
    }
  }
  if (segmentSize == 0) {
    segmentSize = 2000000000LL;
  }
  for (auto& info : infos) {
    info.cmdLineArgs.push_back("--shm-segment-size");
    info.cmdLineArgs.push_back(std::to_string(segmentSize));
  }
}

namespace
{
template <class Container>
void split(const std::string& str, Container& cont)
{
  std::istringstream iss(str);
  std::copy(std::istream_iterator<std::string>(iss),
            std::istream_iterator<std::string>(),
            std::back_inserter(cont));
}
} // namespace

void DeviceSpecHelpers::prepareArguments(bool defaultQuiet, bool defaultStopped, bool interactive,
                                         unsigned short driverPort,
                                         std::vector<DataProcessorInfo> const& processorInfos,
                                         std::vector<DeviceSpec> const& deviceSpecs,
                                         std::vector<DeviceExecution>& deviceExecutions,
                                         std::vector<DeviceControl>& deviceControls,
                                         std::string const& uniqueWorkflowId)
{
  assert(deviceSpecs.size() == deviceExecutions.size());
  assert(deviceControls.size() == deviceExecutions.size());
  for (size_t si = 0; si < deviceSpecs.size(); ++si) {
    auto& spec = deviceSpecs[si];
    auto& control = deviceControls[si];
    auto& execution = deviceExecutions[si];

    control.quiet = defaultQuiet;
    control.stopped = defaultStopped;

    int argc;
    char** argv;
    std::vector<ConfigParamSpec> workflowOptions;
    /// Lookup the executable name in the metadata associated with the workflow.
    /// If we find it, we rewrite the command line arguments to be processed
    /// so that they look like the ones passed to the merged workflow.
    auto pi = std::find_if(processorInfos.begin(), processorInfos.end(), [&](auto const& x) { return x.name == spec.id; });
    argc = pi->cmdLineArgs.size() + 1;
    argv = (char**)malloc(sizeof(char**) * (argc + 1));
    argv[0] = strdup(pi->executable.data());
    for (size_t ai = 0; ai < pi->cmdLineArgs.size(); ++ai) {
      auto const& arg = pi->cmdLineArgs[ai];
      argv[ai + 1] = strdup(arg.data());
    }
    argv[argc] = nullptr;
    workflowOptions = pi->workflowOptions;

    // We duplicate the list of options, filtering only those
    // which are actually relevant for the given device. The additional
    // four are to add
    // * name of the executable
    // * --framework-id <id> so that we can use the workflow
    //   executable also in other context where we do not fork, e.g. DDS.
    // * final NULL required by execvp
    //
    // We do it here because we are still in the parent and we can therefore
    // capture them to be displayed in the GUI or to populate the DDS configuration
    // to dump

    // Set up options for the device running underneath
    // FIXME: add some checksum in framework id. We could use this
    //        to avoid redeploys when only a portion of the workflow is changed.
    // FIXME: this should probably be done in one go with char *, but I am lazy.
    std::vector<std::string> tmpArgs = {argv[0],
                                        "--id", spec.id.c_str(),
                                        "--control", interactive ? "gui" : "static",
                                        "--shm-monitor", "false",
                                        "--log-color", "false",
                                        "--color", "false"};

    // we maintain options in a map so that later occurrences of the same
    // option will overwrite the value. To make unit tests work on all platforms,
    // we need to make the sequence deterministic and store it in a separate vector
    std::vector<std::string> deviceOptionsSequence;
    std::unordered_map<std::string, std::string> uniqueDeviceArgs;
    auto updateDeviceArguments = [&deviceOptionsSequence, &uniqueDeviceArgs](auto key, auto value) {
      if (uniqueDeviceArgs.find(key) == uniqueDeviceArgs.end()) {
        // not yet existing, we add the key to the sequence
        deviceOptionsSequence.emplace_back(key);
      }
      uniqueDeviceArgs[key] = value;
    };
    std::vector<std::string> tmpEnv;
    if (defaultStopped) {
      tmpArgs.emplace_back("-s");
    }

    // do the filtering of options:
    // 1) forward options belonging to this specific DeviceSpec
    // 2) global options defined in getForwardedDeviceOptions and workflow option are
    //    always forwarded and need to be handled separately
    const char* name = spec.name.c_str();
    bpo::options_description od;     // option descriptions per process
    bpo::options_description foDesc; // forwarded options for all processes
    ConfigParamsHelper::dpl2BoostOptions(spec.options, od);
    od.add_options()(name, bpo::value<std::string>());
    ConfigParamsHelper::dpl2BoostOptions(workflowOptions, foDesc);
    auto forwardedOptions = getForwardedDeviceOptions();
    /// Add to foDesc the options which are not already there
    foDesc.add(forwardedOptions);

    // has option --session been specified on the command line?
    bool haveSessionArg = false;
    using FilterFunctionT = std::function<void(decltype(argc), decltype(argv), decltype(od))>;
    bool useDefaultWS = true;

    // the filter function will forward command line arguments based on the option
    // definition passed to it. All options of the program option definition will be forwarded
    // if found in the argument list. If not found they will be added with the default value
    FilterFunctionT filterArgsFct = [&](int largc, char** largv, const bpo::options_description& odesc) {
      // spec contains options
      using namespace bpo::command_line_style;
      auto style = (allow_short | short_allow_adjacent | short_allow_next | allow_long | long_allow_adjacent | long_allow_next | allow_sticky | allow_dash_for_short);

      bpo::command_line_parser parser{largc, largv};
      parser.options(odesc).allow_unregistered();
      parser.style(style);
      bpo::parsed_options parsed_options = parser.run();

      bpo::variables_map varmap;
      bpo::store(parsed_options, varmap);
      if (varmap.count("environment")) {
        auto environment = varmap["environment"].as<std::string>();
        split(environment, tmpEnv);
      }

      /// Add libSegFault to the stack if provided.
      if (varmap.count("stacktrace-on-signal") && varmap["stacktrace-on-signal"].as<std::string>() != "none" && varmap["stacktrace-on-signal"].as<std::string>() != "simple") {
        char const* preload = getenv("LD_PRELOAD");
        if (preload == nullptr || strcmp(preload, "libSegFault.so") == 0) {
          tmpEnv.push_back("LD_PRELOAD=libSegFault.so");
        } else {
          tmpEnv.push_back(fmt::format("LD_PRELOAD={}:libSegFault.so", preload));
        }
        tmpEnv.push_back(fmt::format("SEGFAULT_SIGNALS={}", varmap["stacktrace-on-signal"].as<std::string>()));
      }

      // options can be grouped per processor spec, the group is entered by
      // the option created from the actual processor spec name
      // if specified, the following string is interpreted as a sequence
      // of arguments
      if (varmap.count(name) > 0) {
        // strangely enough, the first argument of the group argument string
        // is marked as defaulted by the parser and is thus ignored. not fully
        // understood but adding a dummy argument in front cures this
        auto arguments = "--unused " + varmap[name].as<std::string>();
        wordexp_t expansions;
        wordexp(arguments.c_str(), &expansions, 0);
        bpo::options_description realOdesc = odesc;
        realOdesc.add_options()("severity", bpo::value<std::string>());
        realOdesc.add_options()("child-driver", bpo::value<std::string>());
        realOdesc.add_options()("rate", bpo::value<std::string>());
        realOdesc.add_options()("exit-transition-timeout", bpo::value<std::string>());
        realOdesc.add_options()("expected-region-callbacks", bpo::value<std::string>());
        realOdesc.add_options()("timeframes-rate-limit", bpo::value<std::string>());
        realOdesc.add_options()("environment", bpo::value<std::string>());
        realOdesc.add_options()("stacktrace-on-signal", bpo::value<std::string>());
        realOdesc.add_options()("post-fork-command", bpo::value<std::string>());
        realOdesc.add_options()("shm-segment-size", bpo::value<std::string>());
        realOdesc.add_options()("shm-mlock-segment", bpo::value<std::string>());
        realOdesc.add_options()("shm-mlock-segment-on-creation", bpo::value<std::string>());
        realOdesc.add_options()("shm-zero-segment", bpo::value<std::string>());
        realOdesc.add_options()("shm-throw-bad-alloc", bpo::value<std::string>());
        realOdesc.add_options()("shm-segment-id", bpo::value<std::string>());
        realOdesc.add_options()("shm-allocation", bpo::value<std::string>());
        realOdesc.add_options()("shm-no-cleanup", bpo::value<std::string>());
        realOdesc.add_options()("shmid", bpo::value<std::string>());
        realOdesc.add_options()("shm-monitor", bpo::value<std::string>());
        realOdesc.add_options()("channel-prefix", bpo::value<std::string>());
        realOdesc.add_options()("network-interface", bpo::value<std::string>());
        realOdesc.add_options()("early-forward-policy", bpo::value<std::string>());
        realOdesc.add_options()("session", bpo::value<std::string>());
        filterArgsFct(expansions.we_wordc, expansions.we_wordv, realOdesc);
        wordfree(&expansions);
        return;
      }

      const char* child_driver_key = "child-driver";
      if (varmap.count(child_driver_key) > 0) {
        auto arguments = varmap[child_driver_key].as<std::string>();
        wordexp_t expansions;
        wordexp(arguments.c_str(), &expansions, 0);
        tmpArgs.insert(tmpArgs.begin(), expansions.we_wordv, expansions.we_wordv + expansions.we_wordc);
      }

      haveSessionArg = haveSessionArg || varmap.count("session") != 0;
      useDefaultWS = useDefaultWS && ((varmap.count("driver-client-backend") == 0) || varmap["driver-client-backend"].as<std::string>() == "ws://");

      auto processRawChannelConfig = [&tmpArgs](const std::string& conf) {
        std::stringstream ss(conf);
        std::string token;
        while (std::getline(ss, token, ';')) { // split to tokens, trim spaces and add each non-empty one with channel-config options
          token.erase(token.begin(), std::find_if(token.begin(), token.end(), [](int ch) { return !std::isspace(ch); }));
          token.erase(std::find_if(token.rbegin(), token.rend(), [](int ch) { return !std::isspace(ch); }).base(), token.end());
          if (!token.empty()) {
            tmpArgs.emplace_back("--channel-config");
            tmpArgs.emplace_back(token);
          }
        }
      };

      for (const auto varit : varmap) {
        // find the option belonging to key, add if the option has been parsed
        // and is not defaulted
        const auto* description = odesc.find_nothrow(varit.first, false);
        if (description && varmap.count(varit.first)) {
          // check the semantics of the value
          auto semantic = description->semantic();
          const char* optarg = "";
          if (semantic) {
            // the value semantics allows different properties like
            // multitoken, zero_token and composing
            // currently only the simple case is supported
            assert(semantic->min_tokens() <= 1);
            // assert(semantic->max_tokens() && semantic->min_tokens());
            if (semantic->min_tokens() > 0) {
              std::string stringRep;
              if (auto v = boost::any_cast<std::string>(&varit.second.value())) {
                stringRep = *v;
              } else if (auto v = boost::any_cast<EarlyForwardPolicy>(&varit.second.value())) {
                stringRep = fmt::format("{}", *v);
              }
              if (varit.first == "channel-config") {
                // FIXME: the parameter to channel-config can be a list of configurations separated
                // by semicolon. The individual configurations will be separated and added individually.
                // The device arguments can then contaoin multiple channel-config entries, but only
                // one for the last configuration is added to control.options
                processRawChannelConfig(stringRep);
                optarg = tmpArgs.back().c_str();
              } else {
                std::string key(fmt::format("--{}", varit.first));
                if (stringRep.length() == 0) {
                  // in order to identify options without parameter we add a string
                  // with one blank for the 'blank' parameter, it is filtered out
                  // further down and a zero-length string is added to argument list
                  stringRep = " ";
                }
                updateDeviceArguments(key, stringRep);
                optarg = uniqueDeviceArgs[key].c_str();
              }
            } else if (semantic->min_tokens() == 0 && varit.second.as<bool>()) {
              updateDeviceArguments(fmt::format("--{}", varit.first), "");
            }
          }
          control.options.insert(std::make_pair(varit.first, optarg));
        }
      }
    };

    // filter global options and workflow options independent of option groups
    filterArgsFct(argc, argv, foDesc);
    // filter device options, and handle option groups
    filterArgsFct(argc, argv, od);

    // Add the channel configuration
    for (auto& channel : spec.outputChannels) {
      tmpArgs.emplace_back(std::string("--channel-config"));
      tmpArgs.emplace_back(outputChannel2String(channel));
    }
    for (auto& channel : spec.inputChannels) {
      tmpArgs.emplace_back(std::string("--channel-config"));
      tmpArgs.emplace_back(inputChannel2String(channel));
    }

    // add the session id if not already specified on command line
    if (!haveSessionArg) {
      updateDeviceArguments(std::string("--session"), "dpl_" + uniqueWorkflowId);
    }
    // In case we use only ws://, we need to expand the address
    // with the correct port.
    if (useDefaultWS) {
      updateDeviceArguments(std::string("--driver-client-backend"), "ws://0.0.0.0:" + std::to_string(driverPort));
    }

    if (spec.resourceMonitoringInterval > 0) {
      updateDeviceArguments(std::string("--resources-monitoring"), std::to_string(spec.resourceMonitoringInterval));
    }

    // We create the final option list, depending on the channels
    // which are present in a device.
    for (auto& arg : tmpArgs) {
      execution.args.emplace_back(strdup(arg.c_str()));
    }
    for (auto& key : deviceOptionsSequence) {
      execution.args.emplace_back(strdup(key.c_str()));
      std::string const& value = uniqueDeviceArgs[key];
      if (value.empty()) {
        // this option does not have a parameter
        continue;
      } else if (value == " ") {
        // this was a placeholder for zero-length parameter string in order
        // to separate this from options without parameter
        execution.args.emplace_back(strdup(""));
      } else {
        execution.args.emplace_back(strdup(value.c_str()));
      }
    }
    // execvp wants a NULL terminated list.
    execution.args.push_back(nullptr);

    for (auto& env : tmpEnv) {
      execution.environ.emplace_back(strdup(env.c_str()));
    }

    // FIXME: this should probably be reflected in the GUI
    std::ostringstream str;
    for (size_t ai = 0; ai < execution.args.size() - 1; ai++) {
      if (execution.args[ai] == nullptr) {
        LOG(error) << "Bad argument for " << execution.args[ai - 1];
      }
      assert(execution.args[ai]);
      str << " " << execution.args[ai];
    }
    LOG(debug) << "The following options are being forwarded to " << spec.id << ":" << str.str();
  }
}

/// define the options which are forwarded to every child
boost::program_options::options_description DeviceSpecHelpers::getForwardedDeviceOptions()
{
  // - rate is an option of FairMQ device for ConditionalRun
  // - child-driver is not a FairMQ device option but used per device to start to process
  bpo::options_description forwardedDeviceOptions;
  forwardedDeviceOptions.add_options()                                                                                                                               //
    ("severity", bpo::value<std::string>()->default_value("info"), "severity level of the log")                                                                      //
    ("plugin,P", bpo::value<std::string>(), "FairMQ plugin list")                                                                                                    //
    ("plugin-search-path,S", bpo::value<std::string>(), "FairMQ plugins search path")                                                                                //
    ("control-port", bpo::value<std::string>(), "Utility port to be used by O2 Control")                                                                             //
    ("rate", bpo::value<std::string>(), "rate for a data source device (Hz)")                                                                                        //
    ("exit-transition-timeout", bpo::value<std::string>(), "timeout before switching to READY state")                                                                //
    ("expected-region-callbacks", bpo::value<std::string>(), "region callbacks to expect before starting")                                                           //
    ("timeframes-rate-limit", bpo::value<std::string>()->default_value("0"), "how many timeframes can be in fly")                                                    //
    ("shm-monitor", bpo::value<std::string>(), "whether to use the shared memory monitor")                                                                           //
    ("channel-prefix", bpo::value<std::string>()->default_value(""), "prefix to use for multiplexing multiple workflows in the same session")                        //
    ("shm-segment-size", bpo::value<std::string>(), "size of the shared memory segment in bytes")                                                                    //
    ("shm-mlock-segment", bpo::value<std::string>()->default_value("false"), "mlock shared memory segment")                                                          //
    ("shm-mlock-segment-on-creation", bpo::value<std::string>()->default_value("false"), "mlock shared memory segment once on creation")                             //
    ("shm-zero-segment", bpo::value<std::string>()->default_value("false"), "zero shared memory segment")                                                            //
    ("shm-throw-bad-alloc", bpo::value<std::string>()->default_value("true"), "throw if insufficient shm memory")                                                    //
    ("shm-segment-id", bpo::value<std::string>()->default_value("0"), "shm segment id")                                                                              //
    ("shm-allocation", bpo::value<std::string>()->default_value("rbtree_best_fit"), "shm allocation method")                                                         //
    ("shm-no-cleanup", bpo::value<std::string>()->default_value("false"), "no shm cleanup")                                                                          //
    ("shmid", bpo::value<std::string>(), "shmid")                                                                                                                    //
    ("environment", bpo::value<std::string>(), "comma separated list of environment variables to set for the device")                                                //
    ("stacktrace-on-signal", bpo::value<std::string>()->default_value("simple"),                                                                                     //
     "dump stacktrace on specified signal(s) (any of `all`, `segv`, `bus`, `ill`, `abrt`, `fpe`, `sys`.)"                                                            //
     "Use `simple` to dump only the main thread in a reliable way")                                                                                                  //
    ("post-fork-command", bpo::value<std::string>(), "post fork command to execute (e.g. numactl {pid}")                                                             //
    ("session", bpo::value<std::string>(), "unique label for the shared memory session")                                                                             //
    ("network-interface", bpo::value<std::string>(), "network interface to which to bind tpc fmq ports without specified address")                                   //
    ("early-forward-policy", bpo::value<EarlyForwardPolicy>()->default_value(EarlyForwardPolicy::NEVER), "when to forward early the messages: never, noraw, always") //
    ("configuration,cfg", bpo::value<std::string>(), "configuration connection string")                                                                              //
    ("driver-client-backend", bpo::value<std::string>(), "driver connection string")                                                                                 //
    ("monitoring-backend", bpo::value<std::string>(), "monitoring connection string")                                                                                //
    ("infologger-mode", bpo::value<std::string>(), "O2_INFOLOGGER_MODE override")                                                                                    //
    ("infologger-severity", bpo::value<std::string>(), "minimun FairLogger severity which goes to info logger")                                                      //
    ("dpl-tracing-flags", bpo::value<std::string>(), "pipe separated list of events to trace")                                                                       //
    ("child-driver", bpo::value<std::string>(), "external driver to start childs with (e.g. valgrind)");                                                             //

  return forwardedDeviceOptions;
}

bool DeviceSpecHelpers::hasLabel(DeviceSpec const& spec, char const* label)
{
  auto sameLabel = [other = DataProcessorLabel{{label}}](DataProcessorLabel const& label) { return label == other; };
  return std::find_if(spec.labels.begin(), spec.labels.end(), sameLabel) != spec.labels.end();
}

} // namespace o2::framework
