// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "Framework/Lifetime.h"
#include "Framework/LifetimeHelpers.h"
#include "Framework/OutputRoute.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ComputingResource.h"
#include "Framework/Logger.h"

#include "WorkflowHelpers.h"

namespace bpo = boost::program_options;

using namespace o2::framework;

namespace o2
{
namespace framework
{

struct ExpirationHandlerHelpers {
  static RouteConfigurator::CreationConfigurator dataDrivenConfigurator()
  {
    return [](ConfigParamRegistry const&) { return LifetimeHelpers::dataDrivenCreation(); };
  }

  static RouteConfigurator::CreationConfigurator timeDrivenConfigurator(InputSpec const& matcher)
  {
    return [matcher](ConfigParamRegistry const& options) {
      std::string rateName = std::string{"period-"} + matcher.binding;
      auto period = options.get<int>(rateName.c_str());
      return LifetimeHelpers::timeDrivenCreation(std::chrono::microseconds(period));
    };
  }

  static RouteConfigurator::CreationConfigurator enumDrivenConfigurator(InputSpec const& matcher)
  {
    return [matcher](ConfigParamRegistry const& options) {
      std::string startName = std::string{"start-value-"} + matcher.binding;
      std::string endName = std::string{"end-value-"} + matcher.binding;
      std::string stepName = std::string{"step-value-"} + matcher.binding;
      auto start = options.get<int>(startName.c_str());
      auto stop = options.get<int>(endName.c_str());
      auto step = options.get<int>(stepName.c_str());
      return LifetimeHelpers::enumDrivenCreation(start, stop, step);
    };
  }

  static RouteConfigurator::DanglingConfigurator danglingTimeframeConfigurator()
  {
    return [](ConfigParamRegistry const&) { return LifetimeHelpers::expireNever(); };
  }

  static RouteConfigurator::ExpirationConfigurator expiringTimeframeConfigurator()
  {
    return [](ConfigParamRegistry const&) { return LifetimeHelpers::doNothing(); };
  }

  static RouteConfigurator::DanglingConfigurator danglingConditionConfigurator()
  {
    return [](ConfigParamRegistry const&) { return LifetimeHelpers::expireAlways(); };
  }

  static RouteConfigurator::ExpirationConfigurator expiringConditionConfigurator(InputSpec const& spec, std::string const& sourceChannel)
  {
    /// FIXME: seems convoluted... Maybe there is a way to avoid all this checking???
    auto m = std::get_if<ConcreteDataMatcher>(&spec.matcher);
    if (m == nullptr) {
      throw std::runtime_error("InputSpec for Conditions must be fully qualified");
    }

    return [s = spec, matcher = *m, sourceChannel](ConfigParamRegistry const& options) {
      auto serverUrl = options.get<std::string>("condition-backend");
      auto timestamp = options.get<std::string>("condition-timestamp");
      return LifetimeHelpers::fetchFromCCDBCache(matcher, serverUrl, timestamp, sourceChannel);
    };
  }

  static RouteConfigurator::DanglingConfigurator danglingQAConfigurator()
  {
    // FIXME: this should really be expireAlways. However, since we do not have
    //        a proper backend for conditions yet, I keep it behaving like it was
    //        before.
    return [](ConfigParamRegistry const&) { return LifetimeHelpers::expireNever(); };
  }

  static RouteConfigurator::ExpirationConfigurator expiringQAConfigurator()
  {
    return [](ConfigParamRegistry const&) { return LifetimeHelpers::fetchFromQARegistry(); };
  }

  static RouteConfigurator::DanglingConfigurator danglingTimerConfigurator(InputSpec const& matcher)
  {
    return [matcher](ConfigParamRegistry const& options) {
      return LifetimeHelpers::expireAlways();
    };
  }

  static RouteConfigurator::DanglingConfigurator danglingEnumerationConfigurator(InputSpec const& matcher)
  {
    return [matcher](ConfigParamRegistry const& options) {
      return LifetimeHelpers::expireAlways();
    };
  }

  static RouteConfigurator::ExpirationConfigurator expiringTimerConfigurator(InputSpec const& spec, std::string const& sourceChannel)
  {
    auto m = std::get_if<ConcreteDataMatcher>(&spec.matcher);
    if (m == nullptr) {
      throw std::runtime_error("InputSpec for Timers must be fully qualified");
    }
    // We copy the matcher to avoid lifetime issues.
    return [matcher = *m, sourceChannel](ConfigParamRegistry const&) { return LifetimeHelpers::enumerate(matcher, sourceChannel); };
  }

  static RouteConfigurator::ExpirationConfigurator expiringEnumerationConfigurator(InputSpec const& spec, std::string const& sourceChannel)
  {
    auto m = std::get_if<ConcreteDataMatcher>(&spec.matcher);
    if (m == nullptr) {
      throw std::runtime_error("InputSpec for Enumeration must be fully qualified");
    }
    // We copy the matcher to avoid lifetime issues.
    return [matcher = *m, sourceChannel](ConfigParamRegistry const&) {
      return LifetimeHelpers::enumerate(matcher, sourceChannel);
    };
  }

  static RouteConfigurator::DanglingConfigurator danglingTransientConfigurator()
  {
    // FIXME: this should really be expireAlways. However, since we do not have
    //        a proper backend for conditions yet, I keep it behaving like it was
    //        before.
    return [](ConfigParamRegistry const&) { return LifetimeHelpers::expireNever(); };
  }

  static RouteConfigurator::ExpirationConfigurator expiringTransientConfigurator(InputSpec const& matcher)
  {
    return [](ConfigParamRegistry const&) { return LifetimeHelpers::fetchFromObjectRegistry(); };
  }
};

/// This creates a string to configure channels of a FairMQDevice
/// FIXME: support shared memory
std::string inputChannel2String(const InputChannelSpec& channel)
{
  std::string result;

  result += "name=" + channel.name;
  result += std::string(",type=") + ChannelSpecHelpers::typeAsString(channel.type);
  result += std::string(",method=") + ChannelSpecHelpers::methodAsString(channel.method);
  result += std::string(",address=") + ChannelSpecHelpers::channelUrl(channel);
  result += std::string(",rateLogging=60");

  return result;
}

std::string outputChannel2String(const OutputChannelSpec& channel)
{
  std::string result;

  result += "name=" + channel.name;
  result += std::string(",type=") + ChannelSpecHelpers::typeAsString(channel.type);
  result += std::string(",method=") + ChannelSpecHelpers::methodAsString(channel.method);
  result += std::string(",address=") + ChannelSpecHelpers::channelUrl(channel);
  result += std::string(",rateLogging=60");

  return result;
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
                                              ComputingOffer const& defaultOffer)
{
  // The topology cannot be empty or not connected. If that is the case, than
  // something before this went wrong.
  // FIXME: is that really true???
  assert(!workflow.empty());

  // Edges are navigated in order for each device, so the device associaited to
  // an edge is always the last one created.
  auto deviceForEdge = [&actions, &workflow, &devices, &logicalEdges, &resourceManager, &defaultOffer](size_t ei, ComputingOffer& acceptedOffer) {
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
    if (processor.maxInputTimeslices != 1) {
      device.id = processor.name + "_t" + std::to_string(edge.producerTimeIndex);
    }
    device.algorithm = processor.algorithm;
    device.options = processor.options;
    device.rank = processor.rank;
    device.nSlots = processor.nSlots;
    device.inputTimesliceId = edge.timeIndex;
    device.resource = {acceptedOffer};
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
    channel.name = "from_" + device.id + "_to_" + consumerDeviceId;
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
    return std::move(channel);
  };

  auto isDifferentDestinationDeviceReferredBy = [&actions](size_t ei) { return actions[ei].requiresNewChannel; };

  // This creates a new channel for a given edge, if needed. Notice that we
  // navigate edges in a per device fashion (creating those if they are not
  // alredy there) and create a new channel only if it connects two new
  // devices. Whether or not this is the case was previously computed
  // in the action.requiresNewChannel field.
  auto createChannelForDeviceEdge = [&devices, &logicalEdges, &channelFromDeviceEdgeAndPort,
                                     &outputsMatchers, &deviceIndex,
                                     &workflow](size_t di, size_t ei, ComputingOffer& offer) {
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
  resourceManager.notifyAcceptedOffer(acceptedOffer);
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
                                             ComputingOffer const& defaultOffer)
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

  auto createNewDeviceForEdge = [&workflow, &logicalEdges, &devices, &deviceIndex, &resourceManager, &defaultOffer](size_t ei, ComputingOffer& acceptedOffer) {
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
    if (processor.maxInputTimeslices != 1) {
      device.id += "_t" + std::to_string(edge.timeIndex);
    }
    device.algorithm = processor.algorithm;
    device.options = processor.options;
    device.rank = processor.rank;
    device.nSlots = processor.nSlots;
    device.inputTimesliceId = edge.timeIndex;
    device.resource = {acceptedOffer};

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
  auto appendInputChannelForConsumerDevice = [&devices, &connections, &checkNoDuplicatesFor, &channelPolicies](
                                               size_t pi, size_t ci, unsigned short port) {
    auto const& producerDevice = devices[pi];
    auto& consumerDevice = devices[ci];
    InputChannelSpec channel;
    channel.name = "from_" + producerDevice.id + "_to_" + consumerDevice.id;
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

    switch (consumer.inputs[edge.consumerInputIndex].lifetime) {
      case Lifetime::Condition:
        route.configurator = {
          ExpirationHandlerHelpers::dataDrivenConfigurator(),
          ExpirationHandlerHelpers::danglingConditionConfigurator(),
          ExpirationHandlerHelpers::expiringConditionConfigurator(inputSpec, sourceChannel)};
        break;
      case Lifetime::QA:
        route.configurator = {
          ExpirationHandlerHelpers::dataDrivenConfigurator(),
          ExpirationHandlerHelpers::danglingQAConfigurator(),
          ExpirationHandlerHelpers::expiringQAConfigurator()};
        break;
      case Lifetime::Timer:
        route.configurator = {
          ExpirationHandlerHelpers::timeDrivenConfigurator(inputSpec),
          ExpirationHandlerHelpers::danglingTimerConfigurator(inputSpec),
          ExpirationHandlerHelpers::expiringTimerConfigurator(inputSpec, sourceChannel)};
        break;
      case Lifetime::Enumeration:
        route.configurator = {
          ExpirationHandlerHelpers::enumDrivenConfigurator(inputSpec),
          ExpirationHandlerHelpers::danglingEnumerationConfigurator(inputSpec),
          ExpirationHandlerHelpers::expiringEnumerationConfigurator(inputSpec, sourceChannel)};
        break;
      case Lifetime::Transient:
        route.configurator = {
          ExpirationHandlerHelpers::dataDrivenConfigurator(),
          ExpirationHandlerHelpers::danglingTransientConfigurator(),
          ExpirationHandlerHelpers::expiringTransientConfigurator(inputSpec)};
        break;
      default:
        break;
    }

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

    size_t consumerDevice;

    if (action.requiresNewDevice) {
      if (hasConsumerForEdge(edge)) {
        consumerDevice = getConsumerForEdge(edge);
      } else {
        consumerDevice = createNewDeviceForEdge(edge, acceptedOffer);
      }
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
  resourceManager.notifyAcceptedOffer(acceptedOffer);
}

// Construct the list of actual devices we want, given a workflow.
//
// FIXME: make start port configurable?
void DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(WorkflowSpec const& workflow,
                                                       std::vector<ChannelConfigurationPolicy> const& channelPolicies,
                                                       std::vector<CompletionPolicy> const& completionPolicies,
                                                       std::vector<DispatchPolicy> const& dispatchPolicies,
                                                       std::vector<DeviceSpec>& devices,
                                                       ResourceManager& resourceManager,
                                                       std::string const& uniqueWorkflowId)
{

  std::vector<LogicalForwardInfo> availableForwardsInfo;
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<DeviceConnectionId> connections;
  std::vector<DeviceId> deviceIndex;

  // This is a temporary store for inputs and outputs,
  // including forwarded channels, so that we can construct
  // them before assigning to a device.
  std::vector<OutputSpec> outputs;

  WorkflowHelpers::verifyWorkflow(workflow);
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
                        outActions, workflow, outputs, channelPolicies, defaultOffer);

  // FIXME: is this not the case???
  std::sort(connections.begin(), connections.end());

  processInEdgeActions(devices, deviceIndex, connections, resourceManager, inEdgeIndex, logicalEdges,
                       inActions, workflow, availableForwardsInfo, channelPolicies, defaultOffer);
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
    throw std::runtime_error("Unable to find device.");
  };

  // Optimize the topology when two devices are
  // running on the same node.
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

void DeviceSpecHelpers::prepareArguments(bool defaultQuiet, bool defaultStopped,
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
    for (auto& processorInfo : processorInfos) {
      if (processorInfo.name == spec.id) {
        argc = processorInfo.cmdLineArgs.size() + 1;
        argv = (char**)malloc(sizeof(char**) * (argc + 1));
        argv[0] = strdup(processorInfo.executable.data());
        for (size_t ai = 0; ai < processorInfo.cmdLineArgs.size(); ++ai) {
          auto& arg = processorInfo.cmdLineArgs[ai];
          argv[ai + 1] = strdup(arg.data());
        }
        argv[argc] = nullptr;
        workflowOptions = processorInfo.workflowOptions;
        break;
      }
    }

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
                                        "--control", "static",
                                        "--session", "dpl_" + uniqueWorkflowId,
                                        "--log-color", "false",
                                        "--color", "false"};
    if (defaultStopped) {
      tmpArgs.push_back("-s");
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
    foDesc.add(getForwardedDeviceOptions());

    using FilterFunctionT = std::function<void(decltype(argc), decltype(argv), decltype(od))>;

    // the filter function will forward command line arguments based on the option
    // definition passed to it. All options of the program option definition will be forwarded
    // if found in the argument list. If not found they will be added with the default value
    FilterFunctionT filterArgsFct = [&](int largc, char** largv, const bpo::options_description& odesc) {
      // spec contains options
      bpo::command_line_parser parser{largc, largv};
      parser.options(odesc).allow_unregistered();
      bpo::parsed_options parsed_options = parser.run();

      bpo::variables_map varmap;
      bpo::store(parsed_options, varmap);

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
        realOdesc.add_options()("child-driver", bpo::value<std::string>());
        realOdesc.add_options()("rate", bpo::value<std::string>());
        realOdesc.add_options()("shm-segment-size", bpo::value<std::string>());
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
            //assert(semantic->max_tokens() && semantic->min_tokens());
            if (semantic->min_tokens() > 0) {
              tmpArgs.emplace_back("--");
              tmpArgs.back() += varit.first;
              // add the token
              tmpArgs.emplace_back(varit.second.as<std::string>());
              optarg = tmpArgs.back().c_str();
            } else if (semantic->min_tokens() == 0 && varit.second.as<bool>()) {
              tmpArgs.emplace_back("--");
              tmpArgs.back() += varit.first;
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

    // We create the final option list, depending on the channels
    // which are present in a device.
    for (auto& arg : tmpArgs) {
      execution.args.emplace_back(strdup(arg.c_str()));
    }
    // execvp wants a NULL terminated list.
    execution.args.push_back(nullptr);

    // FIXME: this should probably be reflected in the GUI
    std::ostringstream str;
    for (size_t ai = 0; ai < execution.args.size() - 1; ai++) {
      if (execution.args[ai] == nullptr) {
        LOG(ERROR) << "Bad argument for " << execution.args[ai - 1];
      }
      assert(execution.args[ai]);
      str << " " << execution.args[ai];
    }
    LOG(DEBUG) << "The following options are being forwarded to " << spec.id << ":" << str.str();
  }
}

/// define the options which are forwarded to every child
boost::program_options::options_description DeviceSpecHelpers::getForwardedDeviceOptions()
{
  // - rate is an option of FairMQ device for ConditionalRun
  // - child-driver is not a FairMQ device option but used per device to start to process
  bpo::options_description forwardedDeviceOptions;
  forwardedDeviceOptions.add_options()                                                                          //
    ("plugin,P", bpo::value<std::string>(), "FairMQ plugin list")                                               //
    ("plugin-search-path,S", bpo::value<std::string>(), "FairMQ plugins search path")                           //
    ("control-port", bpo::value<std::string>(), "Utility port to be used by O2 Control")                        //
    ("rate", bpo::value<std::string>(), "rate for a data source device (Hz)")                                   //
    ("shm-segment-size", bpo::value<std::string>(), "size of the shared memory segment in bytes")               //
    ("session", bpo::value<std::string>(), "unique label for the shm session")                                  //
    ("monitoring-backend", bpo::value<std::string>(), "monitoring connection string")                           //
    ("infologger-mode", bpo::value<std::string>(), "INFOLOGGER_MODE override")                                  //
    ("infologger-severity", bpo::value<std::string>(), "minimun FairLogger severity which goes to info logger") //
    ("child-driver", bpo::value<std::string>(), "external driver to start childs with (e.g. valgrind)");        //

  return forwardedDeviceOptions;
}

} // namespace framework
} // namespace o2
