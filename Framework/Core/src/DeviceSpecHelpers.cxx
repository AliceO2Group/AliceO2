// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "WorkflowHelpers.h"
#include "DeviceSpecHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ChannelMatching.h"
#include "Framework/DeviceControl.h"
#include "Framework/OutputRoute.h"
#include "Framework/ConfigParamsHelper.h"
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <boost/program_options.hpp>
#include <wordexp.h>

namespace bpo = boost::program_options;

using namespace o2::framework;

namespace o2 {
namespace framework {

using LogicalChannelsMap = std::map<LogicalChannel, size_t>;

/// This creates a string to configure channels of a FairMQDevice
/// FIXME: support shared memory
std::string inputChannel2String(const InputChannelSpec &channel) {
  std::string result;
  char buffer[32];
  auto addressFormat = (channel.method == Bind ? "tcp://*:%d" : "tcp://127.0.0.1:%d");

  result += "name=" + channel.name + ",";
  result += std::string("type=") + (channel.type == Pub ? "pub" : "sub") + ",";
  result += std::string("method=") + (channel.method == Bind ? "bind" : "connect") + ",";
  result += std::string("address=") + (snprintf(buffer,32,addressFormat, channel.port), buffer);

  return result;
}

std::string outputChannel2String(const OutputChannelSpec &channel) {
  std::string result;
  char buffer[32];
  auto addressFormat = (channel.method == Bind ? "tcp://*:%d" : "tcp://127.0.0.1:%d");

  result += "name=" + channel.name + ",";
  result += std::string("type=") + (channel.type == Pub ? "pub" : "sub") + ",";
  result += std::string("method=") + (channel.method == Bind ? "bind" : "connect") + ",";
  result += std::string("address=") + (snprintf(buffer,32,addressFormat, channel.port), buffer);

  return result;
}

void
DeviceSpecHelpers::processOutEdgeActions(
      std::vector<DeviceSpec> &devices,
      std::vector<DeviceId> &deviceIndex,
      std::vector<DeviceConnectionId> &connections,
      unsigned short &nextPort,
      const std::vector<size_t> &outEdgeIndex,
      const std::vector<DeviceConnectionEdge> &logicalEdges,
      const std::vector<EdgeAction> &actions,
      const WorkflowSpec &workflow,
      const std::vector<OutputSpec> &outputsMatchers
    ) {
  // The topology cannot be empty or not connected. If that is the case, than
  // something before this went wrong.
  // FIXME: is that really true???
  assert(!workflow.empty());

  // Edges are navigated in order for each device, so the device associaited to
  // an edge is always the last one created.
  auto deviceForEdge = [&actions, &workflow, &devices, &logicalEdges](size_t ei) {
    auto &edge = logicalEdges[ei];
    auto &action = actions[ei];

    if (action.requiresNewDevice == false) {
      assert(devices.empty() == false);
      return devices.size() - 1;
    }
    auto processor = workflow[edge.producer];
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
    devices.push_back(device);
    return devices.size() - 1;
  };

  auto channelFromDeviceEdgeAndPort = [&workflow]
    (const DeviceSpec &device, const DeviceConnectionEdge &edge, short port) {
    OutputChannelSpec channel;
    auto &consumer = workflow[edge.consumer];
    std::string consumerName = workflow[edge.consumer].name;
    if (consumer.maxInputTimeslices != 1) {
      consumerName += "_t" + std::to_string(edge.timeIndex);
    }
    channel.name = "from_" + device.id + "_to_" + consumerName;
    channel.method = Bind;
    channel.type = Pub;
    channel.port = port;
    return std::move(channel);
  };

  auto connectionIdFromEdgeAndPort = [&connections](const DeviceConnectionEdge &edge, size_t port) {
    DeviceConnectionId id{
      edge.producer,
      edge.consumer,
      edge.timeIndex,
      edge.producerTimeIndex,
      port
    };
    connections.push_back(id);
    return connections.back();
  };

  auto isDifferentDestinationDeviceReferredBy = [&actions](size_t ei) {
    return actions[ei].requiresNewChannel;
  };

  // This creates a new channel for a given edge, if needed. Notice that we
  // navigate edges in a per device fashion (creating those if they are not
  // alredy there) and create a new channel only if it connects two new
  // devices. Whether or not this is the case was previously computed
  // in the action.requiresNewChannel field.
  auto createChannelForDeviceEdge = [&devices,
                                     &logicalEdges,
                                     &nextPort,
                                     &channelFromDeviceEdgeAndPort,
                                     &connectionIdFromEdgeAndPort,
                                     &outputsMatchers,
                                     &deviceIndex,
                                     &workflow](size_t di, size_t ei) {
    auto &device = devices[di];
    auto &edge = logicalEdges[ei];

    deviceIndex.emplace_back(DeviceId{edge.producer, edge.producerTimeIndex, di});

    OutputChannelSpec channel = channelFromDeviceEdgeAndPort(device, edge, nextPort);
    const DeviceConnectionId &id = connectionIdFromEdgeAndPort(edge, nextPort);
    nextPort++;

    device.outputChannels.push_back(channel);
    return device.outputChannels.size() - 1;
  };

  // Notice how we need to behave in two different ways depending
  // whether this is a real OutputRoute or if it's a forward from
  // a previous consumer device.
  // FIXME: where do I find the InputSpec for the forward?
  auto appendOutputRouteToSourceDeviceChannel = [&outputsMatchers, &workflow, &devices, &logicalEdges]
      (size_t ei, size_t di, size_t ci) {
    assert(ei < logicalEdges.size());
    assert(di < devices.size());
    assert(ci < devices[di].outputChannels.size());
    auto &edge = logicalEdges[ei];
    auto &device = devices[di];
    assert(edge.consumer < workflow.size());
    auto &consumer = workflow[edge.consumer];
    auto &channel = devices[di].outputChannels[ci];
    assert(edge.outputGlobalIndex < outputsMatchers.size());

    if (edge.isForward == false) {
      OutputRoute route;
      route.matcher = outputsMatchers[edge.outputGlobalIndex];
      route.timeslice = edge.timeIndex;
      route.maxTimeslices = consumer.maxInputTimeslices;
      route.channel = channel.name;
      device.outputs.emplace_back(route);
    } else {
      ForwardRoute route;
      route.matcher = workflow[edge.consumer].inputs[edge.consumerInputIndex];
      route.channel = channel.name;
      device.forwards.emplace_back(route);
    }
  };

  auto sortDeviceIndex = [&deviceIndex]() {
    std::sort(deviceIndex.begin(), deviceIndex.end());
  };

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
  for (auto edge : outEdgeIndex) {
    auto device = deviceForEdge(edge);
    size_t channel = -1;
    if (isDifferentDestinationDeviceReferredBy(edge)) {
      channel = createChannelForDeviceEdge(device, edge);
    } else {
      channel = lastChannelFor(device);
    }
    appendOutputRouteToSourceDeviceChannel(edge, device, channel);
  }
  sortDeviceIndex();
}

void
DeviceSpecHelpers::processInEdgeActions(
      std::vector<DeviceSpec> &devices,
      std::vector<DeviceId> &deviceIndex,
      unsigned short &nextPort,
      const std::vector<DeviceConnectionId> &connections,
      const std::vector<size_t> &inEdgeIndex,
      const std::vector<DeviceConnectionEdge> &logicalEdges,
      const std::vector<EdgeAction> &actions,
      const WorkflowSpec &workflow,
      std::vector<LogicalForwardInfo> const &availableForwardsInfo
    ) {
  auto const &constDeviceIndex = deviceIndex;

  auto findProducerForEdge = [&logicalEdges,&constDeviceIndex](size_t ei) {
    auto &edge = logicalEdges[ei];

    DeviceId pid{edge.producer, edge.producerTimeIndex, 0};
    auto deviceIt = std::lower_bound(constDeviceIndex.cbegin(),
                                     constDeviceIndex.cend(), pid);
    // By construction producer should always be there
    assert(deviceIt != constDeviceIndex.end());
    assert(deviceIt->processorIndex == pid.processorIndex
           && deviceIt->timeslice == pid.timeslice);
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
  auto hasConsumerForEdge = [&lastConsumerSearch, &lastConsumerSearchEdge, &deviceIndex, &logicalEdges]                            (size_t ei) -> int {
    auto &edge = logicalEdges[ei];
    DeviceId cid{edge.consumer, edge.timeIndex, 0};
    lastConsumerSearchEdge = ei; // This will invalidate the cache
    lastConsumerSearch = std::lower_bound(deviceIndex.begin(), deviceIndex.end(), cid);
    return lastConsumerSearch != deviceIndex.end()
        && cid.processorIndex == lastConsumerSearch->processorIndex
        && cid.timeslice == lastConsumerSearch->timeslice;
  };

  // The passed argument is there just to check. We do know that the last searched
  // is the one we want.
  auto getConsumerForEdge = [&lastConsumerSearch, &lastConsumerSearchEdge](size_t ei) {
    assert(ei == lastConsumerSearchEdge);
    return lastConsumerSearch->deviceIndex;
  };

  auto createNewDeviceForEdge = [&workflow, &logicalEdges, &devices, &deviceIndex] (size_t ei) {
    auto &edge = logicalEdges[ei];
    auto &processor = workflow[edge.consumer];
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
    // FIXME: maybe I should use an std::map in the end
    //        but this is really not performance critical
    auto id = DeviceId{edge.consumer,
                       edge.timeIndex,
                       devices.size()};
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
    auto const &edge = logicalEdges[ei];
    DeviceConnectionId connectionId{
      edge.producer,
      edge.consumer,
      edge.timeIndex,
      edge.producerTimeIndex,
      0
    };

    auto it = std::lower_bound(connections.begin(), connections.end(), connectionId);

    assert(it != connections.end());
    assert(it->producer == connectionId.producer);
    assert(it->consumer == connectionId.consumer);
    assert(it->timeIndex == connectionId.timeIndex);
    assert(it->producerTimeIndex == connectionId.producerTimeIndex);
    return it->port;
  };

  auto checkNoDuplicatesFor = [](std::vector<InputChannelSpec> const&channels, 
                                 const std::string &name) {
    for (auto const &channel : channels) {
      if (channel.name == name) {
        return false;
      }
    }
    return true;
  };
  auto appendInputChannelForConsumerDevice = [&devices,&connections,&checkNoDuplicatesFor]
        (size_t pi, size_t ci, int16_t port) {
    auto const &producerDevice = devices[pi];
    auto &consumerDevice = devices[ci];
    InputChannelSpec channel;
    channel.name = "from_" + producerDevice.id + "_to_" + consumerDevice.id;
    channel.method = ChannelMethod::Connect;
    channel.type = ChannelType::Sub;
    channel.port = port;
    assert(checkNoDuplicatesFor(consumerDevice.inputChannels, channel.name));
    consumerDevice.inputChannels.push_back(channel);
    return consumerDevice.inputChannels.size() - 1;
  };

  // I think this is trivial, since I think it should always be the last one,
  // in case it's not actually the case, I should probably do an actual lookup
  // here.
  auto getChannelForEdge = [&devices](size_t pi, size_t ci) {
    auto &consumerDevice = devices[ci];
    return consumerDevice.inputChannels.size() - 1;
  };

  // This is always called when adding a new channel, so we can simply refer
  // to back. Notice also that this is the place where it makes sense to
  // assign the forwarding, given that the forwarded stuff comes from some
  // input.
  auto appendInputRouteToDestDeviceChannel = [&devices,&logicalEdges,&workflow]
      (size_t ei, size_t di, size_t ci) {
    auto const &edge = logicalEdges[ei];
    auto const &consumer = workflow[edge.consumer];
    auto &consumerDevice = devices[di];
    InputRoute route;
    route.matcher = consumer.inputs[edge.consumerInputIndex];
    route.sourceChannel = consumerDevice.inputChannels[ci].name;
    consumerDevice.inputs.push_back(route);
  };

  // Outer loop. A new device is needed for each
  // of the sink data processors.
  // New InputChannels need to refer to preexisting OutputChannels we create
  // previously.
  for (size_t edge : inEdgeIndex) {
    auto &action = actions[edge];

    size_t consumerDevice;

    if (action.requiresNewDevice) {
      if (hasConsumerForEdge(edge)) {
        consumerDevice = getConsumerForEdge(edge);
      } else {
        consumerDevice = createNewDeviceForEdge(edge);
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
}

// Construct the list of actual devices we want, given a workflow.
// 
// FIXME: make start port configurable?
void
DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(
    const o2::framework::WorkflowSpec &workflow,
    std::vector<o2::framework::DeviceSpec> &devices) {

  std::vector<LogicalForwardInfo> availableForwardsInfo;
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<DeviceConnectionId> connections;
  std::vector<DeviceId> deviceIndex;

  // This is a temporary store for inputs and outputs, 
  // including forwarded channels, so that we can construct
  // them before assigning to a device.
  std::vector<OutputSpec> outputs;

  WorkflowHelpers::verifyWorkflow(workflow);
  WorkflowHelpers::constructGraph(workflow,
                                  logicalEdges,
                                  outputs,
                                  availableForwardsInfo);

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
  unsigned short nextPort = 22000;

  std::vector<EdgeAction> actions = WorkflowHelpers::computeOutEdgeActions(logicalEdges, outEdgeIndex);

  DeviceSpecHelpers::processOutEdgeActions(
      devices,
      deviceIndex,
      connections,
      nextPort,
      outEdgeIndex,
      logicalEdges,
      actions,
      workflow,
      outputs
      );


  // Crete the connections on the inverse map for all of them
  // lookup for port and add as input of the current device.
  std::vector<EdgeAction> inActions = WorkflowHelpers::computeInEdgeActions(logicalEdges, inEdgeIndex);

  // FIXME: is this not the case???
  std::sort(connections.begin(), connections.end());

  processInEdgeActions(
      devices,
      deviceIndex,
      nextPort,
      connections,
      inEdgeIndex,
      logicalEdges,
      inActions,
      workflow,
      availableForwardsInfo
      );
}

void
DeviceSpecHelpers::prepareArguments(int argc,
                 char **argv,
                 bool defaultQuiet,
                 bool defaultStopped,
                 const std::vector<DeviceSpec> &deviceSpecs,
                 std::vector<DeviceExecution> &deviceExecutions,
                 std::vector<DeviceControl> &deviceControls)
{
  assert(deviceSpecs.size() == deviceExecutions.size());
  assert(deviceControls.size() == deviceExecutions.size());
  for (size_t si = 0; si < deviceSpecs.size(); ++si) {
    auto &spec = deviceSpecs[si];
    auto &control = deviceControls[si];
    auto &execution = deviceExecutions[si];

    control.quiet = defaultQuiet;
    control.stopped = defaultStopped;

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
    std::vector<std::string> tmpArgs = {
      argv[0],
      "--id",
      spec.id.c_str(),
      "--control",
      "static",
      "--log-color",
      "0"
    };

    // do the filtering of options, forward options belonging to this specific
    // DeviceSpec, and some global options from getForwardedDeviceOptions
    const char* name = spec.name.c_str();
    bpo::options_description od;
    prepareOptionsDescription(spec.options, od);
    od.add(getForwardedDeviceOptions());
    od.add_options()(name, bpo::value<std::string>());

    using FilterFunctionT = std::function<void (decltype(argc), decltype(argv),
                                                decltype(od))>;

    FilterFunctionT filterArgsFct = [&] (int largc, char **largv,
                                         const bpo::options_description &odesc) {
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
        filterArgsFct(expansions.we_wordc, expansions.we_wordv, odesc);
        wordfree(&expansions);
        return;
      }

      for (const auto varit : varmap) {
        // find the option belonging to key, add if the option has been parsed
        // and is not defaulted
        const auto * description = odesc.find_nothrow(varit.first, false);
        if (description && varmap.count(varit.first)) {
          tmpArgs.emplace_back("--");
          tmpArgs.back() += varit.first;
          // check the semantics of the value
          auto semantic = description->semantic();
          const char* optarg = "";
          if (semantic) {
            // the value semantics allows different properties like
            // multitoken, zero_token and composing
            // currently only the simple case is supported
            assert(semantic->min_tokens() <= 1);
            assert(semantic->max_tokens() && semantic->min_tokens());
            if (semantic->min_tokens() > 0 ) {
              // add the token
              tmpArgs.emplace_back(varit.second.as<std::string>());
              optarg = tmpArgs.back().c_str();
            }
          }
          control.options.insert(std::make_pair(varit.first,
                                                optarg));
        }
      }
    };

    filterArgsFct(argc, argv, od);

    // Add the channel configuration
    for (auto &channel : spec.outputChannels) {
      tmpArgs.emplace_back(std::string("--channel-config"));
      tmpArgs.emplace_back(outputChannel2String(channel));
    }
    for (auto &channel : spec.inputChannels) {
      tmpArgs.emplace_back(std::string("--channel-config"));
      tmpArgs.emplace_back(inputChannel2String(channel));
    }

    // We create the final option list, depending on the channels
    // which are present in a device.
    for (auto &arg : tmpArgs) {
      execution.args.emplace_back(strdup(arg.c_str()));
    }
    // execvp wants a NULL terminated list.
    execution.args.push_back(nullptr);

    //FIXME: this should probably be reflected in the GUI
    std::ostringstream str;
    for (size_t ai = 0; ai < execution.args.size() - 1; ai++) {
      assert(execution.args[ai]);
      str << " " << execution.args[ai];
    }
    LOG(DEBUG) << "The following options are being forwarded to "
               << spec.id << ":" << str.str();
  }
}

boost::program_options::options_description DeviceSpecHelpers::getForwardedDeviceOptions()
{
  bpo::options_description forwardedDeviceOptions;
  forwardedDeviceOptions.add_options()
    ("rate",
     bpo::value<std::string>(),
     "rate for a data source device (Hz)");

  return forwardedDeviceOptions;
}

} // namespace framework
} // namespace o2
