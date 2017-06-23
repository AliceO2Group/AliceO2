// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "FairMQDevice.h"
#include "Framework/DataProcessingDevice.h"
#include "Framework/FrameworkGUIDebugger.h"
#include "Framework/DataSourceDevice.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceControl.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DebugGUI.h"

#include "GraphvizHelpers.h"
#include "options/FairMQProgOptions.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <getopt.h>

#include "fairmq/tools/runSimpleMQStateMachine.h"

using namespace o2::framework;

std::vector<DeviceInfo> gDeviceInfos;
std::vector<DeviceControl> gDeviceControls;

// Read from a given fd and print it.
// return true if we can still read from it,
// return false if we need to close the input pipe.
// 
// FIXME: We should really print full lines.
bool getChildData(int infd, DeviceInfo &outinfo) {
  char buffer[1024];
  int bytes_read;
  // NOTE: do not quite understand read ends up blocking if I read more than
  //        once. Oh well... Good enough for now.
  // do {
      bytes_read = read(infd, buffer, 1024);
      if (bytes_read == 0) {
        return false;
      }
      if (bytes_read == -1) {
        switch(errno) {
          case EAGAIN:
            return true;
          default:
            return false;
        }
      }
      assert(bytes_read > 0);
      outinfo.unprinted += std::string(buffer, bytes_read);
//  } while (bytes_read != 0);
  return true;
}

// This is the handler for the parent inner loop.
// So far the only responsibility for it are:
//
// - Echo children output in a sensible manner
//
//
// - TODO: allow single child view?
// - TODO: allow last line per child mode?
// - TODO: allow last error per child mode?
int doParent(fd_set *in_fdset,
             int maxFd,
             std::vector<DeviceInfo> infos,
             std::vector<DeviceSpec> specs,
             std::vector<DeviceControl> controls,
             std::map<int,size_t> &socket2Info) {
  void *window = initGUI("O2 Framework debug GUI");
  // FIXME: I should really have some way of exiting the
  // parent..
  auto debugGUICallback = getGUIDebugger(infos, specs, controls);

  while (pollGUI(window, debugGUICallback)) {
    // Wait for children to say something. When they do
    // print it.
    fd_set *fdset = (fd_set *)malloc(sizeof(fd_set));
    timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 16666; // This should be enough to allow 60 HZ redrawing.
    FD_COPY(in_fdset, fdset);
    int numFd = select(maxFd, fdset, NULL, NULL, &timeout);
    if (numFd == 0) {
      continue;
    }
    for (int si = 0; si < maxFd; ++si) {
      if (FD_ISSET(si, fdset)) {
        assert(socket2Info.find(si) != socket2Info.end());
        auto &info = infos[socket2Info[si]];
        bool fdActive = getChildData(si, info);
        // If the pipe was closed due to the process exiting, we
        // can avoid the select.
        if (!fdActive) {
          info.active = false;
          close(si);
          FD_CLR(si, in_fdset);
        }
        --numFd;
      }
      // FIXME: no need to check after numFd gets to 0.
    }
    // Display part. All you need to display should actually be in
    // `infos`.
    // TODO: split at \n 
    // TODO: update this only once per 1/60 of a second or
    // things like this. 
    // TODO: have multiple display modes
    // TODO: graphical view of the processing?
    assert(infos.size() == controls.size());
    for (size_t di = 0, de = infos.size(); di < de; ++di) {
      DeviceInfo &info = infos[di];
      DeviceControl &control = controls[di];

      if (info.unprinted.empty()) {
        continue;
      }

      auto s = info.unprinted;
      std::string delimiter("\n");
      size_t pos = 0;
      std::string token;
      info.history.resize(info.historySize);
      while ((pos = s.find(delimiter)) != std::string::npos) {
          token = s.substr(0, pos);
          if (!control.quiet && (strstr(token.c_str(), control.logFilter) != NULL)) {
            assert(info.historyPos >= 0);
            assert(info.historyPos < info.history.size());
            info.history[info.historyPos] = token;
            info.historyPos = (info.historyPos + 1) % info.history.size();
            std::cout << "[" << info.pid << "]: " << token << std::endl;
          }
          s.erase(0, pos + delimiter.length());
      }
      info.unprinted = s;
    }
    // FIXME: for the gui to work correctly I would actually need to 
    //        run the loop more often and update whenever enough time has
    //        passed.
  }
  return 0;
}

struct LogicalChannel {
  std::string name;
  bool operator<(LogicalChannel const&other) const {
    return this->name < other.name;
  }
};


struct PhysicalChannel {
  std::string id;
  bool operator<(PhysicalChannel const&other) const {
    return this->id < other.id;
  }
};

LogicalChannel outputSpec2LogicalChannel(const OutputSpec &spec) {
  return LogicalChannel{std::string("out_") + spec.origin.str + "_" + spec.description.str};
}

PhysicalChannel outputSpec2PhysicalChannel(const OutputSpec &spec, int count) {
  char buffer[16];
  auto channel = outputSpec2LogicalChannel(spec);
  return PhysicalChannel{channel.name + (snprintf(buffer, 16, "_%d", count), buffer)};
}

LogicalChannel inputSpec2LogicalChannelMatcher(const InputSpec &spec) {
  return LogicalChannel{std::string("out_") + spec.origin.str + "_" + spec.description.str};
}

PhysicalChannel inputSpec2PhysicalChannelMatcher(const InputSpec&spec, int count) {
  char buffer[16];
  auto channel = inputSpec2LogicalChannelMatcher(spec);
  return PhysicalChannel{channel.name + (snprintf(buffer, 16, "_%d", count), buffer)};
}

/// @return true if a given DataSpec can use the provided channel.
/// FIXME: for the moment we require a full match, however matcher could really be
///        a *-expression or even a regular expression.
bool matchDataSpec2Channel(const InputSpec &spec, const LogicalChannel &channel) {
  auto matcher = inputSpec2LogicalChannelMatcher(spec);
  return matcher.name == channel.name;
}

// Construct the list of actual devices we want, given a workflow.
void
dataProcessorSpecs2DeviceSpecs(const o2::framework::WorkflowSpec &workflow,
                               std::vector<o2::framework::DeviceSpec> &devices) {
  // FIXME: for the moment we assume one channel per kind of product. Later on we 
  //        should optimize and make sure that multiple products can be multiplexed
  //        on a single channel in case of a single receiver.
  // FIXME: make start port configurable?
  std::map<LogicalChannel, int> maxOutputCounts; // How many outputs of a given kind do we need?
  // First we calculate the available outputs
  for (auto &spec : workflow) {
    for (auto &out : spec.outputs) {
      maxOutputCounts.insert(std::make_pair(outputSpec2LogicalChannel(out), 0));
    }
  }

  // Then we calculate how many inputs match a given output.
  for (auto &output: maxOutputCounts) {
    for (auto &spec : workflow) {
      for (auto &in : spec.inputs) {
        if (matchDataSpec2Channel(in, output.first)) {
          maxOutputCounts[output.first] += 1;
        }
      }
    }
  }

  // We then create the actual devices.
  // - If an input is used only once, we simply connect to the output.
  // - If an input is used by multiple entries, we keep track of how
  //   how many have used it so far and make sure that all but the last
  //   device using such an output forward it as `out_<origin>_<description>_<N>`.
  // - If there is no output matching a given input, we should complain
  // - If there is no input using a given output, we should warn?
  std::map<PhysicalChannel, short> portMappings;
  unsigned short nextPort = 22000;
  std::map<LogicalChannel, size_t> outputUsages;

  for (auto &processor : workflow) {
    DeviceSpec device;
    device.id = processor.name;
    device.process = processor.process;
    device.onError = processor.onError;
    // Channels which need to be forwarded (because they are used by
    // a downstream provider).
    std::vector<ChannelSpec> forwardedChannels;

    for (auto &outData : processor.outputs) {
      ChannelSpec channel;
      channel.method = Bind;
      channel.type = Pub;
      channel.port = nextPort;

      auto logicalChannel = outputSpec2LogicalChannel(outData);
      auto physicalChannel = outputSpec2PhysicalChannel(outData, 0);
      auto channelUsage = outputUsages.find(logicalChannel);
      // Decide the name of the channel. If this is
      // the first time this channel is used, it means it
      // is really an output.
      if (channelUsage != outputUsages.end()) {
        throw std::runtime_error("Too many outputs with the same name");
      }

      outputUsages.insert(std::make_pair(logicalChannel, 0));
      auto portAlloc = std::make_pair(physicalChannel, nextPort);
      channel.name = physicalChannel.id;

      device.channels.push_back(channel);
      device.outputs.insert(std::make_pair(channel.name, outData));

      // This should actually be implied by the previous assert.
      assert(portMappings.find(portAlloc.first) == portMappings.end());
      portMappings.insert(portAlloc);
      nextPort++;
    }

    // We now process the inputs. They are all of connect kind and we 
    // should look up in the previously created channel map where to
    // connect to. If we have more than one device which reads
    // from the same output, we should actually treat them as
    // serialised output.
    // FIXME: Alexey was referring to a device which can duplicate output
    //        without overhead (not sure how that would work). Maybe we should use
    //        that instead.
    for (auto &input : processor.inputs) {
      ChannelSpec channel;
      channel.method = Connect;
      channel.type = Sub;

      // Create the channel name and find how many times we have used it.
      auto logicalChannel = inputSpec2LogicalChannelMatcher(input);
      auto usagesIt = outputUsages.find(logicalChannel);
      if (usagesIt == outputUsages.end()) {
        throw std::runtime_error("Could not find output matching" + logicalChannel.name);
      }

      // Find the maximum number of usages for the channel.
      auto maxUsagesIt = maxOutputCounts.find(logicalChannel);
      if (maxUsagesIt == maxOutputCounts.end()) {
        // The previous throw should already catch this condition.
        assert(false && "The previous check should have already caught this");
      }
      auto maxUsages = maxUsagesIt->second;

      // Create the input channels:
      // - Name of the channel to lookup always channel_name_<usages>
      // - If usages is different from maxUsages, we should create a forwarding
      // channel.
      auto logicalInput = inputSpec2LogicalChannelMatcher(input);
      auto currentChannelId = outputUsages.find(logicalInput);
      if (currentChannelId == outputUsages.end()) {
        std::runtime_error("Missing output for " + logicalInput.name);
      }

      auto physicalChannel = inputSpec2PhysicalChannelMatcher(input, currentChannelId->second);
      auto currentPort = portMappings.find(physicalChannel);
      if (currentPort == portMappings.end()) {
        std::runtime_error("Missing physical channel " + physicalChannel.id);
      }

      channel.name = "in_" + physicalChannel.id;
      channel.port = currentPort->second;
      device.channels.push_back(channel);
      device.inputs.insert(std::make_pair(channel.name, input));
      // Increase the number of usages we did for a given logical channel
      currentChannelId->second += 1;

      // Here is where we create the forwarding port which can be later reused.
      if (currentChannelId->second != maxUsages) {
        ChannelSpec forwardedChannel;
        forwardedChannel.method = Bind;
        forwardedChannel.type = Pub;
        auto physicalForward = inputSpec2PhysicalChannelMatcher(input, currentChannelId->second);
        forwardedChannel.port = nextPort;
        portMappings.insert(std::make_pair(physicalForward, nextPort));
        forwardedChannel.name = physicalForward.id;

        device.channels.push_back(forwardedChannel);
        device.forwards.insert(std::make_pair(forwardedChannel.name, input));
        nextPort++;
      }
    }
    devices.push_back(device);
  }
}

/// This creates a string to configure channels of a FairMQDevice
/// FIXME: support shared memory
std::string channel2String(const ChannelSpec &channel) {
  std::string result;
  char buffer[32];
  auto addressFormat = (channel.method == Bind ? "tcp://*:%d" : "tcp://127.0.0.1:%d");

  result += "name=" + channel.name + ",";
  result += std::string("type=") + (channel.type == Pub ? "pub" : "sub") + ",";
  result += std::string("method=") + (channel.method == Bind ? "bind" : "connect") + ",";
  result += std::string("address=") + (snprintf(buffer,32,addressFormat, channel.port), buffer);

  std::cout << result << std::endl;
  return result;
}

int doChild(int argc, char **argv, const o2::framework::DeviceSpec &spec) {
  std::cout << "Spawing new device " << spec.id
            << " in process with pid " << getpid() << std::endl;
  // Set up options for the device running underneath
  std::vector<std::string> generalArgs = {
    "data-processor",
    "--id",
    spec.id.c_str(),
    "--control",
    "static",
    "--log-color",
    "0"
  };

  // Create the channel configuration
  for (auto &channel : spec.channels) {
    generalArgs.emplace_back(std::string("--channel-config"));
    generalArgs.emplace_back(channel2String(channel));
  }

  // We create the final option list, depending on the channels
  // which are present in a device.
  std::vector<char *> args;
  for (auto &arg : generalArgs) {
    args.emplace_back(strdup(arg.c_str()));
  }

  try {
    FairMQProgOptions config;
    config.ParseAll(args.size(), args.data());

    std::unique_ptr<FairMQDevice> device;
    if (spec.inputs.empty()) {
      LOG(DEBUG) << spec.id << " is a source\n";
      device.reset(new DataSourceDevice(spec));
    } else {
      LOG(DEBUG) << spec.id << " is a processor\n";
      device.reset(new DataProcessingDevice(spec));
    }

    if (!device)
    {
      LOG(ERROR) << "getDevice(): no valid device provided. Exiting.";
      return 1;
    }

    int result = runStateMachine(*device, config);

    if (result > 0) {
      return 1;
    }
  }
  catch(std::exception &e) {
    LOG(ERROR) << "Unhandled exception reached the top of main: " << e.what() << ", device shutting down.";
    return 1;
  }
  catch(...) {
    LOG(ERROR) << "Unknown exception reached the top of main.\n";
    return 1;
  }
  return 0;
}

int createPipes(int maxFd, int *pipes) {
    auto p = pipe(pipes);
    maxFd = maxFd > pipes[0] ? maxFd : pipes[0];
    maxFd = maxFd > pipes[1] ? maxFd : pipes[1];

    if (p == -1) {
      std::cerr << "Unable to create PIPE: ";
      switch(errno) {
        case EFAULT:
          assert(false && "EFAULT while reading from pipe");
        break;
        case EMFILE:
          std::cerr << "Too many active descriptors";
        break;
        case ENFILE:
          std::cerr << "System file table is full";
        break;
        default:
          std::cerr << "Unknown PIPE" << std::endl;
      };
      // Kill immediately both the parent and all the children
      kill(-1*getpid(), SIGKILL);
    }
    return maxFd;
}


void verifyWorkflow(const o2::framework::WorkflowSpec &specs) {
  std::set<std::string> validNames;
  std::vector<OutputSpec> availableOutputs;
  std::vector<InputSpec> requiredInputs;

  // An index many to one index to go from a given input to the
  // associated spec
  std::map<size_t, size_t> inputToSpec;
  // A one to one index to go from a given output to the Spec emitting it
  std::map<size_t, size_t> outputToSpec;

  for (auto &spec : specs)
  {
    if (spec.name.empty())
      throw std::runtime_error("Invalid DataProcessorSpec name");
    if (validNames.find(spec.name) != validNames.end())
      throw std::runtime_error("Name " + spec.name + " is used twice.");
  }
}

// Kill all the active children
void killChildren(std::vector<DeviceInfo> &infos) {
  for (auto &info : infos) {
    if (!info.active) {
      continue;
    }
    kill(info.pid, SIGKILL);
    int status;
    waitpid(info.pid, &status, 0);
  }
}

static void handle_sigint(int signum) {
  killChildren(gDeviceInfos);
  // We kill ourself after having killed all our children (SPOOKY!)
  signal(SIGINT, SIG_DFL);
  kill(getpid(), SIGINT);
}

void handle_sigchld(int sig) {
  int saved_errno = errno;
  pid_t exited = -1;
  std::vector<pid_t> pids;
  while (true) {
    pid_t pid = waitpid((pid_t)(-1), 0, WNOHANG);
    if (pid > 0) {
      pids.push_back(pid);
      continue;
    } else {
      break;
    }
  }
  errno = saved_errno;
  for (auto &pid : pids) {
    printf("Child exited: %d\n", pid);
    gDeviceInfos[pid].active = false;
    fflush(stdout);
  }
}

// This is a toy executor for the workflow spec
// What it needs to do is:
//
// - Print the properties of each DataProcessorSpec
// - Fork one process per DataProcessorSpec
//   - Parent -> wait for all the children to complete (eventually
//     killing them all on ctrl-c).
//   - Child, pick the data-processor ID and start a O2DataProcessorDevice for
//     each DataProcessorSpec
int doMain(int argc, char **argv, const o2::framework::WorkflowSpec & specs) {
  static struct option longopts[] = {
    {"quiet",     no_argument,  NULL, 'q' },
    {"stop",   no_argument,  NULL, 's' },
    {"batch", no_argument, NULL, 'b'},
    {"graphviz", no_argument, NULL, 'g'},
    { NULL,         0,            NULL, 0 }
  };

  int defaultQuiet = false;
  int defaultStopped = false;
  int noGui = false;
  int graphViz = false;

  int opt;
  while ((opt = getopt_long(argc, argv, "qsb",longopts, NULL)) != -1) {
    switch (opt) {
    case 'q':
        defaultQuiet = true;
        break;
    case 's':
        defaultStopped = true;
        break;
    case 'b':
        noGui = true;
        break;
    case 'g':
        graphViz = true;
        break;
    default: /* '?' */
        fprintf(stderr, "Usage: %s [--silent] [--stopped] [--batch]\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }
  }

  std::vector<DeviceSpec> deviceSpecs;

  try {
    verifyWorkflow(specs);
    dataProcessorSpecs2DeviceSpecs(specs, deviceSpecs);
    // This should expand nodes so that we can build a consistent DAG.
  } catch (std::runtime_error &e) {
    std::cerr << "Invalid workflow: " << e.what() << std::endl;
    dumpDataProcessorSpec2Graphviz(specs);
    return 1;
  }
  // Dump a graphviz representation of what I will do.
  dumpDeviceSpec2Graphviz(deviceSpecs);
  if (graphViz) {
    exit(0);
  }

  // Description of the running processes 
  // Mapping between various pipes and the actual device information.
  // Key is the file description, value is index in the previous vector.
  std::map<int, size_t> socket2DeviceInfo;
  int maxFd = 0;

  fd_set childFdset;
  FD_ZERO(&childFdset);

  struct sigaction sa_handle_child;
  sa_handle_child.sa_handler = &handle_sigchld;
  sigemptyset(&sa_handle_child.sa_mask);
  sa_handle_child.sa_flags = SA_RESTART | SA_NOCLDSTOP;
  if (sigaction(SIGCHLD, &sa_handle_child, 0) == -1) {
    perror(0);
    exit(1);
  }

  for (const auto &spec : deviceSpecs) {
    int childstdout[2];
    int childstderr[2];

    maxFd = createPipes(maxFd, childstdout);
    maxFd = createPipes(maxFd, childstderr);

    DeviceControl control;
    control.quiet = defaultQuiet;
    control.stopped = defaultStopped;

    gDeviceControls.emplace_back(control);

    pid_t id = fork();
    if (id == 0) {
      // This is the child.

      // We allow being debugged and do not terminate on SIGTRAP
      signal(SIGTRAP, SIG_IGN);

      // We do not start the process if control.noStart is set.
      if (control.stopped) {
        kill(getpid(), SIGSTOP);
      }

      // This is the child. We close the read part of the pipe, stdout
      // and dup2 the write part of the pipe on it.
      close(childstdout[0]);
      close(childstderr[0]);
      close(STDOUT_FILENO);
      close(STDERR_FILENO);
      dup2(childstdout[1], STDOUT_FILENO);
      dup2(childstderr[1], STDERR_FILENO);
      return doChild(argc, argv, spec);
    }

    // This is the parent. We close the write end of
    // the child pipe and and keep track of the fd so
    // that we can later select on it.
    struct sigaction sa_handle_int;
    sa_handle_int.sa_handler = handle_sigint;
    sigemptyset(&sa_handle_int.sa_mask);
    sa_handle_int.sa_flags = SA_RESTART;
    if (sigaction(SIGINT, &sa_handle_int, NULL) == -1) {
      perror("Unable to install signal handler");
      exit(1);
    }

    std::cout << "Starting " << spec.id << " on pid " << id << "\n";
    DeviceInfo info;
    info.pid = id;
    info.active = true;
    info.historySize = 1000;
    info.historyPos = 0;

    socket2DeviceInfo.insert(std::make_pair(childstdout[0], gDeviceInfos.size()));
    socket2DeviceInfo.insert(std::make_pair(childstderr[0], gDeviceInfos.size()));
    gDeviceInfos.emplace_back(info);

    close(childstdout[1]);
    close(childstderr[1]);
    FD_SET(childstdout[0], &childFdset);
    FD_SET(childstderr[0], &childFdset);
  }
  maxFd += 1;
  auto exitCode = doParent(&childFdset, maxFd, gDeviceInfos, deviceSpecs, gDeviceControls, socket2DeviceInfo);
  killChildren(gDeviceInfos);
  return exitCode;
}
