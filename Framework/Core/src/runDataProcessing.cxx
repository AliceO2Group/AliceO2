// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "FairMQDevice.h"
#include "Framework/ChannelMatching.h"
#include "Framework/DataProcessingDevice.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSourceDevice.h"
#include "Framework/ConfigParamsHelper.h"
#include "Framework/DebugGUI.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/FrameworkGUIDebugger.h"
#include "Framework/SimpleMetricsService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/LocalRootFileService.h"
#include "Framework/TextControlService.h"

#include "GraphvizHelpers.h"
#include "DDSConfigHelpers.h"
#include "options/FairMQProgOptions.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <string>

#include <getopt.h>
#include <csignal>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>


#include "fairmq/tools/runSimpleMQStateMachine.h"

using namespace o2::framework;

std::vector<DeviceInfo> gDeviceInfos;
std::vector<DeviceMetricsInfo> gDeviceMetricsInfos;
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
             std::vector<DeviceMetricsInfo> metricsInfos,
             std::map<int,size_t> &socket2Info) {
  void *window = initGUI("O2 Framework debug GUI");
  // FIXME: I should really have some way of exiting the
  // parent..
  auto debugGUICallback = getGUIDebugger(infos, specs, metricsInfos, controls);

  while (pollGUI(window, debugGUICallback)) {
    // Exit this loop if all the children say they want to quit.
    bool allReadyToQuit = true;
    for (auto &info : infos) {
      allReadyToQuit &= info.readyToQuit;
    }
    if (allReadyToQuit) {
      break;
    }

    // Wait for children to say something. When they do
    // print it.
    fd_set *fdset = (fd_set *)malloc(sizeof(fd_set));
    timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 16666; // This should be enough to allow 60 HZ redrawing.
    memcpy(fdset, in_fdset, sizeof(fd_set));
    int numFd = select(maxFd, fdset, nullptr, nullptr, &timeout);
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
    std::smatch match;
    std::string token;
    const std::string delimiter("\n");
    for (size_t di = 0, de = infos.size(); di < de; ++di) {
      DeviceInfo &info = infos[di];
      DeviceControl &control = controls[di];
      DeviceMetricsInfo &metrics = metricsInfos[di];

      if (info.unprinted.empty()) {
        continue;
      }

      auto s = info.unprinted;
      size_t pos = 0;
      info.history.resize(info.historySize);

      while ((pos = s.find(delimiter)) != std::string::npos) {
          token = s.substr(0, pos);
          // Check if the token is a metric from SimpleMetricsService
          // if yes, we do not print it out and simply store it to be displayed
          // in the GUI.
          // Then we check if it is part of our Poor man control system
          // if yes, we execute the associated command.
          if (parseMetric(token, match)) {
            LOG(INFO) << "Found metric with key " << match[2]
                      << " and value " <<  match[4];
            processMetric(match, metrics);
          } else if (parseControl(token, match)) {
            auto command = match[1];
            auto validFor = match[2];
            LOG(INFO) << "Found control command " << command
                      << " valid for " << validFor;
            if (command == "QUIT") {
              if (validFor == "ALL") {
                for (auto &deviceInfo : infos) {
                  deviceInfo.readyToQuit = true;
                }
              }
            }
          } else if (!control.quiet && (strstr(token.c_str(), control.logFilter) != nullptr)) {
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

  return result;
}

int doChild(int argc, char **argv, const o2::framework::DeviceSpec &spec) {
  std::cout << "Spawing new device " << spec.id
            << " in process with pid " << getpid() << std::endl;
  try {
    // Populate options from the command line. Notice that only the options
    // declared in the workflow definition are allowed.
    FairMQProgOptions config;
    boost::program_options::options_description optsDesc;
    populateBoostProgramOptions(optsDesc, spec.options);
    config.AddToCmdLineOptions(optsDesc, true);
    config.ParseAll(argc, argv);

    // We initialise this in the driver, because different drivers might have
    // different versions of the service
    ServiceRegistry serviceRegistry;
    serviceRegistry.registerService<MetricsService>(new SimpleMetricsService());
    serviceRegistry.registerService<RootFileService>(new LocalRootFileService());
    serviceRegistry.registerService<ControlService>(new TextControlService());

    std::unique_ptr<FairMQDevice> device;
    if (spec.inputs.empty()) {
      LOG(DEBUG) << spec.id << " is a source\n";
      device.reset(new DataSourceDevice(spec, serviceRegistry));
    } else {
      LOG(DEBUG) << spec.id << " is a processor\n";
      device.reset(new DataProcessingDevice(spec, serviceRegistry));
    }

    if (!device) {
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
    for (auto &option : spec.options) {
      if (option.type != option.defaultValue.type()) {
        std::ostringstream ss;
        ss << "Mismatch between declared option type and default value type"
           << "for " << option.name << " in DataProcessorSpec of "
           << spec.name;
        throw std::runtime_error(ss.str());
      }
    }
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
    pid_t pid = waitpid((pid_t)(-1), nullptr, WNOHANG);
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
    {"quiet",     no_argument,  nullptr, 'q' },
    {"stop",   no_argument,  nullptr, 's' },
    {"batch", no_argument, nullptr, 'b'},
    {"graphviz", no_argument, nullptr, 'g'},
    {"dds", no_argument, nullptr, 'D'},
    {"id", required_argument, nullptr, 'i'},
    { nullptr,         0,            nullptr, 0 }
  };

  bool defaultQuiet = false;
  bool defaultStopped = false;
  bool noGui = false;
  bool graphViz = false;
  bool generateDDS = false;
  std::string frameworkId;

  int opt;
  size_t safeArgsSize = sizeof(char**)*argc+1;
  char **safeArgv = reinterpret_cast<char**>(malloc(safeArgsSize));
  memcpy(safeArgv, argv, safeArgsSize);

  while ((opt = getopt_long(argc, argv, "qsbgDi",longopts, nullptr)) != -1) {
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
    case 'D':
        generateDDS = true;
        break;
    case 'i':
        frameworkId = optarg;
        break;
    case ':':
    case '?':
    default: /* '?' */
        // By default we ignore all the other options: we assume they will be
        // need by on of the underlying devices.
        break;
    }
  }

  std::vector<DeviceSpec> deviceSpecs;

  try {
    verifyWorkflow(specs);
    dataProcessorSpecs2DeviceSpecs(specs, deviceSpecs);
    // This should expand nodes so that we can build a consistent DAG.
  } catch (std::runtime_error &e) {
    std::cerr << "Invalid workflow: " << e.what() << std::endl;
    return 1;
  }

  // Up to here, parent and child need to do exactly the same thing. After, we
  // distinguish between something which has a framework id (the children) and something
  // which does not, the parent.
  if (frameworkId.empty() == false) {
    for (auto &spec : deviceSpecs) {
      if (spec.id == frameworkId) {
        return doChild(argc, safeArgv, spec);
      }
    }
    LOG(ERROR) << "Unable to find component with id" << frameworkId;
  }

  assert(frameworkId.empty());

  // Description of the running processes
  gDeviceControls.resize(deviceSpecs.size());

  for (size_t si = 0; si < deviceSpecs.size(); ++si) {
    auto &spec = deviceSpecs[si];
    auto &control = gDeviceControls[si];
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

    // Do filtering. Since we should only have few options,
    // FIXME: finish here...
    for (size_t ai = 0; ai < argc; ++ai) {
      for (size_t oi = 0; oi < spec.options.size(); ++oi) {
        char *currentOpt = argv[ai];
        LOG(DEBUG) << "Checking if " << currentOpt << " is needed by " << spec.id;
        if (ai + 1 > argc) {
          std::cerr << "Missing value for " << currentOpt;
          exit(1);
        }
        char *currentOptValue = argv[ai+1];
        const std::string &validOption = "--" + spec.options[oi].name;
        if (strncmp(currentOpt, validOption.c_str(), validOption.size()) == 0) {
          tmpArgs.emplace_back(strdup(validOption.c_str()));
          tmpArgs.emplace_back(strdup(currentOptValue));
          control.options.insert(std::make_pair(spec.options[oi].name,
                                                currentOptValue));
          break;
        }
      }
    }

    // Add the channel configuration
    for (auto &channel : spec.channels) {
      tmpArgs.emplace_back(std::string("--channel-config"));
      tmpArgs.emplace_back(channel2String(channel));
    }

    // We create the final option list, depending on the channels
    // which are present in a device.
    for (auto &arg : tmpArgs) {
      spec.args.emplace_back(strdup(arg.c_str()));
    }
    // execvp wants a NULL terminated list.
    spec.args.push_back(nullptr);

    //FIXME: this should probably be reflected in the GUI
    std::ostringstream str;
    for (size_t ai = 0; ai < spec.args.size() - 1; ai++) {
      assert(spec.args[ai]);
      str << " " << spec.args[ai];
    }
    LOG(DEBUG) << "The following options are being forwarded to "
               << spec.id << ":" << str.str();
  }

  if (graphViz) {
    // Dump a graphviz representation of what I will do.
    dumpDeviceSpec2Graphviz(std::cout, deviceSpecs);
    exit(0);
  }

  if (generateDDS) {
    dumpDeviceSpec2DDS(deviceSpecs);
    exit(0);
  }

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
  if (sigaction(SIGCHLD, &sa_handle_child, nullptr) == -1) {
    perror(nullptr);
    exit(1);
  }

  for (size_t di = 0; di < deviceSpecs.size(); ++di) {
    auto &spec = deviceSpecs[di];
    auto &control = gDeviceControls[di];
    int childstdout[2];
    int childstderr[2];

    maxFd = createPipes(maxFd, childstdout);
    maxFd = createPipes(maxFd, childstderr);

    // If we have a framework id, it means we have already been respawned
    // and that we are in a child. If not, we need to fork and re-exec, adding
    // the framework-id as one of the options.
    pid_t id = 0;
    id = fork();
    // We are the child: prepare options and reexec.
    if (id == 0) {
      // We allow being debugged and do not terminate on SIGTRAP
      signal(SIGTRAP, SIG_IGN);

      // We do not start the process if control.noStart is set.
      if (control.stopped) {
        kill(getpid(), SIGSTOP);
      }

      // This is the child. We close the read part of the pipe, stdout
      // and dup2 the write part of the pipe on it. Then we can restart.
      close(childstdout[0]);
      close(childstderr[0]);
      close(STDOUT_FILENO);
      close(STDERR_FILENO);
      dup2(childstdout[1], STDOUT_FILENO);
      dup2(childstderr[1], STDERR_FILENO);
      execvp(spec.args[0], spec.args.data());
    }

    // This is the parent. We close the write end of
    // the child pipe and and keep track of the fd so
    // that we can later select on it.
    struct sigaction sa_handle_int;
    sa_handle_int.sa_handler = handle_sigint;
    sigemptyset(&sa_handle_int.sa_mask);
    sa_handle_int.sa_flags = SA_RESTART;
    if (sigaction(SIGINT, &sa_handle_int, nullptr) == -1) {
      perror("Unable to install signal handler");
      exit(1);
    }

    std::cout << "Starting " << spec.id << " on pid " << id << "\n";
    DeviceInfo info;
    info.pid = id;
    info.active = true;
    info.readyToQuit = false;
    info.historySize = 1000;
    info.historyPos = 0;

    socket2DeviceInfo.insert(std::make_pair(childstdout[0], gDeviceInfos.size()));
    socket2DeviceInfo.insert(std::make_pair(childstderr[0], gDeviceInfos.size()));
    gDeviceInfos.emplace_back(info);
    // Let's add also metrics information for the given device
    gDeviceMetricsInfos.emplace_back(DeviceMetricsInfo{});

    close(childstdout[1]);
    close(childstderr[1]);
    FD_SET(childstdout[0], &childFdset);
    FD_SET(childstderr[0], &childFdset);
  }
  maxFd += 1;
  auto exitCode = doParent(&childFdset,
                           maxFd,
                           gDeviceInfos,
                           deviceSpecs,
                           gDeviceControls,
                           gDeviceMetricsInfos,
                           socket2DeviceInfo);
  killChildren(gDeviceInfos);
  return exitCode;
}
