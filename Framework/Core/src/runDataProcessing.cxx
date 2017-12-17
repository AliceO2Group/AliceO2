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
#include "Framework/DeviceExecution.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/FrameworkGUIDebugger.h"
#include "Framework/SimpleMetricsService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/LocalRootFileService.h"
#include "Framework/LogParsingHelpers.h"
#include "Framework/TextControlService.h"
#include "Framework/ParallelContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/SimpleRawDeviceService.h"

#include "GraphvizHelpers.h"
#include "DDSConfigHelpers.h"
#include "DeviceSpecHelpers.h"
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

#include <csignal>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <boost/program_options.hpp>


#include <fairmq/DeviceRunner.h>

using namespace o2::framework;

std::vector<DeviceInfo> gDeviceInfos;
std::vector<DeviceMetricsInfo> gDeviceMetricsInfos;
std::vector<DeviceControl> gDeviceControls;
std::vector<DeviceExecution> gDeviceExecutions;

namespace bpo = boost::program_options;

// FIXME: probably find a better place
// these are the device options added by the framework, but they can be
// overloaded in the config spec
bpo::options_description gHiddenDeviceOptions("Hidden child options");

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
void doParent(fd_set *in_fdset,
             int maxFd,
             std::vector<DeviceInfo> infos,
             std::vector<DeviceSpec> specs,
             std::vector<DeviceControl> controls,
             std::vector<DeviceMetricsInfo> metricsInfos,
             std::map<int,size_t> &socket2Info,
             bool batch) {
  void *window = nullptr;
  decltype(getGUIDebugger(infos, specs, metricsInfos, controls)) debugGUICallback;

  if (batch == false) {
    window = initGUI("O2 Framework debug GUI");
    debugGUICallback = getGUIDebugger(infos, specs, metricsInfos, controls);
  }
  if (batch == false && window == nullptr) {
    LOG(WARN) << "Could not create GUI. Switching to batch mode. Do you have GLFW on your system?";
    batch = true;
  }
  // FIXME: I should really have some way of exiting the
  // parent..
  while (batch || pollGUI(window, debugGUICallback)) {
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
      info.historyLevel.resize(info.historySize);

      while ((pos = s.find(delimiter)) != std::string::npos) {
          token = s.substr(0, pos);
          auto logLevel = LogParsingHelpers::parseTokenLevel(token);

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
                      << " from pid " << info.pid
                      << " valid for " << validFor;
            if (command == "QUIT") {
              if (validFor == "ALL") {
                for (auto &deviceInfo : infos) {
                  deviceInfo.readyToQuit = true;
                }
              }
            }
          } else if (!control.quiet
                     && (strstr(token.c_str(), control.logFilter) != nullptr)
                     && logLevel >= control.logLevel ) {
            assert(info.historyPos >= 0);
            assert(info.historyPos < info.history.size());
            info.history[info.historyPos] = token;
            info.historyLevel[info.historyPos] = logLevel;
            info.historyPos = (info.historyPos + 1) % info.history.size();
            std::cout << "[" << info.pid << "]: " << token << std::endl;
          }
          // We keep track of the maximum log error a
          // device has seen.
          if (logLevel > info.maxLogLevel
              && logLevel > LogParsingHelpers::LogLevel::Info) {
            info.maxLogLevel = logLevel;
          }
          s.erase(0, pos + delimiter.length());
      }
      info.unprinted = s;
    }
    // FIXME: for the gui to work correctly I would actually need to
    //        run the loop more often and update whenever enough time has
    //        passed.
  }
}

int doChild(int argc, char **argv, const o2::framework::DeviceSpec &spec) {
  fair::mq::logger::ReinitLogger(false);

  LOG(INFO) << "Spawing new device " << spec.id
            << " in process with pid " << getpid();
  try {
    fair::mq::DeviceRunner runner{argc, argv};

    // Populate options from the command line. Notice that only the options
    // declared in the workflow definition are allowed.
    runner.AddHook<fair::mq::hooks::SetCustomCmdLineOptions>([&spec](fair::mq::DeviceRunner& r){
      boost::program_options::options_description optsDesc;
      populateBoostProgramOptions(optsDesc, spec.options, gHiddenDeviceOptions);
      r.fConfig.AddToCmdLineOptions(optsDesc, true);
    });

    // We initialise this in the driver, because different drivers might have
    // different versions of the service
    ServiceRegistry serviceRegistry;
    serviceRegistry.registerService<MetricsService>(new SimpleMetricsService());
    serviceRegistry.registerService<RootFileService>(new LocalRootFileService());
    serviceRegistry.registerService<ControlService>(new TextControlService());
    serviceRegistry.registerService<ParallelContext>(new ParallelContext(spec.rank, spec.nSlots));

    std::unique_ptr<FairMQDevice> device;
    serviceRegistry.registerService<RawDeviceService>(new SimpleRawDeviceService(nullptr));

    if (spec.inputs.empty()) {
      LOG(DEBUG) << spec.id << " is a source\n";
      device.reset(new DataSourceDevice(spec, serviceRegistry));
    } else {
      LOG(DEBUG) << spec.id << " is a processor\n";
      device.reset(new DataProcessingDevice(spec, serviceRegistry));
    }

    serviceRegistry.get<RawDeviceService>().setDevice(device.get());

    runner.AddHook<fair::mq::hooks::InstantiateDevice>([&device](fair::mq::DeviceRunner& r){
      r.fDevice = std::shared_ptr<FairMQDevice>{std::move(device)};
    });

    return runner.Run();
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

// Kill all the active children. Exit code
// is != 0 if any of the children had an error.
int killChildren(std::vector<DeviceInfo> &infos) {
  int exitCode = 0;
  for (auto &info : infos) {
    if (exitCode == 0
        && info.maxLogLevel >= LogParsingHelpers::LogLevel::Error){
      LOG(ERROR) << "Child " << info.pid << " had at least one "
                 << "message above severity ERROR";
      exitCode = 1;
    }
    if (!info.active) {
      continue;
    }
    kill(info.pid, SIGKILL);
    int status;
    waitpid(info.pid, &status, 0);
  }
  return exitCode;
}

// FIXME: I should really do this gracefully, by doing the following:
// - Kill all the children
// - Set a sig_atomic_t to say we did.
// - Wait for all the children to exit
// - Return gracefully.
static void handle_sigint(int signum) {
  auto exitCode = killChildren(gDeviceInfos);
  // We kill ourself after having killed all our children (SPOOKY!)
  signal(SIGINT, SIG_DFL);
  kill(getpid(), SIGINT);
}

void handle_sigchld(int sig) {
  int saved_errno = errno;
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
    for (auto &info : gDeviceInfos) {
      if (info.pid == pid) {
        info.active = false;
      }
    }
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
  bpo::options_description executorOptions("Executor options");
  executorOptions.add_options()
    ((std::string("help") + ",h").c_str(),
     "print this help")
    ((std::string("quiet") + ",q").c_str(),
     bpo::value<bool>()->zero_tokens()->default_value(false),
     "quiet operation")
    ((std::string("stop") + ",s").c_str(),
     bpo::value<bool>()->zero_tokens()->default_value(false),
     "stop before device start")
    ((std::string("batch") + ",b").c_str(),
     bpo::value<bool>()->zero_tokens()->default_value(false),
     "batch processing mode")
    ((std::string("graphviz") + ",g").c_str(),
     bpo::value<bool>()->zero_tokens()->default_value(false),
     "produce graph output")
    ((std::string("dds") + ",D").c_str(),
     bpo::value<bool>()->zero_tokens()->default_value(false),
     "create DDS configuration");

  // some of the options must be forwarded by default to the device
  executorOptions.add(DeviceSpecHelpers::getForwardedDeviceOptions());

  gHiddenDeviceOptions.add_options()
    ((std::string("id") + ",i").c_str(),
     bpo::value<std::string>(),
     "device id for child spawning")
    ("channel-config",
     bpo::value<std::vector<std::string>>(),
     "channel configuration")
    ("control",
     "control plugin")
    ("log-color",
     "logging color scheme");

  bpo::options_description visibleOptions;
  visibleOptions.add(executorOptions);
  // Use the hidden options as veto, all config specs matching a definition
  // in the hidden options are skipped in order to avoid duplicate definitions
  // in the main parser. Note: all config specs are forwarded to devices
  visibleOptions.add(prepareOptionDescriptions(specs, gHiddenDeviceOptions));

  bpo::options_description od;
  od.add(visibleOptions);
  od.add(gHiddenDeviceOptions);

  // FIXME: decide about the policy for handling unrecognized arguments
  // command_line_parser with option allow_unregistered() can be used
  bpo::variables_map varmap;
  bpo::store(bpo::parse_command_line(argc, argv, od), varmap);

  bool defaultQuiet = varmap["quiet"].as<bool>();
  bool defaultStopped = varmap["stop"].as<bool>();
  bool batch = varmap["batch"].as<bool>();
  bool graphViz = varmap["graphviz"].as<bool>();
  bool generateDDS = varmap["dds"].as<bool>();
  std::string frameworkId;
  if (varmap.count("id")) frameworkId = varmap["id"].as<std::string>();
  if (varmap.count("help")) {
    bpo::options_description helpOptions;
    helpOptions.add(executorOptions);
    // this time no veto is applied, so all the options are added for printout
    helpOptions.add(prepareOptionDescriptions(specs));
    std::cout << helpOptions << std::endl;
    exit(0);
  }

  std::vector<DeviceSpec> deviceSpecs;

  try {
    DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(specs, deviceSpecs);
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
        return doChild(argc, argv, spec);
      }
    }
    LOG(ERROR) << "Unable to find component with id " << frameworkId;
  }

  assert(frameworkId.empty());

  gDeviceControls.resize(deviceSpecs.size());
  gDeviceExecutions.resize(deviceSpecs.size());

  DeviceSpecHelpers::prepareArguments(
      argc, argv, defaultQuiet,
      defaultStopped, deviceSpecs, gDeviceExecutions, gDeviceControls);

  if (graphViz) {
    // Dump a graphviz representation of what I will do.
    GraphvizHelpers::dumpDeviceSpec2Graphviz(std::cout, deviceSpecs);
    exit(0);
  }

  if (generateDDS) {
    dumpDeviceSpec2DDS(std::cout, deviceSpecs, gDeviceExecutions);
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
    auto &execution = gDeviceExecutions[di];
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
      execvp(execution.args[0], execution.args.data());
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
    info.maxLogLevel = LogParsingHelpers::LogLevel::Debug;

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
  doParent(&childFdset,
           maxFd,
           gDeviceInfos,
           deviceSpecs,
           gDeviceControls,
           gDeviceMetricsInfos,
           socket2DeviceInfo,
           batch);
  return killChildren(gDeviceInfos);
}
