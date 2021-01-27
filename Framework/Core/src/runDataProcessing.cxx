// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/BoostOptionsRetriever.h"
#include "Framework/ChannelConfigurationPolicy.h"
#include "Framework/ChannelMatching.h"
#include "Framework/ConfigParamsHelper.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigContext.h"
#include "Framework/DataProcessingDevice.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Plugins.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceExecution.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceConfigInfo.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceState.h"
#include "Framework/DebugGUI.h"
#include "Framework/LocalRootFileService.h"
#include "Framework/LogParsingHelpers.h"
#include "Framework/Logger.h"
#include "Framework/ParallelContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/SimpleRawDeviceService.h"
#define O2_SIGNPOST_DEFINE_CONTEXT
#include "Framework/Signpost.h"
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Monitoring.h"
#include "Framework/DataProcessorInfo.h"
#include "Framework/DriverInfo.h"
#include "Framework/DriverControl.h"

#include "ComputingResourceHelpers.h"
#include "DataProcessingStatus.h"
#include "DDSConfigHelpers.h"
#include "O2ControlHelpers.h"
#include "DeviceSpecHelpers.h"
#include "GraphvizHelpers.h"
#include "PropertyTreeHelpers.h"
#include "SimpleResourceManager.h"
#include "WorkflowSerializationHelpers.h"

#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>
#include <Monitoring/MonitoringFactory.h>
#include <InfoLogger/InfoLogger.hxx>
#include "ResourcesMonitoringHelper.h"

#include "FairMQDevice.h"
#include <fairmq/DeviceRunner.h>
#if __has_include(<fairmq/shmem/Monitor.h>)
#include <fairmq/shmem/Monitor.h>
#endif
#include "options/FairMQProgOptions.h"

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <uv.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <type_traits>
#include <tuple>
#include <chrono>
#include <utility>
#include <numeric>
#include <functional>

#include <fcntl.h>
#include <netinet/ip.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <execinfo.h>
#if defined(__linux__) && __has_include(<sched.h>)
#include <sched.h>
#elif __has_include(<linux/getcpu.h>)
#include <linux/getcpu.h>
#elif __has_include(<cpuid.h>)
#include <cpuid.h>
#define CPUID(INFO, LEAF, SUBLEAF) __cpuid_count(LEAF, SUBLEAF, INFO[0], INFO[1], INFO[2], INFO[3])
#define GETCPU(CPU)                                 \
  {                                                 \
    uint32_t CPUInfo[4];                            \
    CPUID(CPUInfo, 1, 0);                           \
    /* CPUInfo[1] is EBX, bits 24-31 are APIC ID */ \
    if ((CPUInfo[3] & (1 << 9)) == 0) {             \
      CPU = -1; /* no APIC on chip */               \
    } else {                                        \
      CPU = (unsigned)CPUInfo[1] >> 24;             \
    }                                               \
    if (CPU < 0)                                    \
      CPU = 0;                                      \
  }
#endif

using namespace o2::monitoring;
using namespace o2::configuration;
using namespace AliceO2::InfoLogger;

using namespace o2::framework;
namespace bpo = boost::program_options;
using DataProcessorInfos = std::vector<DataProcessorInfo>;
using DeviceExecutions = std::vector<DeviceExecution>;
using DeviceSpecs = std::vector<DeviceSpec>;
using DeviceInfos = std::vector<DeviceInfo>;
using DeviceControls = std::vector<DeviceControl>;
using DataProcessorSpecs = std::vector<DataProcessorSpec>;

template class std::vector<DeviceSpec>;

std::vector<DeviceMetricsInfo> gDeviceMetricsInfos;

// FIXME: probably find a better place
// these are the device options added by the framework, but they can be
// overloaded in the config spec
bpo::options_description gHiddenDeviceOptions("Hidden child options");

// To be used to allow specifying the TerminationPolicy on the command line.
namespace o2
{
namespace framework
{
std::istream& operator>>(std::istream& in, enum TerminationPolicy& policy)
{
  std::string token;
  in >> token;
  if (token == "quit") {
    policy = TerminationPolicy::QUIT;
  } else if (token == "wait") {
    policy = TerminationPolicy::WAIT;
  } else {
    in.setstate(std::ios_base::failbit);
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, const enum TerminationPolicy& policy)
{
  if (policy == TerminationPolicy::QUIT) {
    out << "quit";
  } else if (policy == TerminationPolicy::WAIT) {
    out << "wait";
  } else {
    out.setstate(std::ios_base::failbit);
  }
  return out;
}
} // namespace framework
} // namespace o2

size_t current_time_with_ms()
{
  long ms;  // Milliseconds
  time_t s; // Seconds
  struct timespec spec;

  clock_gettime(CLOCK_REALTIME, &spec);

  s = spec.tv_sec;
  ms = round(spec.tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds
  if (ms > 999) {
    s++;
    ms = 0;
  }
  return s * 1000 + ms;
}

// Read from a given fd and print it.
// return true if we can still read from it,
// return false if we need to close the input pipe.
//
// FIXME: We should really print full lines.
void getChildData(int infd, DeviceInfo& outinfo)
{
  char buffer[1024 * 16];
  int bytes_read;
  // NOTE: do not quite understand read ends up blocking if I read more than
  //        once. Oh well... Good enough for now.
  O2_SIGNPOST_START(DriverStatus::ID, DriverStatus::BYTES_READ, outinfo.pid, infd, 0);
  while (true) {
    bytes_read = read(infd, buffer, 1024 * 16);
    if (bytes_read == 0) {
      O2_SIGNPOST_END(DriverStatus::ID, DriverStatus::BYTES_READ, bytes_read, 0, 0);
      return;
    }
    if (bytes_read < 0) {
      switch (errno) {
        case EWOULDBLOCK:
          O2_SIGNPOST_END(DriverStatus::ID, DriverStatus::BYTES_READ, bytes_read, 0, 0);
          return;
        default:
          O2_SIGNPOST_END(DriverStatus::ID, DriverStatus::BYTES_READ, bytes_read, 0, 0);
          return;
      }
    }
    assert(bytes_read > 0);
    outinfo.unprinted.append(buffer, bytes_read);
  }
}

/// Return true if all the DeviceInfo in \a infos are
/// ready to quit. false otherwise.
/// FIXME: move to an helper class
bool checkIfCanExit(std::vector<DeviceInfo> const& infos)
{
  if (infos.empty()) {
    return false;
  }
  for (auto& info : infos) {
    if (info.readyToQuit == false) {
      return false;
    }
  }
  return true;
}

// Kill all the active children. Exit code
// is != 0 if any of the children had an error.
void killChildren(std::vector<DeviceInfo>& infos, int sig)
{
  for (auto& info : infos) {
    if (info.active == true) {
      kill(info.pid, sig);
    }
  }
}

/// Check the state of the children
bool areAllChildrenGone(std::vector<DeviceInfo>& infos)
{
  for (auto& info : infos) {
    if ((info.pid != 0) && info.active) {
      return false;
    }
  }
  return true;
}

/// Calculate exit code
int calculateExitCode(DeviceSpecs& deviceSpecs, DeviceInfos& infos)
{
  std::regex regexp(R"(^\[([\d+:]*)\]\[\w+\] )");
  int exitCode = 0;
  for (size_t di = 0; di < deviceSpecs.size(); ++di) {
    auto& info = infos[di];
    auto& spec = deviceSpecs[di];
    if (exitCode == 0 && info.maxLogLevel >= LogParsingHelpers::LogLevel::Error) {
      LOG(ERROR) << "SEVERE: Device " << spec.name << " (" << info.pid << ") had at least one "
                 << "message above severity ERROR: " << std::regex_replace(info.firstError, regexp, "");
      exitCode = 1;
    }
  }
  return exitCode;
}

void createPipes(int* pipes)
{
  auto p = pipe(pipes);

  if (p == -1) {
    std::cerr << "Unable to create PIPE: ";
    switch (errno) {
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
    kill(-1 * getpid(), SIGKILL);
  }
}

// We don't do anything in the signal handler but
// we simply note down the fact a signal arrived.
// All the processing is done by the state machine.
volatile sig_atomic_t graceful_exit = false;
volatile sig_atomic_t forceful_exit = false;
volatile sig_atomic_t sigchld_requested = false;

static void handle_sigint(int)
{
  if (graceful_exit == false) {
    graceful_exit = true;
  } else {
    forceful_exit = true;
  }
}

/// Helper to invoke shared memory cleanup
void cleanupSHM(std::string const& uniqueWorkflowId)
{
#if __has_include(<fairmq/shmem/Monitor.h>)
  using namespace fair::mq::shmem;
  Monitor::Cleanup(SessionId{"dpl_" + uniqueWorkflowId}, false);
#else
  // Old code, invoking external fairmq-shmmonitor
  auto shmCleanup = fmt::format("fairmq-shmmonitor --cleanup -s dpl_{} 2>&1 >/dev/null", uniqueWorkflowId);
  LOG(debug)
    << "Cleaning up shm memory session with " << shmCleanup;
  auto result = system(shmCleanup.c_str());
  if (result != 0) {
    LOG(error) << "Unable to cleanup shared memory, run " << shmCleanup << "by hand to fix";
  }
#endif
}

static void handle_sigchld(int) { sigchld_requested = true; }

void spawnRemoteDevice(std::string const& forwardedStdin,
                       DeviceSpec const& spec,
                       DeviceControl& control,
                       DeviceExecution& execution,
                       std::vector<DeviceInfo>& deviceInfos)
{
  LOG(INFO) << "Starting " << spec.id << " as remote device";
  DeviceInfo info;
  // FIXME: we should make sure we do not sent a kill to pid 0.
  info.pid = 0;
  info.active = true;
  info.readyToQuit = false;
  info.historySize = 1000;
  info.historyPos = 0;
  info.maxLogLevel = LogParsingHelpers::LogLevel::Debug;
  info.dataRelayerViewIndex = Metric2DViewIndex{"data_relayer", 0, 0, {}};
  info.variablesViewIndex = Metric2DViewIndex{"matcher_variables", 0, 0, {}};
  info.queriesViewIndex = Metric2DViewIndex{"data_queries", 0, 0, {}};

  deviceInfos.emplace_back(info);
  // Let's add also metrics information for the given device
  gDeviceMetricsInfos.emplace_back(DeviceMetricsInfo{});
}

struct DeviceLogContext {
  int fd;
  int index;
  std::vector<DeviceInfo>* infos;
};

void log_callback(uv_poll_t* handle, int status, int events)
{
  DeviceLogContext* logContext = reinterpret_cast<DeviceLogContext*>(handle->data);
  std::vector<DeviceInfo>* infos = logContext->infos;
  DeviceInfo& info = infos->at(logContext->index);

  if (status < 0) {
    info.active = false;
  }
  if (events & UV_READABLE) {
    getChildData(logContext->fd, info);
  }
  if (events & UV_DISCONNECT) {
    info.active = false;
  }
}

/// This will start a new device by forking and executing a
/// new child
void spawnDevice(std::string const& forwardedStdin,
                 DeviceSpec const& spec,
                 DriverInfo& driverInfo,
                 DeviceControl& control,
                 DeviceExecution& execution,
                 std::vector<DeviceInfo>& deviceInfos,
                 ServiceRegistry& serviceRegistry,
                 boost::program_options::variables_map& varmap,
                 uv_loop_t* loop,
                 std::vector<uv_poll_t*> handles,
                 unsigned parentCPU,
                 unsigned parentNode)
{
  int childstdin[2];
  int childstdout[2];
  int childstderr[2];

  createPipes(childstdin);
  createPipes(childstdout);
  createPipes(childstderr);

  // FIXME: this might not work when more than one DPL driver on the same
  // machine. Hopefully we do not care.
  // Not how the first port is actually used to broadcast clients.
  driverInfo.tracyPort++;

  for (auto& service : spec.services) {
    if (service.preFork != nullptr) {
      service.preFork(serviceRegistry, varmap);
    }
  }
  // If we have a framework id, it means we have already been respawned
  // and that we are in a child. If not, we need to fork and re-exec, adding
  // the framework-id as one of the options.
  pid_t id = 0;
  id = fork();
  // We are the child: prepare options and reexec.
  if (id == 0) {
    // We allow being debugged and do not terminate on SIGTRAP
    signal(SIGTRAP, SIG_IGN);

    // This is the child.
    // For stdout / stderr, we close the read part of the pipe, the
    // old descriptor, and then replace it with the write part of the pipe.
    // For stdin, we close the write part of the pipe, the old descriptor,
    // and then we replace it with the read part of the pipe.
    close(childstdin[1]);
    close(childstdout[0]);
    close(childstderr[0]);
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);
    dup2(childstdin[0], STDIN_FILENO);
    dup2(childstdout[1], STDOUT_FILENO);
    dup2(childstderr[1], STDERR_FILENO);
    auto portS = std::to_string(driverInfo.tracyPort);
    setenv("TRACY_PORT", portS.c_str(), 1);
    for (auto& service : spec.services) {
      if (service.postForkChild != nullptr) {
        service.postForkChild(serviceRegistry);
      }
    }
    for (auto& env : execution.environ) {
      char* formatted = strdup(fmt::format(env,
                                           fmt::arg("timeslice0", spec.inputTimesliceId),
                                           fmt::arg("timeslice1", spec.inputTimesliceId + 1),
                                           fmt::arg("timeslice4", spec.inputTimesliceId + 4))
                                 .c_str());
      putenv(formatted);
    }
    execvp(execution.args[0], execution.args.data());
  }
  if (varmap.count("post-fork-command")) {
    auto templateCmd = varmap["post-fork-command"];
    auto cmd = fmt::format(templateCmd.as<std::string>(),
                           fmt::arg("pid", id),
                           fmt::arg("id", spec.id),
                           fmt::arg("cpu", parentCPU),
                           fmt::arg("node", parentNode),
                           fmt::arg("name", spec.name),
                           fmt::arg("timeslice0", spec.inputTimesliceId),
                           fmt::arg("timeslice1", spec.inputTimesliceId + 1),
                           fmt::arg("rank0", spec.rank),
                           fmt::arg("maxRank0", spec.nSlots));
    int err = system(cmd.c_str());
    if (err) {
      LOG(error) << "Post fork command `" << cmd << "` returned with status " << err;
    }
    LOG(debug) << "Successfully executed `" << cmd;
  }
  // This is the parent. We close the write end of
  // the child pipe and and keep track of the fd so
  // that we can later select on it.
  for (auto& service : spec.services) {
    if (service.postForkParent != nullptr) {
      service.postForkParent(serviceRegistry);
    }
  }

  struct sigaction sa_handle_int;
  sa_handle_int.sa_handler = handle_sigint;
  sigemptyset(&sa_handle_int.sa_mask);
  sa_handle_int.sa_flags = SA_RESTART;
  if (sigaction(SIGINT, &sa_handle_int, nullptr) == -1) {
    perror("Unable to install signal handler");
    exit(1);
  }

  LOG(INFO) << "Starting " << spec.id << " on pid " << id;
  DeviceInfo info;
  info.pid = id;
  info.active = true;
  info.readyToQuit = false;
  info.historySize = 1000;
  info.historyPos = 0;
  info.maxLogLevel = LogParsingHelpers::LogLevel::Debug;
  info.dataRelayerViewIndex = Metric2DViewIndex{"data_relayer", 0, 0, {}};
  info.variablesViewIndex = Metric2DViewIndex{"matcher_variables", 0, 0, {}};
  info.queriesViewIndex = Metric2DViewIndex{"data_queries", 0, 0, {}};
  info.tracyPort = driverInfo.tracyPort;

  deviceInfos.emplace_back(info);
  // Let's add also metrics information for the given device
  gDeviceMetricsInfos.emplace_back(DeviceMetricsInfo{});

  close(childstdin[0]);
  close(childstdout[1]);
  close(childstderr[1]);
  size_t result = write(childstdin[1], forwardedStdin.data(), forwardedStdin.size());
  if (result != forwardedStdin.size()) {
    LOG(ERROR) << "Unable to pass configuration to children";
  }
  close(childstdin[1]); // Not allowing further communication...

  // Setting them to non-blocking to avoid haing the driver hang when
  // reading from child.
  int resultCode = fcntl(childstdout[0], F_SETFL, O_NONBLOCK);
  if (resultCode == -1) {
    LOGP(ERROR, "Error while setting the socket to non-blocking: {}", strerror(errno));
  }
  resultCode = fcntl(childstderr[0], F_SETFL, O_NONBLOCK);
  if (resultCode == -1) {
    LOGP(ERROR, "Error while setting the socket to non-blocking: {}", strerror(errno));
  }
  /// Add pollers for stdout and stderr
  auto addPoller = [&handles, &deviceInfos, &loop](int index, int fd) {
    DeviceLogContext* context = new DeviceLogContext{};
    context->index = index;
    context->fd = fd;
    context->infos = &deviceInfos;
    handles.push_back((uv_poll_t*)malloc(sizeof(uv_poll_t)));
    auto handle = handles.back();
    handle->data = context;
    uv_poll_init(loop, handle, fd);
    uv_poll_start(handle, UV_READABLE, log_callback);
  };

  addPoller(deviceInfos.size() - 1, childstdout[0]);
  addPoller(deviceInfos.size() - 1, childstderr[0]);
}

void updateMetricsNames(DriverInfo& driverInfo, std::vector<DeviceMetricsInfo> const& metricsInfos)
{
  // Calculate the unique set of metrics, as available in the metrics service
  static std::unordered_set<std::string> allMetricsNames;
  for (const auto& metricsInfo : metricsInfos) {
    for (const auto& labelsPairs : metricsInfo.metricLabelsIdx) {
      allMetricsNames.insert(std::string(labelsPairs.label));
    }
  }
  for (const auto& labelsPairs : driverInfo.metrics.metricLabelsIdx) {
    allMetricsNames.insert(std::string(labelsPairs.label));
  }
  std::vector<std::string> result(allMetricsNames.begin(), allMetricsNames.end());
  std::sort(result.begin(), result.end());
  driverInfo.availableMetrics.swap(result);
}

struct LogProcessingState {
  bool didProcessLog = false;
  bool didProcessControl = true;
  bool didProcessConfig = true;
  bool didProcessMetric = false;
  bool hasNewMetric = false;
};

void processCommand(DeviceInfos& infos, pid_t pid, std::string const& command, std::string const& arg)
{
  auto doToMatchingPid = [](std::vector<DeviceInfo>& infos, pid_t pid, auto lambda) {
    for (auto& deviceInfo : infos) {
      if (deviceInfo.pid == pid) {
        lambda(deviceInfo);
        break;
      }
    }
  };
  LOGP(debug, "Found control command {} from pid {} with argument {}.", command, pid, arg);
  if (command == "QUIT" && arg == "ALL") {
    for (auto& deviceInfo : infos) {
      deviceInfo.readyToQuit = true;
    }
  } else if (command == "QUIT" && arg == "ME") {
    doToMatchingPid(infos, pid, [](DeviceInfo& info) { info.readyToQuit = true; });
  } else if (command == "NOTIFY_STREAMING_STATE" && arg == "IDLE") {
    // FIXME: this should really be a policy...
    doToMatchingPid(infos, pid, [](DeviceInfo& info) { info.readyToQuit = true; info.streamingState = StreamingState::Idle; });
  } else if (command == "NOTIFY_STREAMING_STATE" && arg == "STREAMING") {
    // FIXME: this should really be a policy...
    doToMatchingPid(infos, pid, [](DeviceInfo& info) { info.streamingState = StreamingState::Streaming; });
  } else if (command == "NOTIFY_STREAMING_STATE" && arg == "EOS") {
    // FIXME: this should really be a policy...
    doToMatchingPid(infos, pid, [](DeviceInfo& info) { info.streamingState = StreamingState::EndOfStreaming; });
  }
};

LogProcessingState processChildrenOutput(DriverInfo& driverInfo,
                                         DeviceInfos& infos,
                                         DeviceSpecs const& specs,
                                         DeviceControls& controls,
                                         std::vector<DeviceMetricsInfo>& metricsInfos)
{
  // Display part. All you need to display should actually be in
  // `infos`.
  // TODO: split at \n
  // TODO: update this only once per 1/60 of a second or
  // things like this.
  // TODO: have multiple display modes
  // TODO: graphical view of the processing?
  assert(infos.size() == controls.size());
  std::smatch match;
  ParsedMetricMatch metricMatch;
  ParsedConfigMatch configMatch;
  const std::string delimiter("\n");
  bool hasNewMetric = false;
  LogProcessingState result;

  for (size_t di = 0, de = infos.size(); di < de; ++di) {
    DeviceInfo& info = infos[di];
    DeviceControl& control = controls[di];
    DeviceMetricsInfo& metrics = metricsInfos[di];
    assert(specs.size() == infos.size());
    DeviceSpec const& spec = specs[di];

    if (info.unprinted.empty()) {
      continue;
    }

    O2_SIGNPOST_START(DriverStatus::ID, DriverStatus::BYTES_PROCESSED, info.pid, 0, 0);

    std::string_view s = info.unprinted;
    size_t pos = 0;
    info.history.resize(info.historySize);
    info.historyLevel.resize(info.historySize);

    auto updateMetricsViews =
      Metric2DViewIndex::getUpdater({&info.dataRelayerViewIndex,
                                     &info.variablesViewIndex,
                                     &info.queriesViewIndex});

    auto newMetricCallback = [&updateMetricsViews, &driverInfo, &metricsInfos, &hasNewMetric](std::string const& name, MetricInfo const& metric, int value, size_t metricIndex) {
      updateMetricsViews(name, metric, value, metricIndex);
      hasNewMetric = true;
    };


    while ((pos = s.find(delimiter)) != std::string::npos) {
      std::string token{s.substr(0, pos)};
      auto logLevel = LogParsingHelpers::parseTokenLevel(token);

      // Check if the token is a metric from SimpleMetricsService
      // if yes, we do not print it out and simply store it to be displayed
      // in the GUI.
      // Then we check if it is part of our Poor man control system
      // if yes, we execute the associated command.
      if (DeviceMetricsHelper::parseMetric(token, metricMatch)) {
        // We use this callback to cache which metrics are needed to provide a
        // the DataRelayer view.
        DeviceMetricsHelper::processMetric(metricMatch, metrics, newMetricCallback);
        result.didProcessMetric = true;
      } else if (logLevel == LogParsingHelpers::LogLevel::Info && parseControl(token, match)) {
        processCommand(infos, info.pid, match[1].str(), match[2].str());
        result.didProcessControl = true;
      } else if (logLevel == LogParsingHelpers::LogLevel::Info && DeviceConfigHelper::parseConfig(token, configMatch)) {
        DeviceConfigHelper::processConfig(configMatch, info);
        result.didProcessConfig = true;
      } else if (!control.quiet && (token.find(control.logFilter) != std::string::npos) &&
                 logLevel >= control.logLevel) {
        assert(info.historyPos >= 0);
        assert(info.historyPos < info.history.size());
        info.history[info.historyPos] = token;
        info.historyLevel[info.historyPos] = logLevel;
        info.historyPos = (info.historyPos + 1) % info.history.size();
        std::cout << "[" << info.pid << ":" << spec.name << "]: " << token << std::endl;
        result.didProcessLog = true;
      }
      // We keep track of the maximum log error a
      // device has seen.
      if (logLevel > info.maxLogLevel && logLevel > LogParsingHelpers::LogLevel::Info &&
          logLevel != LogParsingHelpers::LogLevel::Unknown) {
        info.maxLogLevel = logLevel;
      }
      if (logLevel == LogParsingHelpers::LogLevel::Error) {
        info.lastError = token;
        if (info.firstError.empty()) {
          info.firstError = token;
        }
      }
      s.remove_prefix(pos + delimiter.length());
    }
    size_t oldSize = info.unprinted.size();
    info.unprinted = std::string(s);
    O2_SIGNPOST_END(DriverStatus::ID, DriverStatus::BYTES_PROCESSED, oldSize - info.unprinted.size(), 0, 0);
  }
  result.hasNewMetric = hasNewMetric;
  if (hasNewMetric) {
    hasNewMetric = false;
    updateMetricsNames(driverInfo, metricsInfos);
  }
  return result;
}

// Process all the sigchld which are pending
// @return wether or not a given child exited with an error condition.
bool processSigChild(DeviceInfos& infos)
{
  bool hasError = false;
  while (true) {
    int status;
    pid_t pid = waitpid((pid_t)(-1), &status, WNOHANG);
    if (pid > 0) {
      int es = WEXITSTATUS(status);

      if (es) {
        hasError = true;
      }
      for (auto& info : infos) {
        if (info.pid == pid) {
          info.active = false;
          info.exitStatus = es;
        }
      }
      continue;
    } else {
      break;
    }
  }
  return hasError;
}

void doDPLException(RuntimeErrorRef& e)
{
  auto& err = o2::framework::error_from_ref(e);
  if (err.maxBacktrace != 0) {
    LOG(ERROR) << "Unhandled o2::framework::runtime_error reached the top of main, device shutting down. Details follow: \n";
    backtrace_symbols_fd(err.backtrace, err.maxBacktrace, STDERR_FILENO);
  } else {
    LOG(ERROR) << "Unhandled o2::framework::runtime_error reached the top of main, device shutting down."
                  " Recompile with DPL_ENABLE_BACKTRACE=1 to get more information.";
  }
}

void doUnknownException(std::string const& s)
{
  if (s.empty()) {
    LOG(ERROR) << "Unknown error while setting up workflow.";
  } else {
    LOG(ERROR) << "error while setting up workflow: " << s;
  }
}

void doDefaultWorkflowTerminationHook()
{
  //LOG(INFO) << "Process " << getpid() << " is exiting.";
}

int doChild(int argc, char** argv, ServiceRegistry& serviceRegistry, const o2::framework::DeviceSpec& spec, TerminationPolicy errorPolicy,
            uv_loop_t* loop)
{
  fair::Logger::SetConsoleColor(false);
  LOG(INFO) << "Spawing new device " << spec.id << " in process with pid " << getpid();

  try {
    fair::mq::DeviceRunner runner{argc, argv};

    // Populate options from the command line. Notice that only the options
    // declared in the workflow definition are allowed.
    runner.AddHook<fair::mq::hooks::SetCustomCmdLineOptions>([&spec](fair::mq::DeviceRunner& r) {
      boost::program_options::options_description optsDesc;
      ConfigParamsHelper::populateBoostProgramOptions(optsDesc, spec.options, gHiddenDeviceOptions);
      optsDesc.add_options()("monitoring-backend", bpo::value<std::string>()->default_value("infologger://"), "monitoring backend info") //
        ("infologger-severity", bpo::value<std::string>()->default_value(""), "minimum FairLogger severity to send to InfoLogger")       //
        ("configuration,cfg", bpo::value<std::string>()->default_value("command-line"), "configuration backend")                         //
        ("infologger-mode", bpo::value<std::string>()->default_value(""), "INFOLOGGER_MODE override");
      r.fConfig.AddToCmdLineOptions(optsDesc, true);
    });

    // This is to control lifetime. All these services get destroyed
    // when the runner is done.
    std::unique_ptr<SimpleRawDeviceService> simpleRawDeviceService;
    std::unique_ptr<DeviceState> deviceState;

    auto afterConfigParsingCallback = [&simpleRawDeviceService,
                                       &spec,
                                       &serviceRegistry,
                                       &deviceState,
                                       &errorPolicy,
                                       &loop](fair::mq::DeviceRunner& r) {
      deviceState = std::make_unique<DeviceState>();
      deviceState->loop = loop;

      simpleRawDeviceService = std::make_unique<SimpleRawDeviceService>(nullptr, spec);

      serviceRegistry.registerService(ServiceRegistryHelpers::handleForService<RawDeviceService>(simpleRawDeviceService.get()));
      serviceRegistry.registerService(ServiceRegistryHelpers::handleForService<DeviceSpec>(&spec));

      // The decltype stuff is to be able to compile with both new and old
      // FairMQ API (one which uses a shared_ptr, the other one a unique_ptr.
      decltype(r.fDevice) device;
      device = std::move(make_matching<decltype(device), DataProcessingDevice>(spec, serviceRegistry, *deviceState.get()));
      dynamic_cast<DataProcessingDevice*>(device.get())->SetErrorPolicy(errorPolicy);

      serviceRegistry.get<RawDeviceService>().setDevice(device.get());
      r.fDevice = std::move(device);
      fair::Logger::SetConsoleColor(false);

      /// Create all the requested services and initialise them
      for (auto& service : spec.services) {
        LOG(debug) << "Declaring service " << service.name;
        serviceRegistry.declareService(service, *deviceState.get(), r.fConfig);
      }
      if (ResourcesMonitoringHelper::isResourcesMonitoringEnabled(spec.resourceMonitoringInterval)) {
        serviceRegistry.get<Monitoring>().enableProcessMonitoring(spec.resourceMonitoringInterval);
      }
    };

    runner.AddHook<fair::mq::hooks::InstantiateDevice>(afterConfigParsingCallback);
    return runner.Run();
  } catch (boost::exception& e) {
    LOG(ERROR) << "Unhandled boost::exception reached the top of main, device shutting down. Details follow: \n"
               << boost::current_exception_diagnostic_information(true);
    return 1;
  } catch (o2::framework::RuntimeErrorRef e) {
    doDPLException(e);
    return 1;
  } catch (std::exception& e) {
    LOG(ERROR) << "Unhandled std::exception reached the top of main: " << e.what() << ", device shutting down.";
    return 1;
  } catch (...) {
    LOG(ERROR) << "Unknown exception reached the top of main.\n";
    return 1;
  }
  return 0;
}

struct WorkflowInfo {
  std::string executable;
  std::vector<std::string> args;
  std::vector<ConfigParamSpec> options;
};

struct GuiCallbackContext {
  uint64_t frameLast;
  float* frameLatency;
  float* frameCost;
  DebugGUI* plugin;
  void* window;
  bool* guiQuitRequested;
  std::function<void(void)> callback;
};

void gui_callback(uv_timer_s* ctx)
{
  GuiCallbackContext* gui = reinterpret_cast<GuiCallbackContext*>(ctx->data);
  if (gui->plugin == nullptr) {
    return;
  }
  uint64_t frameStart = uv_hrtime();
  uint64_t frameLatency = frameStart - gui->frameLast;
  *(gui->guiQuitRequested) = (gui->plugin->pollGUI(gui->window, gui->callback) == false);
  uint64_t frameEnd = uv_hrtime();
  *(gui->frameCost) = (frameEnd - frameStart) / 1000000;
  *(gui->frameLatency) = frameLatency / 1000000;
  gui->frameLast = frameStart;
}

/// Force single stepping of the children
void single_step_callback(uv_timer_s* ctx)
{
  DeviceInfos* infos = reinterpret_cast<DeviceInfos*>(ctx->data);
  killChildren(*infos, SIGUSR1);
}

// This is the handler for the parent inner loop.
int runStateMachine(DataProcessorSpecs const& workflow,
                    WorkflowInfo const& workflowInfo,
                    DataProcessorInfos const& previousDataProcessorInfos,
                    DriverControl& driverControl,
                    DriverInfo& driverInfo,
                    std::vector<DeviceMetricsInfo>& metricsInfos,
                    boost::program_options::variables_map& varmap,
                    std::string frameworkId)
{
  DeviceSpecs deviceSpecs;
  DeviceInfos infos;
  DeviceControls controls;
  DeviceExecutions deviceExecutions;
  DataProcessorInfos dataProcessorInfos = previousDataProcessorInfos;

  std::vector<uv_poll_t*> pollHandles;
  std::vector<ComputingResource> resources;

  if (driverInfo.resources != "") {
    resources = ComputingResourceHelpers::parseResources(driverInfo.resources);
  } else {
    resources = {ComputingResourceHelpers::getLocalhostResource()};
  }

  auto resourceManager = std::make_unique<SimpleResourceManager>(resources);

  DebugGUI* debugGUI = nullptr;
  void* window = nullptr;
  decltype(debugGUI->getGUIDebugger(infos, deviceSpecs, dataProcessorInfos, metricsInfos, driverInfo, controls, driverControl)) debugGUICallback;

  // An empty frameworkId means this is the driver, so we initialise the GUI
  if (driverInfo.batch == false && frameworkId.empty()) {
    auto initDebugGUI = []() -> DebugGUI* {
      uv_lib_t supportLib;
      int result = 0;
  #ifdef __APPLE__
      result = uv_dlopen("libO2FrameworkGUISupport.dylib", &supportLib);
  #else
      result = uv_dlopen("libO2FrameworkGUISupport.so", &supportLib);
  #endif
      if (result == -1) {
        LOG(ERROR) << uv_dlerror(&supportLib);
        return nullptr;
      }
      void* callback = nullptr;
      DPLPluginHandle* (*dpl_plugin_callback)(DPLPluginHandle*);

      result = uv_dlsym(&supportLib, "dpl_plugin_callback", (void**)&dpl_plugin_callback);
      if (result == -1) {
        LOG(ERROR) << uv_dlerror(&supportLib);
        return nullptr;
      }
      DPLPluginHandle* pluginInstance = dpl_plugin_callback(nullptr);
      return PluginManager::getByName<DebugGUI>(pluginInstance, "ImGUIDebugGUI");
    };
    debugGUI = initDebugGUI();
    if (debugGUI) {
      window = debugGUI->initGUI("O2 Framework debug GUI");
    }
  }
  if (driverInfo.batch == false && window == nullptr && frameworkId.empty()) {
    LOG(WARN) << "Could not create GUI. Switching to batch mode. Do you have GLFW on your system?";
    driverInfo.batch = true;
  }
  bool guiQuitRequested = false;
  bool hasError = false;

  // FIXME: I should really have some way of exiting the
  // parent..
  DriverState current;
  DriverState previous;

  uv_loop_t* loop = uv_loop_new();
  uv_idle_t idler;

  uv_timer_t gui_timer;
  if (window) {
    uv_timer_init(loop, &gui_timer);
  }

  GuiCallbackContext guiContext;
  guiContext.plugin = debugGUI;
  guiContext.frameLast = uv_hrtime();
  guiContext.frameLatency = &driverInfo.frameLatency;
  guiContext.frameCost = &driverInfo.frameCost;
  guiContext.guiQuitRequested = &guiQuitRequested;
  auto inputProcessingLast = guiContext.frameLast;

  uv_timer_t force_step_timer;
  uv_timer_init(loop, &force_step_timer);

  // We initialise this in the driver, because different drivers might have
  // different versions of the service
  ServiceRegistry serviceRegistry;
  std::vector<ServiceMetricHandling> metricProcessingCallbacks;
  std::vector<ServicePreSchedule> preScheduleCallbacks;
  std::vector<ServicePostSchedule> postScheduleCallbacks;

  bool guiDeployedOnce = false;

  while (true) {
    // If control forced some transition on us, we push it to the queue.
    if (driverControl.forcedTransitions.empty() == false) {
      for (auto transition : driverControl.forcedTransitions) {
        driverInfo.states.push_back(transition);
      }
      driverControl.forcedTransitions.resize(0);
    }
    // In case a timeout was requested, we check if we are running
    // for more than the timeout duration and exit in case that's the case.
    {
      auto currentTime = uv_hrtime();
      uint64_t diff = (currentTime - driverInfo.startTime) / 1000000000LL;
      if ((graceful_exit == false) && (driverInfo.timeout > 0) && (diff > driverInfo.timeout)) {
        LOG(INFO) << "Timout ellapsed. Requesting to quit.";
        graceful_exit = true;
      }
    }
    // Move to exit loop if sigint was sent we execute this only once.
    if (graceful_exit == true && driverInfo.sigintRequested == false) {
      driverInfo.sigintRequested = true;
      driverInfo.states.resize(0);
      driverInfo.states.push_back(DriverState::QUIT_REQUESTED);
    }
    // If one of the children dies and sigint was not requested
    // we should decide what to do.
    if (sigchld_requested == true && driverInfo.sigchldRequested == false) {
      driverInfo.sigchldRequested = true;
      driverInfo.states.push_back(DriverState::HANDLE_CHILDREN);
    }
    if (driverInfo.states.empty() == false) {
      previous = current;
      current = driverInfo.states.back();
    } else {
      current = DriverState::UNKNOWN;
    }
    driverInfo.states.pop_back();
    switch (current) {
      case DriverState::INIT:
        LOG(INFO) << "Initialising O2 Data Processing Layer";

        // Install signal handler for quitting children.
        driverInfo.sa_handle_child.sa_handler = &handle_sigchld;
        sigemptyset(&driverInfo.sa_handle_child.sa_mask);
        driverInfo.sa_handle_child.sa_flags = SA_RESTART | SA_NOCLDSTOP;
        if (sigaction(SIGCHLD, &driverInfo.sa_handle_child, nullptr) == -1) {
          perror(nullptr);
          exit(1);
        }

        /// Cleanup the shared memory for the uniqueWorkflowId, in
        /// case we are unlucky and an old one is already present.
        if (driverInfo.noSHMCleanup) {
          LOGP(warning, "Not cleaning up shared memory.");
        } else {
          cleanupSHM(driverInfo.uniqueWorkflowId);
        }
        /// After INIT we go into RUNNING and eventually to SCHEDULE from
        /// there and back into running. This is because the general case
        /// would be that we start an application and then we wait for
        /// resource offers from DDS or whatever resource manager we use.
        driverInfo.states.push_back(DriverState::RUNNING);
        //        driverInfo.states.push_back(DriverState::REDEPLOY_GUI);
        LOG(INFO) << "O2 Data Processing Layer initialised. We brake for nobody.";
#ifdef NDEBUG
        LOGF(info, "Optimised build. O2DEBUG / LOG(DEBUG) / LOGF(DEBUG) / assert statement will not be shown.");
#endif
        break;
      case DriverState::IMPORT_CURRENT_WORKFLOW:
        // This state is needed to fill the metadata structure
        // which contains how to run the current workflow
        dataProcessorInfos = previousDataProcessorInfos;
        for (auto const& device : deviceSpecs) {
          auto exists = std::find_if(dataProcessorInfos.begin(),
                                     dataProcessorInfos.end(),
                                     [id = device.id](DataProcessorInfo const& info) -> bool { return info.name == id; });
          if (exists != dataProcessorInfos.end()) {
            continue;
          }
          std::vector<std::string> channels;
          for (auto channel : device.inputChannels) {
            channels.push_back(channel.name);
          }
          for (auto channel : device.outputChannels) {
            channels.push_back(channel.name);
          }
          dataProcessorInfos.push_back(
            DataProcessorInfo{
              device.id,
              workflowInfo.executable,
              workflowInfo.args,
              workflowInfo.options,
              channels});
        }
        break;
      case DriverState::MATERIALISE_WORKFLOW:
        try {
          auto workflowState = WorkflowHelpers::verifyWorkflow(workflow);
          if (driverInfo.batch == true && workflowState == WorkflowParsingState::Empty) {
            throw runtime_error("Empty workflow provided while running in batch mode.");
          }
          DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow,
                                                            driverInfo.channelPolicies,
                                                            driverInfo.completionPolicies,
                                                            driverInfo.dispatchPolicies,
                                                            deviceSpecs,
                                                            *resourceManager,
                                                            driverInfo.uniqueWorkflowId,
                                                            !varmap["no-IPC"].as<bool>(),
                                                            driverInfo.resourcesMonitoringInterval,
                                                            varmap["channel-prefix"].as<std::string>());
          metricProcessingCallbacks.clear();
          for (auto& device : deviceSpecs) {
            for (auto& service : device.services) {
              if (service.metricHandling) {
                metricProcessingCallbacks.push_back(service.metricHandling);
              }
            }
          }
          preScheduleCallbacks.clear();
          for (auto& device : deviceSpecs) {
            for (auto& service : device.services) {
              if (service.preSchedule) {
                preScheduleCallbacks.push_back(service.preSchedule);
              }
            }
          }
          postScheduleCallbacks.clear();
          for (auto& device : deviceSpecs) {
            for (auto& service : device.services) {
              if (service.postSchedule) {
                postScheduleCallbacks.push_back(service.postSchedule);
              }
            }
          }

          // This should expand nodes so that we can build a consistent DAG.
        } catch (std::runtime_error& e) {
          std::cerr << "Invalid workflow: " << e.what() << std::endl;
          return 1;
        } catch (o2::framework::RuntimeErrorRef ref) {
          auto& err = o2::framework::error_from_ref(ref);
#ifdef DPL_ENABLE_BACKTRACE
          backtrace_symbols_fd(err.backtrace, err.maxBacktrace, STDERR_FILENO);
#endif
          std::cerr << "Invalid workflow: " << err.what << std::endl;
          return 1;
        } catch (...) {
          std::cerr << "Unknown error while materialising workflow";
          return 1;
        }
        break;
      case DriverState::DO_CHILD:
        // We do not start the process if by default we are stopped.
        if (driverControl.defaultStopped) {
          kill(getpid(), SIGSTOP);
        }
        for (auto& spec : deviceSpecs) {
          if (spec.id == frameworkId) {
            return doChild(driverInfo.argc, driverInfo.argv,
                           serviceRegistry, spec,
                           driverInfo.errorPolicy, loop);
          }
        }
        {
          std::ostringstream ss;
          for (auto& processor : workflow) {
            ss << " - " << processor.name << "\n";
          }
          for (auto& spec : deviceSpecs) {
            ss << " - " << spec.name << "(" << spec.id << ")"
               << "\n";
          }
          LOG(ERROR) << "Unable to find component with id "
                     << frameworkId << ". Available options:\n"
                     << ss.str();
          driverInfo.states.push_back(DriverState::QUIT_REQUESTED);
        }
        break;
      case DriverState::REDEPLOY_GUI:
        // The callback for the GUI needs to be recalculated every time
        // the deployed configuration changes, e.g. a new device
        // has been added to the topology.
        // We need to recreate the GUI callback every time we reschedule
        // because getGUIDebugger actually recreates the GUI state.
        if (window) {
          uv_timer_stop(&gui_timer);
          guiContext.callback = debugGUI->getGUIDebugger(infos, deviceSpecs, dataProcessorInfos, metricsInfos, driverInfo, controls, driverControl);
          guiContext.window = window;
          gui_timer.data = &guiContext;
          uv_timer_start(&gui_timer, gui_callback, 0, 20);
          guiDeployedOnce = true;
        }
        break;
      case DriverState::MERGE_CONFIGS: {
        try {
          controls.resize(deviceSpecs.size());
          deviceExecutions.resize(deviceSpecs.size());

          DeviceSpecHelpers::reworkShmSegmentSize(dataProcessorInfos);
          DeviceSpecHelpers::prepareArguments(driverControl.defaultQuiet,
                                              driverControl.defaultStopped,
                                              dataProcessorInfos,
                                              deviceSpecs,
                                              deviceExecutions, controls,
                                              driverInfo.uniqueWorkflowId);
        } catch (o2::framework::RuntimeErrorRef& ref) {
          auto& err = o2::framework::error_from_ref(ref);
          LOG(ERROR) << "Unable to merge configurations: " << err.what;
#ifdef DPL_ENABLE_BACKTRACE
          std::cerr << "\nStacktrace follows:\n\n";
          backtrace_symbols_fd(err.backtrace, err.maxBacktrace, STDERR_FILENO);
#endif
          return 1;
        }

      } break;
      case DriverState::SCHEDULE: {
        // FIXME: for the moment modifying the topology means we rebuild completely
        //        all the devices and we restart them. This is also what DDS does at
        //        a larger scale. In principle one could try to do a delta and only
        //        restart the data processors which need to be restarted.
        LOG(INFO) << "Redeployment of configuration asked.";
        std::ostringstream forwardedStdin;
        WorkflowSerializationHelpers::dump(forwardedStdin, workflow, dataProcessorInfos);
        infos.reserve(deviceSpecs.size());

        // This is guaranteed to be a single CPU.
        unsigned parentCPU = -1;
        unsigned parentNode = -1;
#if defined(__linux__) && __has_include(<sched.h>)
        parentCPU = sched_getcpu();
#elif __has_include(<linux/getcpu.h>)
        getcpu(&parentCPU, &parentNode, nullptr);
#elif __has_include(<cpuid.h>)
        // FIXME: this is a last resort as it is apparently buggy
        //        on some Intel CPUs.
        GETCPU(parentCPU);
#endif
        for (auto& callback : preScheduleCallbacks) {
          callback(serviceRegistry, varmap);
        }
        for (size_t di = 0; di < deviceSpecs.size(); ++di) {
          if (deviceSpecs[di].resource.hostname != driverInfo.deployHostname) {
            spawnRemoteDevice(forwardedStdin.str(),
                              deviceSpecs[di], controls[di], deviceExecutions[di], infos);
          } else {
            spawnDevice(forwardedStdin.str(),
                        deviceSpecs[di], driverInfo,
                        controls[di], deviceExecutions[di], infos,
                        serviceRegistry, varmap, loop, pollHandles, parentCPU, parentNode);
          }
        }
        for (auto& callback : postScheduleCallbacks) {
          callback(serviceRegistry, varmap);
        }
        assert(infos.empty() == false);
        LOG(INFO) << "Redeployment of configuration done.";
      } break;
      case DriverState::RUNNING:
        // Run any pending libUV event loop, block if
        // any, so that we do not consume CPU time when the driver is
        // idle.
        uv_run(loop, UV_RUN_ONCE);
        // Calculate what we should do next and eventually
        // show the GUI
        if (guiQuitRequested ||
            (driverInfo.terminationPolicy == TerminationPolicy::QUIT && (checkIfCanExit(infos) == true))) {
          // Something requested to quit. This can be a user
          // interaction with the GUI or (if --completion-policy=quit)
          // it could mean that the workflow does not have anything else to do.
          // Let's update the GUI one more time and then EXIT.
          LOG(INFO) << "Quitting";
          driverInfo.states.push_back(DriverState::QUIT_REQUESTED);
        } else if (infos.size() != deviceSpecs.size()) {
          // If the number of deviceSpecs is different from
          // the DeviceInfos it means the speicification
          // does not match what is running, so we need to do
          // further scheduling.
          driverInfo.states.push_back(DriverState::RUNNING);
          driverInfo.states.push_back(DriverState::REDEPLOY_GUI);
          driverInfo.states.push_back(DriverState::SCHEDULE);
          driverInfo.states.push_back(DriverState::MERGE_CONFIGS);
        } else if (deviceSpecs.empty() && driverInfo.batch == true) {
          LOG(INFO) << "No device resulting from the workflow. Quitting.";
          // If there are no deviceSpecs, we exit.
          driverInfo.states.push_back(DriverState::EXIT);
        } else if (deviceSpecs.empty() && driverInfo.batch == false && !guiDeployedOnce) {
          // In case of an empty workflow, we need to deploy the GUI at least once.
          driverInfo.states.push_back(DriverState::RUNNING);
          driverInfo.states.push_back(DriverState::REDEPLOY_GUI);
        } else {
          driverInfo.states.push_back(DriverState::RUNNING);
        }
        {
          uint64_t inputProcessingStart = uv_hrtime();
          auto inputProcessingLatency = inputProcessingStart - inputProcessingLast;
          auto outputProcessing = processChildrenOutput(driverInfo, infos, deviceSpecs, controls, metricsInfos);
          if (outputProcessing.didProcessMetric) {
            size_t timestamp = current_time_with_ms();
            for (auto& callback : metricProcessingCallbacks) {
              callback(serviceRegistry, metricsInfos, deviceSpecs, infos, driverInfo.metrics, timestamp);
            }
            for (auto& metricsInfo : metricsInfos) {
              std::fill(metricsInfo.changed.begin(), metricsInfo.changed.end(), false);
            }
          }
          auto inputProcessingEnd = uv_hrtime();
          driverInfo.inputProcessingCost = (inputProcessingEnd - inputProcessingStart) / 1000000;
          driverInfo.inputProcessingLatency = (inputProcessingLatency) / 1000000;
          inputProcessingLast = inputProcessingStart;
        }
        break;
      case DriverState::QUIT_REQUESTED:
        LOG(INFO) << "QUIT_REQUESTED";
        guiQuitRequested = true;
        // We send SIGCONT to make sure stopped children are resumed
        killChildren(infos, SIGCONT);
        // We send SIGTERM to make sure we do the STOP transition in FairMQ
        killChildren(infos, SIGTERM);
        // We have a timer to send SIGUSR1 to make sure we advance all devices
        // in a timely manner.
        force_step_timer.data = &infos;
        uv_timer_start(&force_step_timer, single_step_callback, 0, 300);
        driverInfo.states.push_back(DriverState::HANDLE_CHILDREN);
        break;
      case DriverState::HANDLE_CHILDREN: {
        // Run any pending libUV event loop, block if
        // any, so that we do not consume CPU time when the driver is
        // idle.
        uv_run(loop, UV_RUN_ONCE);
        // I allow queueing of more sigchld only when
        // I process the previous call
        if (forceful_exit == true) {
          static bool forcefulExitMessage = true;
          if (forcefulExitMessage) {
            LOG(INFO) << "Forceful exit requested.";
            forcefulExitMessage = false;
          }
          killChildren(infos, SIGCONT);
          killChildren(infos, SIGKILL);
        }
        sigchld_requested = false;
        driverInfo.sigchldRequested = false;
        auto outputProcessing = processChildrenOutput(driverInfo, infos, deviceSpecs, controls, metricsInfos);
        if (outputProcessing.didProcessMetric) {
          size_t timestamp = current_time_with_ms();
          for (auto& callback : metricProcessingCallbacks) {
            callback(serviceRegistry, metricsInfos, deviceSpecs, infos, driverInfo.metrics, timestamp);
          }
        }
        hasError = processSigChild(infos);
        if (areAllChildrenGone(infos) == true &&
            (guiQuitRequested || (checkIfCanExit(infos) == true) || graceful_exit)) {
          // We move to the exit, regardless of where we were
          driverInfo.states.resize(0);
          driverInfo.states.push_back(DriverState::EXIT);
        } else if (areAllChildrenGone(infos) == false &&
                   (guiQuitRequested || checkIfCanExit(infos) == true || graceful_exit)) {
          driverInfo.states.push_back(DriverState::HANDLE_CHILDREN);
        } else if (hasError && driverInfo.errorPolicy == TerminationPolicy::QUIT &&
                   !(guiQuitRequested || checkIfCanExit(infos) == true || graceful_exit)) {
          graceful_exit = 1;
          driverInfo.states.push_back(DriverState::QUIT_REQUESTED);
        } else {
        }
      } break;
      case DriverState::EXIT: {
        if (ResourcesMonitoringHelper::isResourcesMonitoringEnabled(driverInfo.resourcesMonitoringInterval)) {
          LOG(INFO) << "Dumping performance metrics to performanceMetrics.json file";
          auto performanceMetrics = o2::monitoring::ProcessMonitor::getAvailableMetricsNames();
          performanceMetrics.push_back("arrow-bytes-delta");
          performanceMetrics.push_back("aod-bytes-read-uncompressed");
          performanceMetrics.push_back("aod-bytes-read-compressed");
          performanceMetrics.push_back("aod-file-read-info");
          ResourcesMonitoringHelper::dumpMetricsToJSON(metricsInfos, driverInfo.metrics, deviceSpecs, performanceMetrics);
        }
        // This is a clean exit. Before we do so, if required,
        // we dump the configuration of all the devices so that
        // we can reuse it. Notice we do not dump anything if
        // the workflow was not really run.
        // NOTE: is this really what we want? should we run
        // SCHEDULE and dump the full configuration as well?
        if (infos.empty()) {
          return 0;
        }
        boost::property_tree::ptree finalConfig;
        assert(infos.size() == deviceSpecs.size());
        for (size_t di = 0; di < infos.size(); ++di) {
          auto info = infos[di];
          auto spec = deviceSpecs[di];
          finalConfig.put_child(spec.name, info.currentConfig);
        }
        LOG(INFO) << "Dumping used configuration in dpl-config.json";
        boost::property_tree::write_json("dpl-config.json", finalConfig);
        if (driverInfo.noSHMCleanup) {
          LOGP(warning, "Not cleaning up shared memory.");
        } else {
          cleanupSHM(driverInfo.uniqueWorkflowId);
        }
        return calculateExitCode(deviceSpecs, infos);
      }
      case DriverState::PERFORM_CALLBACKS:
        for (auto& callback : driverControl.callbacks) {
          callback(workflow, deviceSpecs, deviceExecutions, dataProcessorInfos);
        }
        driverControl.callbacks.clear();
        break;
      default:
        LOG(ERROR) << "Driver transitioned in an unknown state("
                   << "current: " << (int)current
                   << ", previous: " << (int)previous
                   << "). Shutting down.";
        driverInfo.states.push_back(DriverState::QUIT_REQUESTED);
    }
  }
}

// Print help
void printHelp(bpo::variables_map const& varmap,
               bpo::options_description const& executorOptions,
               std::vector<DataProcessorSpec> const& physicalWorkflow,
               std::vector<ConfigParamSpec> const& currentWorkflowOptions)
{
  auto mode = varmap["help"].as<std::string>();
  bpo::options_description helpOptions;
  if (mode == "full" || mode == "short" || mode == "executor") {
    helpOptions.add(executorOptions);
  }
  // this time no veto is applied, so all the options are added for printout
  if (mode == "executor") {
    // nothing more
  } else if (mode == "workflow") {
    // executor options and workflow options, skip the actual workflow
    o2::framework::WorkflowSpec emptyWorkflow;
    helpOptions.add(ConfigParamsHelper::prepareOptionDescriptions(emptyWorkflow, currentWorkflowOptions));
  } else if (mode == "full" || mode == "short") {
    helpOptions.add(ConfigParamsHelper::prepareOptionDescriptions(physicalWorkflow, currentWorkflowOptions,
                                                                  bpo::options_description(),
                                                                  mode));
  } else {
    helpOptions.add(ConfigParamsHelper::prepareOptionDescriptions(physicalWorkflow, {},
                                                                  bpo::options_description(),
                                                                  mode));
  }
  if (helpOptions.options().size() == 0) {
    // the specified argument is invalid, add at leat the executor options
    mode += " is an invalid argument, please use correct argument for";
    helpOptions.add(executorOptions);
  }
  std::cout << "ALICE O2 DPL workflow driver"        //
            << " (" << mode << " help)" << std::endl //
            << helpOptions << std::endl;             //
}

// Helper to find out if stdout is actually attached to a pipe.
bool isOutputToPipe()
{
  struct stat s;
  fstat(STDOUT_FILENO, &s);
  return ((s.st_mode & S_IFIFO) != 0);
}

void overrideSuffix(ConfigContext& ctx, WorkflowSpec& workflow)
{
  auto suffix = ctx.options().get<std::string>("workflow-suffix");
  if (suffix.empty()) {
    return;
  }
  for (auto& processor : workflow) {
    processor.name = processor.name + suffix;
  }
}

void overrideCloning(ConfigContext& ctx, WorkflowSpec& workflow)
{
  struct CloningSpec {
    std::string templateMatcher;
    std::string cloneName;
  };
  auto s = ctx.options().get<std::string>("clone");
  std::vector<CloningSpec> specs;
  std::string delimiter = ",";

  size_t pos = 0;
  while (s.empty() == false) {
    auto newPos = s.find(delimiter);
    auto token = s.substr(0, newPos);
    auto split = token.find(":");
    if (split == std::string::npos) {
      throw std::runtime_error("bad clone definition. Syntax <template-processor>:<clone-name>");
    }
    auto key = token.substr(0, split);
    token.erase(0, split + 1);
    size_t error;
    std::string value = "";
    try {
      auto numValue = std::stoll(token, &error, 10);
      if (token[error] != '\0') {
        throw std::runtime_error("bad name for clone:" + token);
      }
      value = key + "_c" + std::to_string(numValue);
    } catch (std::invalid_argument& e) {
      value = token;
    }
    specs.push_back({key, value});
    s.erase(0, newPos + (newPos == std::string::npos ? 0 : 1));
  }
  if (s.empty() == false && specs.empty() == true) {
    throw std::runtime_error("bad pipeline definition. Syntax <processor>:<pipeline>");
  }

  std::vector<DataProcessorSpec> extraSpecs;
  for (auto& spec : specs) {
    for (auto& processor : workflow) {
      if (processor.name == spec.templateMatcher) {
        auto clone = processor;
        clone.name = spec.cloneName;
        extraSpecs.push_back(clone);
      }
    }
  }
  workflow.insert(workflow.end(), extraSpecs.begin(), extraSpecs.end());
}

void overridePipeline(ConfigContext& ctx, WorkflowSpec& workflow)
{
  struct PipelineSpec {
    std::string matcher;
    int64_t pipeline;
  };
  auto s = ctx.options().get<std::string>("pipeline");
  std::vector<PipelineSpec> specs;
  std::string delimiter = ",";

  size_t pos = 0;
  while (s.empty() == false) {
    auto newPos = s.find(delimiter);
    auto token = s.substr(0, newPos);
    auto split = token.find(":");
    if (split == std::string::npos) {
      throw std::runtime_error("bad pipeline definition. Syntax <processor>:<pipeline>");
    }
    auto key = token.substr(0, split);
    token.erase(0, split + 1);
    size_t error;
    auto value = std::stoll(token, &error, 10);
    if (token[error] != '\0') {
      throw std::runtime_error("Bad pipeline definition. Expecting integer");
    }
    specs.push_back({key, value});
    s.erase(0, newPos + (newPos == std::string::npos ? 0 : 1));
  }
  if (s.empty() == false && specs.empty() == true) {
    throw std::runtime_error("bad pipeline definition. Syntax <processor>:<pipeline>");
  }

  for (auto& spec : specs) {
    for (auto& processor : workflow) {
      if (processor.name == spec.matcher) {
        processor.maxInputTimeslices = spec.pipeline;
      }
    }
  }
}

/// Helper function to initialise the controller from the command line options.
void initialiseDriverControl(bpo::variables_map const& varmap,
                             DriverControl& control)
{
  // Control is initialised outside the main loop because
  // command line options are really affecting control.
  control.defaultQuiet = varmap["quiet"].as<bool>();
  control.defaultStopped = varmap["stop"].as<bool>();

  if (varmap["single-step"].as<bool>()) {
    control.state = DriverControlState::STEP;
  } else {
    control.state = DriverControlState::PLAY;
  }

  if (varmap["graphviz"].as<bool>()) {
    // Dump a graphviz representation of what I will do.
    control.callbacks = {[](WorkflowSpec const& workflow,
                            DeviceSpecs const& specs,
                            DeviceExecutions const&,
                            DataProcessorInfos&) {
      GraphvizHelpers::dumpDeviceSpec2Graphviz(std::cout, specs);
    }};
    control.forcedTransitions = {
      DriverState::EXIT,                    //
      DriverState::PERFORM_CALLBACKS,       //
      DriverState::MERGE_CONFIGS,           //
      DriverState::IMPORT_CURRENT_WORKFLOW, //
      DriverState::MATERIALISE_WORKFLOW     //
    };
  } else if (varmap["dds"].as<bool>()) {
    // Dump a DDS representation of what I will do.
    // Notice that compared to DDS we need to schedule things,
    // because DDS needs to be able to have actual Executions in
    // order to provide a correct configuration.
    control.callbacks = {[](WorkflowSpec const& workflow,
                            DeviceSpecs const& specs,
                            DeviceExecutions const& executions,
                            DataProcessorInfos&) {
      dumpDeviceSpec2DDS(std::cout, specs, executions);
    }};
    control.forcedTransitions = {
      DriverState::EXIT,                    //
      DriverState::PERFORM_CALLBACKS,       //
      DriverState::SCHEDULE,                //
      DriverState::MERGE_CONFIGS,           //
      DriverState::IMPORT_CURRENT_WORKFLOW, //
      DriverState::MATERIALISE_WORKFLOW     //
    };
  } else if (varmap["o2-control"].as<bool>()) {
    control.callbacks = {[](WorkflowSpec const& workflow,
                            DeviceSpecs const& specs,
                            DeviceExecutions const& executions,
                            DataProcessorInfos&) {
      dumpDeviceSpec2O2Control(std::cout, specs, executions);
    }};
    control.forcedTransitions = {
      DriverState::EXIT,                    //
      DriverState::PERFORM_CALLBACKS,       //
      DriverState::SCHEDULE,                //
      DriverState::MERGE_CONFIGS,           //
      DriverState::IMPORT_CURRENT_WORKFLOW, //
      DriverState::MATERIALISE_WORKFLOW     //
    };
  } else if (varmap.count("id")) {
    // FIXME: for the time being each child needs to recalculate the workflow,
    //        so that it can understand what it needs to do. This is obviously
    //        a bad idea. In the future we should have the client be pushed
    //        it's own configuration by the driver.
    control.forcedTransitions = {
      DriverState::DO_CHILD,                //
      DriverState::MERGE_CONFIGS,           //
      DriverState::IMPORT_CURRENT_WORKFLOW, //
      DriverState::MATERIALISE_WORKFLOW     //
    };
  } else if ((varmap["dump-workflow"].as<bool>() == true) || (varmap["run"].as<bool>() == false && varmap.count("id") == 0 && isOutputToPipe())) {
    control.callbacks = {[filename = varmap["dump-workflow-file"].as<std::string>()](WorkflowSpec const& workflow,
                                                                                     DeviceSpecs const devices,
                                                                                     DeviceExecutions const&,
                                                                                     DataProcessorInfos& dataProcessorInfos) {
      if (filename == "-") {
        WorkflowSerializationHelpers::dump(std::cout, workflow, dataProcessorInfos);
        // FIXME: this is to avoid trailing garbage..
        exit(0);
      } else {
        std::ofstream output(filename);
        WorkflowSerializationHelpers::dump(output, workflow, dataProcessorInfos);
      }
    }};
    control.forcedTransitions = {
      DriverState::EXIT,                    //
      DriverState::PERFORM_CALLBACKS,       //
      DriverState::MERGE_CONFIGS,           //
      DriverState::IMPORT_CURRENT_WORKFLOW, //
      DriverState::MATERIALISE_WORKFLOW     //
    };
  } else {
    // By default we simply start the main loop of the driver.
    control.forcedTransitions = {
      DriverState::INIT,                    //
      DriverState::IMPORT_CURRENT_WORKFLOW, //
      DriverState::MATERIALISE_WORKFLOW     //
    };
  }
}

/// Helper to to detect conflicting options
void conflicting_options(const boost::program_options::variables_map& vm,
                         const std::string& opt1, const std::string& opt2)
{
  if (vm.count(opt1) && !vm[opt1].defaulted() &&
      vm.count(opt2) && !vm[opt2].defaulted()) {
    throw std::logic_error(std::string("Conflicting options '") +
                           opt1 + "' and '" + opt2 + "'.");
  }
}

template <typename T>
void apply_permutation(
  std::vector<T>& v,
  std::vector<int>& indices)
{
  using std::swap; // to permit Koenig lookup
  for (size_t i = 0; i < indices.size(); i++) {
    auto current = i;
    while (i != indices[current]) {
      auto next = indices[current];
      swap(v[current], v[next]);
      indices[current] = current;
      current = next;
    }
    indices[current] = current;
  }
}

std::string debugTopoInfo(std::vector<DataProcessorSpec> const& specs,
                          std::vector<TopoIndexInfo> const& infos,
                          std::vector<std::pair<int, int>> const& edges)
{
  std::ostringstream out;

  out << "\nTopological info:\n";
  for (auto& ti : infos) {
    out << specs[ti.index].name << " (index: " << ti.index << ", layer: " << ti.layer << ")\n";
    out << " Inputs:\n";
    for (auto& ii : specs[ti.index].inputs) {
      out << "   - " << DataSpecUtils::describe(ii) << "\n";
    }
    out << "\n Outputs:\n";
    for (auto& ii : specs[ti.index].outputs) {
      out << "   - " << DataSpecUtils::describe(ii) << "\n";
    }
  }
  out << "\nEdges values:\n";
  for (auto& e : edges) {
    out << specs[e.second].name << " depends on " << specs[e.first].name << "\n";
  }
  for (auto& d : specs) {
    out << "- " << d.name << std::endl;
  }
  return out.str();
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
int doMain(int argc, char** argv, o2::framework::WorkflowSpec const& workflow,
           std::vector<ChannelConfigurationPolicy> const& channelPolicies,
           std::vector<CompletionPolicy> const& completionPolicies,
           std::vector<DispatchPolicy> const& dispatchPolicies,
           std::vector<ConfigParamSpec> const& currentWorkflowOptions,
           o2::framework::ConfigContext& configContext)
{
  O2_SIGNPOST_INIT();
  std::vector<std::string> currentArgs;
  for (size_t ai = 1; ai < argc; ++ai) {
    currentArgs.push_back(argv[ai]);
  }

  WorkflowInfo currentWorkflow{
    argv[0],
    currentArgs,
    currentWorkflowOptions};

  enum TerminationPolicy policy;
  enum TerminationPolicy errorPolicy;
  bpo::options_description executorOptions("Executor options");
  const char* helpDescription = "print help: short, full, executor, or processor name";
  executorOptions.add_options()                                                                                                                //
    ("help,h", bpo::value<std::string>()->implicit_value("short"), helpDescription)                                                            //                                                                                                       //
    ("quiet,q", bpo::value<bool>()->zero_tokens()->default_value(false), "quiet operation")                                                    //                                                                                                         //
    ("stop,s", bpo::value<bool>()->zero_tokens()->default_value(false), "stop before device start")                                            //                                                                                                           //
    ("single-step", bpo::value<bool>()->zero_tokens()->default_value(false), "start in single step mode")                                      //                                                                                                             //
    ("batch,b", bpo::value<bool>()->zero_tokens()->default_value(isatty(fileno(stdout)) == 0), "batch processing mode")                        //                                                                                                               //
    ("no-cleanup", bpo::value<bool>()->zero_tokens()->default_value(false), "do not cleanup the shm segment")                                  //                                                                                                               //
    ("hostname", bpo::value<std::string>()->default_value("localhost"), "hostname to deploy")                                                  //                                                                                                                 //
    ("resources", bpo::value<std::string>()->default_value(""), "resources allocated for the workflow")                                        //                                                                                                                   //
    ("start-port,p", bpo::value<unsigned short>()->default_value(22000), "start port to allocate")                                             //                                                                                                                     //
    ("port-range,pr", bpo::value<unsigned short>()->default_value(1000), "ports in range")                                                     //                                                                                                                       //
    ("completion-policy,c", bpo::value<TerminationPolicy>(&policy)->default_value(TerminationPolicy::QUIT),                                    //                                                                                                                       //
     "what to do when processing is finished: quit, wait")                                                                                     //                                                                                                                      //
    ("error-policy", bpo::value<TerminationPolicy>(&errorPolicy)->default_value(TerminationPolicy::QUIT),                                      //                                                                                                                          //
     "what to do when a device has an error: quit, wait")                                                                                      //                                                                                                                            //
    ("graphviz,g", bpo::value<bool>()->zero_tokens()->default_value(false), "produce graph output")                                            //                                                                                                                              //
    ("timeout,t", bpo::value<uint64_t>()->default_value(0), "forced exit timeout (in seconds)")                                                //                                                                                                                                //
    ("dds,D", bpo::value<bool>()->zero_tokens()->default_value(false), "create DDS configuration")                                             //                                                                                                                                  //
    ("dump-workflow,dump", bpo::value<bool>()->zero_tokens()->default_value(false), "dump workflow as JSON")                                   //                                                                                                                                    //
    ("dump-workflow-file", bpo::value<std::string>()->default_value("-"), "file to which do the dump")                                         //                                                                                                                                      //
    ("run", bpo::value<bool>()->zero_tokens()->default_value(false), "run workflow merged so far")                                             //                                                                                                                                        //
    ("no-IPC", bpo::value<bool>()->zero_tokens()->default_value(false), "disable IPC topology optimization")                                   //                                                                                                                                        //
    ("o2-control,o2", bpo::value<bool>()->zero_tokens()->default_value(false), "create O2 Control configuration")                              //
    ("resources-monitoring", bpo::value<unsigned short>()->default_value(0), "enable cpu/memory monitoring for provided interval in seconds"); //
  // some of the options must be forwarded by default to the device
  executorOptions.add(DeviceSpecHelpers::getForwardedDeviceOptions());

  gHiddenDeviceOptions.add_options()                                                    //
    ("id,i", bpo::value<std::string>(), "device id for child spawning")                 //
    ("channel-config", bpo::value<std::vector<std::string>>(), "channel configuration") //
    ("control", "control plugin")                                                       //
    ("log-color", "logging color scheme")("color", "logging color scheme");

  bpo::options_description visibleOptions;
  visibleOptions.add(executorOptions);

  auto physicalWorkflow = workflow;
  std::map<std::string, size_t> rankIndex;
  // We remove the duplicates because for the moment child get themself twice:
  // once from the actual definition in the child, a second time from the
  // configuration they get passed by their parents.
  // Notice that we do not know in which order we will get the workflows, so
  // while we keep the order of DataProcessors we reshuffle them based on
  // some hopefully unique hash.
  size_t workflowHashA = 0;
  std::hash<std::string> hash_fn;

  for (auto& dp : workflow) {
    workflowHashA += hash_fn(dp.name);
  }

  for (auto& dp : workflow) {
    rankIndex.insert(std::make_pair(dp.name, workflowHashA));
  }

  std::vector<DataProcessorInfo> dataProcessorInfos;
  if (isatty(STDIN_FILENO) == false) {
    std::vector<DataProcessorSpec> importedWorkflow;
    WorkflowSerializationHelpers::import(std::cin, importedWorkflow, dataProcessorInfos);

    size_t workflowHashB = 0;
    for (auto& dp : importedWorkflow) {
      workflowHashB += hash_fn(dp.name);
    }

    // FIXME: Streamline...
    // We remove the duplicates because for the moment child get themself twice:
    // once from the actual definition in the child, a second time from the
    // configuration they get passed by their parents.
    for (auto& dp : importedWorkflow) {
      auto found = std::find_if(physicalWorkflow.begin(), physicalWorkflow.end(),
                                [& name = dp.name](DataProcessorSpec const& spec) { return spec.name == name; });
      if (found == physicalWorkflow.end()) {
        physicalWorkflow.push_back(dp);
        rankIndex.insert(std::make_pair(dp.name, workflowHashB));
      }
    }
  }

  // We insert the hash for the internal devices.
  WorkflowHelpers::injectServiceDevices(physicalWorkflow, configContext);
  for (auto& dp : physicalWorkflow) {
    if (dp.name.rfind("internal-") == 0) {
      rankIndex.insert(std::make_pair(dp.name, hash_fn("internal")));
    }
  }

  // We sort dataprocessors and Inputs / outputs by name, so that the edges are
  // always in the same order.
  std::stable_sort(physicalWorkflow.begin(), physicalWorkflow.end(), [](DataProcessorSpec const& a, DataProcessorSpec const& b) {
    return a.name < b.name;
  });

  for (auto& dp : physicalWorkflow) {
    std::stable_sort(dp.inputs.begin(), dp.inputs.end(),
                     [](InputSpec const& a, InputSpec const& b) { return DataSpecUtils::describe(a) < DataSpecUtils::describe(b); });
    std::stable_sort(dp.outputs.begin(), dp.outputs.end(),
                     [](OutputSpec const& a, OutputSpec const& b) { return DataSpecUtils::describe(a) < DataSpecUtils::describe(b); });
  }

  // check if DataProcessorSpec at i depends on j
  auto checkDependencies = [& workflow = physicalWorkflow](int i, int j) {
    DataProcessorSpec const& a = workflow[i];
    DataProcessorSpec const& b = workflow[j];
    for (size_t ii = 0; ii < a.inputs.size(); ++ii) {
      for (size_t oi = 0; oi < b.outputs.size(); ++oi) {
        try {
          if (DataSpecUtils::match(a.inputs[ii], b.outputs[oi])) {
            return true;
          }
        } catch (...) {
          continue;
        }
      }
    }
    return false;
  };

  // Create a list of all the edges, so that we can do a topological sort
  // before we create the graph.
  std::vector<std::pair<int, int>> edges;

  if (physicalWorkflow.size() > 1) {
    for (size_t i = 0; i < physicalWorkflow.size() - 1; ++i) {
      for (size_t j = i; j < physicalWorkflow.size(); ++j) {
        if (i == j && checkDependencies(i, j)) {
          throw std::runtime_error(physicalWorkflow[i].name + " depends on itself");
        }
        bool both = false;
        if (checkDependencies(i, j)) {
          edges.emplace_back(j, i);
          both = true;
        }
        if (checkDependencies(j, i)) {
          edges.emplace_back(i, j);
          if (both) {
            throw std::runtime_error(physicalWorkflow[i].name + " has circular dependency with " + physicalWorkflow[j].name);
          }
        }
      }
    }

    auto topoInfos = WorkflowHelpers::topologicalSort(physicalWorkflow.size(), &edges[0].first, &edges[0].second, sizeof(std::pair<int, int>), edges.size());
    if (topoInfos.size() != physicalWorkflow.size()) {
      throw std::runtime_error("Unable to do topological sort of the resulting workflow. Do you have loops?\n" + debugTopoInfo(physicalWorkflow, topoInfos, edges));
    }
    // Sort by layer and then by name, to ensure stability.
    std::stable_sort(topoInfos.begin(), topoInfos.end(), [& workflow = physicalWorkflow, &rankIndex, &topoInfos](TopoIndexInfo const& a, TopoIndexInfo const& b) {
      auto aRank = std::make_tuple(a.layer, -workflow.at(a.index).outputs.size(), workflow.at(a.index).name);
      auto bRank = std::make_tuple(b.layer, -workflow.at(b.index).outputs.size(), workflow.at(b.index).name);
      return aRank < bRank;
    });
    // Reverse index and apply the result
    std::vector<int> dataProcessorOrder;
    dataProcessorOrder.resize(topoInfos.size());
    for (size_t i = 0; i < topoInfos.size(); ++i) {
      dataProcessorOrder[topoInfos[i].index] = i;
    }
    std::vector<int> newLocations;
    newLocations.resize(dataProcessorOrder.size());
    for (size_t i = 0; i < dataProcessorOrder.size(); ++i) {
      newLocations[dataProcessorOrder[i]] = i;
    }
    apply_permutation(physicalWorkflow, newLocations);
  }

  // Use the hidden options as veto, all config specs matching a definition
  // in the hidden options are skipped in order to avoid duplicate definitions
  // in the main parser. Note: all config specs are forwarded to devices
  visibleOptions.add(ConfigParamsHelper::prepareOptionDescriptions(physicalWorkflow, currentWorkflowOptions, gHiddenDeviceOptions));

  bpo::options_description od;
  od.add(visibleOptions);
  od.add(gHiddenDeviceOptions);

  // FIXME: decide about the policy for handling unrecognized arguments
  // command_line_parser with option allow_unregistered() can be used
  using namespace bpo::command_line_style;
  auto style = (allow_short | short_allow_adjacent | short_allow_next | allow_long | long_allow_adjacent | long_allow_next | allow_sticky | allow_dash_for_short);
  bpo::variables_map varmap;
  try {
    bpo::store(
      bpo::command_line_parser(argc, argv)
        .options(od)
        .style(style)
        .run(),
      varmap);
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    exit(1);
  }
  conflicting_options(varmap, "dds", "o2-control");
  conflicting_options(varmap, "dds", "dump-workflow");
  conflicting_options(varmap, "dds", "run");
  conflicting_options(varmap, "dds", "graphviz");
  conflicting_options(varmap, "o2-control", "dump-workflow");
  conflicting_options(varmap, "o2-control", "run");
  conflicting_options(varmap, "o2-control", "graphviz");
  conflicting_options(varmap, "run", "dump-workflow");
  conflicting_options(varmap, "run", "graphviz");
  conflicting_options(varmap, "dump-workflow", "graphviz");

  if (varmap.count("help")) {
    printHelp(varmap, executorOptions, physicalWorkflow, currentWorkflowOptions);
    exit(0);
  }
  DriverControl driverControl;
  initialiseDriverControl(varmap, driverControl);

  DriverInfo driverInfo;
  driverInfo.states.reserve(10);
  driverInfo.sigintRequested = false;
  driverInfo.sigchldRequested = false;
  driverInfo.channelPolicies = channelPolicies;
  driverInfo.completionPolicies = completionPolicies;
  driverInfo.dispatchPolicies = dispatchPolicies;
  driverInfo.argc = argc;
  driverInfo.argv = argv;
  driverInfo.batch = varmap["batch"].as<bool>();
  driverInfo.noSHMCleanup = varmap["no-cleanup"].as<bool>();
  driverInfo.terminationPolicy = varmap["completion-policy"].as<TerminationPolicy>();
  if (varmap["error-policy"].defaulted() && driverInfo.batch == false) {
    driverInfo.errorPolicy = TerminationPolicy::WAIT;
  } else {
    driverInfo.errorPolicy = varmap["error-policy"].as<TerminationPolicy>();
  }
  driverInfo.startTime = uv_hrtime();
  driverInfo.timeout = varmap["timeout"].as<uint64_t>();
  driverInfo.deployHostname = varmap["hostname"].as<std::string>();
  driverInfo.resources = varmap["resources"].as<std::string>();
  driverInfo.resourcesMonitoringInterval = varmap["resources-monitoring"].as<unsigned short>();

  // FIXME: should use the whole dataProcessorInfos, actually...
  driverInfo.processorInfo = dataProcessorInfos;
  driverInfo.configContext = &configContext;

  std::string frameworkId;
  // If the id is set, this means this is a device,
  // otherwise this is the driver.
  if (varmap.count("id")) {
    frameworkId = varmap["id"].as<std::string>();
    driverInfo.uniqueWorkflowId = fmt::format("{}", getppid());
  } else {
    driverInfo.uniqueWorkflowId = fmt::format("{}", getpid());
  }
  return runStateMachine(physicalWorkflow,
                         currentWorkflow,
                         dataProcessorInfos,
                         driverControl,
                         driverInfo,
                         gDeviceMetricsInfos,
                         varmap,
                         frameworkId);
}

void doBoostException(boost::exception& e)
{
  LOG(ERROR) << "error while setting up workflow: \n"
             << boost::current_exception_diagnostic_information(true);
}
