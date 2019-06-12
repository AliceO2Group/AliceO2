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
#include "Framework/DebugGUI.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceExecution.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceSpec.h"
#include "Framework/FrameworkGUIDebugger.h"
#include "Framework/FreePortFinder.h"
#include "Framework/LocalRootFileService.h"
#include "Framework/LogParsingHelpers.h"
#include "Framework/Logger.h"
#include "Framework/ParallelContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/SimpleRawDeviceService.h"
#include "Framework/Signpost.h"
#include "Framework/TextControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/WorkflowSpec.h"

#include "DataProcessingStatus.h"
#include "DDSConfigHelpers.h"
#include "O2ControlHelpers.h"
#include "DeviceSpecHelpers.h"
#include "DriverControl.h"
#include "DriverInfo.h"
#include "DataProcessorInfo.h"
#include "GraphvizHelpers.h"
#include "SimpleResourceManager.h"
#include "WorkflowSerializationHelpers.h"

#include <Monitoring/MonitoringFactory.h>
#include <InfoLogger/InfoLogger.hxx>

#include "FairMQDevice.h"
#include <fairmq/DeviceRunner.h>
#include "options/FairMQProgOptions.h"

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

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
#include <chrono>
#include <utility>

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

using namespace o2::monitoring;
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
  } else
    in.setstate(std::ios_base::failbit);
  return in;
}

std::ostream& operator<<(std::ostream& out, const enum TerminationPolicy& policy)
{
  if (policy == TerminationPolicy::QUIT) {
    out << "quit";
  } else if (policy == TerminationPolicy::WAIT) {
    out << "wait";
  } else
    out.setstate(std::ios_base::failbit);
  return out;
}
} // namespace framework
} // namespace o2

// Read from a given fd and print it.
// return true if we can still read from it,
// return false if we need to close the input pipe.
//
// FIXME: We should really print full lines.
bool getChildData(int infd, DeviceInfo& outinfo)
{
  char buffer[1024 * 16];
  int bytes_read;
  // NOTE: do not quite understand read ends up blocking if I read more than
  //        once. Oh well... Good enough for now.
  O2_SIGNPOST_START(DriverStatus::ID, DriverStatus::BYTES_READ, outinfo.pid, infd, 0);
  // do {
  bytes_read = read(infd, buffer, 1024 * 16);
  if (bytes_read == 0) {
    return false;
  }
  if (bytes_read == -1) {
    switch (errno) {
      case EAGAIN:
        return true;
      default:
        return false;
    }
  }
  assert(bytes_read > 0);
  outinfo.unprinted += std::string(buffer, bytes_read);
  // } while (bytes_read != 0);
  O2_SIGNPOST_END(DriverStatus::ID, DriverStatus::BYTES_READ, bytes_read, 0, 0);
  return true;
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
    if (info.active) {
      return false;
    }
  }
  return true;
}

/// Calculate exit code
int calculateExitCode(std::vector<DeviceInfo>& infos)
{
  int exitCode = 0;
  for (auto& info : infos) {
    if (exitCode == 0 && info.maxLogLevel >= LogParsingHelpers::LogLevel::Error) {
      LOG(ERROR) << "Child " << info.pid << " had at least one "
                 << "message above severity ERROR: " << info.lastError;
      exitCode = 1;
    }
  }
  return exitCode;
}

int createPipes(int maxFd, int* pipes)
{
  auto p = pipe(pipes);
  maxFd = maxFd > pipes[0] ? maxFd : pipes[0];
  maxFd = maxFd > pipes[1] ? maxFd : pipes[1];

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
  return maxFd;
}

// We don't do anything in the signal handler but
// we simply note down the fact a signal arrived.
// All the processing is done by the state machine.
volatile sig_atomic_t graceful_exit = false;
volatile sig_atomic_t sigchld_requested = false;

static void handle_sigint(int) { graceful_exit = true; }

static void handle_sigchld(int) { sigchld_requested = true; }


/// This will start a new device by forking and executing a
/// new child
void spawnDevice(std::string const& forwardedStdin,
                 DeviceSpec const& spec,
                 std::map<int, size_t>& socket2DeviceInfo,
                 DeviceControl& control,
                 DeviceExecution& execution,
                 std::vector<DeviceInfo>& deviceInfos,
                 int& maxFd, fd_set& childFdset)
{
  int childstdin[2];
  int childstdout[2];
  int childstderr[2];

  maxFd = createPipes(maxFd, childstdin);
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

  LOG(INFO) << "Starting " << spec.id << " on pid " << id << "\n";
  DeviceInfo info;
  info.pid = id;
  info.active = true;
  info.readyToQuit = false;
  info.historySize = 1000;
  info.historyPos = 0;
  info.maxLogLevel = LogParsingHelpers::LogLevel::Debug;
  info.dataRelayerViewIndex = Metric2DViewIndex{ "data_relayer", 0, 0, {} };
  info.variablesViewIndex = Metric2DViewIndex{ "matcher_variables", 0, 0, {} };
  info.queriesViewIndex = Metric2DViewIndex{ "data_queries", 0, 0, {} };

  socket2DeviceInfo.insert(std::make_pair(childstdout[0], deviceInfos.size()));
  socket2DeviceInfo.insert(std::make_pair(childstderr[0], deviceInfos.size()));
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

  FD_SET(childstdout[0], &childFdset);
  FD_SET(childstderr[0], &childFdset);
}

void updateMetricsNames(DriverInfo& state, std::vector<DeviceMetricsInfo> const& metricsInfos)
{
  // Calculate the unique set of metrics, as available in the metrics service
  static std::unordered_set<std::string> allMetricsNames;
  for (const auto& metricsInfo : metricsInfos) {
    for (const auto& labelsPairs : metricsInfo.metricLabelsIdx) {
      allMetricsNames.insert(std::string(labelsPairs.label));
    }
  }
  std::vector<std::string> result(allMetricsNames.begin(), allMetricsNames.end());
  std::sort(result.begin(), result.end());
  state.availableMetrics.swap(result);
}

void processChildrenOutput(DriverInfo& driverInfo, DeviceInfos& infos, DeviceSpecs const& specs,
                           DeviceControls& controls, std::vector<DeviceMetricsInfo>& metricsInfos)
{
  // Wait for children to say something. When they do
  // print it.
  fd_set fdset;
  timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 16666; // This should be enough to allow 60 HZ redrawing.
  memcpy(&fdset, &driverInfo.childFdset, sizeof(fd_set));
  int numFd = select(driverInfo.maxFd, &fdset, nullptr, nullptr, &timeout);
  if (numFd == 0) {
    return;
  }
  for (int si = 0; si < driverInfo.maxFd; ++si) {
    if (FD_ISSET(si, &fdset)) {
      assert(driverInfo.socket2DeviceInfo.find(si) != driverInfo.socket2DeviceInfo.end());
      auto& info = infos[driverInfo.socket2DeviceInfo[si]];

      bool fdActive = getChildData(si, info);
      // If the pipe was closed due to the process exiting, we
      // can avoid the select.
      if (!fdActive) {
        info.active = false;
        close(si);
        FD_CLR(si, &driverInfo.childFdset);
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
  ParsedMetricMatch metricMatch;
  const std::string delimiter("\n");
  bool hasNewMetric = false;
  for (size_t di = 0, de = infos.size(); di < de; ++di) {
    DeviceInfo& info = infos[di];
    DeviceControl& control = controls[di];
    DeviceMetricsInfo& metrics = metricsInfos[di];

    if (info.unprinted.empty()) {
      continue;
    }

    O2_SIGNPOST_START(DriverStatus::ID, DriverStatus::BYTES_PROCESSED, info.pid, 0, 0);

    std::string_view s = info.unprinted;
    size_t pos = 0;
    info.history.resize(info.historySize);
    info.historyLevel.resize(info.historySize);

    auto updateMetricsViews =
      Metric2DViewIndex::getUpdater({ &info.dataRelayerViewIndex,
                                      &info.variablesViewIndex,
                                      &info.queriesViewIndex });

    auto newMetricCallback = [&updateMetricsViews, &driverInfo, &metricsInfos, &hasNewMetric](std::string const& name, MetricInfo const& metric, int value, size_t metricIndex) {
      updateMetricsViews(name, metric, value, metricIndex);
      hasNewMetric = true;
    };

    while ((pos = s.find(delimiter)) != std::string::npos) {
      std::string token{ s.substr(0, pos) };
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
      } else if (logLevel == LogParsingHelpers::LogLevel::Info && parseControl(token, match)) {
        auto command = match[1];
        auto validFor = match[2];
        LOG(DEBUG) << "Found control command " << command << " from pid " << info.pid << " valid for " << validFor;
        if (command == "QUIT") {
          if (validFor == "ALL") {
            for (auto& deviceInfo : infos) {
              deviceInfo.readyToQuit = true;
            }
          }
          // in case of "ME", fetch the pid and modify only matching deviceInfos
          if (validFor == "ME") {
            for (auto& deviceInfo : infos) {
              if (deviceInfo.pid == info.pid) {
                deviceInfo.readyToQuit = true;
              }
            }
          }
        }
      } else if (!control.quiet && (token.find(control.logFilter) != std::string::npos) &&
                 logLevel >= control.logLevel) {
        assert(info.historyPos >= 0);
        assert(info.historyPos < info.history.size());
        info.history[info.historyPos] = token;
        info.historyLevel[info.historyPos] = logLevel;
        info.historyPos = (info.historyPos + 1) % info.history.size();
        std::cout << "[" << info.pid << "]: " << token << std::endl;
      }
      // We keep track of the maximum log error a
      // device has seen.
      if (logLevel > info.maxLogLevel && logLevel > LogParsingHelpers::LogLevel::Info &&
          logLevel != LogParsingHelpers::LogLevel::Unknown) {
        info.maxLogLevel = logLevel;
      }
      if (logLevel == LogParsingHelpers::LogLevel::Error) {
        info.lastError = token;
      }
      s.remove_prefix(pos + delimiter.length());
    }
    size_t oldSize = info.unprinted.size();
    info.unprinted = s;
    O2_SIGNPOST_END(DriverStatus::ID, DriverStatus::BYTES_PROCESSED, oldSize - info.unprinted.size(), 0, 0);
  }
  if (hasNewMetric) {
    hasNewMetric = false;
    updateMetricsNames(driverInfo, metricsInfos);
  }
  // FIXME: for the gui to work correctly I would actually need to
  //        run the loop more often and update whenever enough time has
  //        passed.
}

// Process all the sigchld which are pending
void processSigChild(DeviceInfos& infos)
{
  while (true) {
    pid_t pid = waitpid((pid_t)(-1), nullptr, WNOHANG);
    if (pid > 0) {
      for (auto& info : infos) {
        if (info.pid == pid) {
          info.active = false;
        }
      }
      continue;
    } else {
      break;
    }
  }
}

// Creates the sink for FairLogger / InfoLogger integration
auto createInfoLoggerSinkHelper(std::unique_ptr<InfoLogger>& logger, std::unique_ptr<InfoLoggerContext>& ctx)
{
  return [&logger,
          &ctx](const std::string& content, const fair::LogMetaData& metadata) {
    // translate FMQ metadata
    InfoLogger::InfoLogger::Severity severity = InfoLogger::Severity::Undefined;
    int level = InfoLogger::undefinedMessageOption.level;

    if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::nolog)) {
      // discard
      return;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::fatal)) {
      severity = InfoLogger::Severity::Fatal;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::error)) {
      severity = InfoLogger::Severity::Error;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::warn)) {
      severity = InfoLogger::Severity::Warning;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::state)) {
      severity = InfoLogger::Severity::Info;
      level = 10;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::info)) {
      severity = InfoLogger::Severity::Info;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug)) {
      severity = InfoLogger::Severity::Debug;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug1)) {
      severity = InfoLogger::Severity::Debug;
      level = 10;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug2)) {
      severity = InfoLogger::Severity::Debug;
      level = 20;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug3)) {
      severity = InfoLogger::Severity::Debug;
      level = 30;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::debug4)) {
      severity = InfoLogger::Severity::Debug;
      level = 40;
    } else if (metadata.severity_name == fair::Logger::SeverityName(fair::Severity::trace)) {
      severity = InfoLogger::Severity::Debug;
      level = 50;
    }

    InfoLogger::InfoLoggerMessageOption opt = {
      severity,
      level,
      InfoLogger::undefinedMessageOption.errorCode,
      metadata.file.c_str(),
      atoi(metadata.line.c_str())
    };

    if (logger) {
      logger->log(opt, *ctx, "DPL: %s", content.c_str());
    }
  };
};

int doChild(int argc, char** argv, const o2::framework::DeviceSpec& spec)
{
  fair::Logger::SetConsoleColor(false);
  LOG(INFO) << "Spawing new device " << spec.id << " in process with pid " << getpid();

  try {
    fair::mq::DeviceRunner runner{ argc, argv };

    // Populate options from the command line. Notice that only the options
    // declared in the workflow definition are allowed.
    runner.AddHook<fair::mq::hooks::SetCustomCmdLineOptions>([&spec](fair::mq::DeviceRunner& r) {
      boost::program_options::options_description optsDesc;
      ConfigParamsHelper::populateBoostProgramOptions(optsDesc, spec.options, gHiddenDeviceOptions);
      optsDesc.add_options()("monitoring-backend", bpo::value<std::string>()->default_value("infologger://"), "monitoring backend info") //
        ("infologger-severity", bpo::value<std::string>()->default_value(""), "minimum FairLogger severity to send to InfoLogger")       //
        ("infologger-mode", bpo::value<std::string>()->default_value(""), "INFOLOGGER_MODE override");
      r.fConfig.AddToCmdLineOptions(optsDesc, true);
    });

    // We initialise this in the driver, because different drivers might have
    // different versions of the service
    ServiceRegistry serviceRegistry;

    // This is to control lifetime. All these services get destroyed
    // when the runner is done.
    std::unique_ptr<LocalRootFileService> localRootFileService;
    std::unique_ptr<TextControlService> textControlService;
    std::unique_ptr<ParallelContext> parallelContext;
    std::unique_ptr<SimpleRawDeviceService> simpleRawDeviceService;
    std::unique_ptr<CallbackService> callbackService;
    std::unique_ptr<Monitoring> monitoringService;
    std::unique_ptr<InfoLogger> infoLoggerService;
    std::unique_ptr<InfoLoggerContext> infoLoggerContext;
    std::unique_ptr<TimesliceIndex> timesliceIndex;

    auto afterConfigParsingCallback = [&localRootFileService,
                                       &textControlService,
                                       &parallelContext,
                                       &simpleRawDeviceService,
                                       &callbackService,
                                       &monitoringService,
                                       &infoLoggerService,
                                       &spec,
                                       &serviceRegistry,
                                       &infoLoggerContext,
                                       &timesliceIndex](fair::mq::DeviceRunner& r) {
      localRootFileService = std::make_unique<LocalRootFileService>();
      textControlService = std::make_unique<TextControlService>();
      parallelContext = std::make_unique<ParallelContext>(spec.rank, spec.nSlots);
      simpleRawDeviceService = std::make_unique<SimpleRawDeviceService>(nullptr);
      callbackService = std::make_unique<CallbackService>();
      monitoringService = MonitoringFactory::Get(r.fConfig.GetStringValue("monitoring-backend"));
      auto infoLoggerMode = r.fConfig.GetStringValue("infologger-mode");
      if (infoLoggerMode != "") {
        setenv("INFOLOGGER_MODE", r.fConfig.GetStringValue("infologger-mode").c_str(), 1);
      }
      infoLoggerService = std::make_unique<InfoLogger>();
      infoLoggerContext = std::make_unique<InfoLoggerContext>();

      auto infoLoggerSeverity = r.fConfig.GetStringValue("infologger-severity");
      if (infoLoggerSeverity != "") {
        fair::Logger::AddCustomSink("infologger", infoLoggerSeverity, createInfoLoggerSinkHelper(infoLoggerService, infoLoggerContext));
      }
      timesliceIndex = std::make_unique<TimesliceIndex>();

      serviceRegistry.registerService<Monitoring>(monitoringService.get());
      serviceRegistry.registerService<InfoLogger>(infoLoggerService.get());
      serviceRegistry.registerService<RootFileService>(localRootFileService.get());
      serviceRegistry.registerService<ControlService>(textControlService.get());
      serviceRegistry.registerService<ParallelContext>(parallelContext.get());
      serviceRegistry.registerService<RawDeviceService>(simpleRawDeviceService.get());
      serviceRegistry.registerService<CallbackService>(callbackService.get());
      serviceRegistry.registerService<TimesliceIndex>(timesliceIndex.get());
      serviceRegistry.registerService<DeviceSpec>(&spec);

      // The decltype stuff is to be able to compile with both new and old
      // FairMQ API (one which uses a shared_ptr, the other one a unique_ptr.
      decltype(r.fDevice) device;
      device = std::move(make_matching<decltype(device), DataProcessingDevice>(spec, serviceRegistry));

      serviceRegistry.get<RawDeviceService>().setDevice(device.get());
      r.fDevice = std::move(device);
      fair::Logger::SetConsoleColor(false);
    };

    runner.AddHook<fair::mq::hooks::InstantiateDevice>(afterConfigParsingCallback);
    return runner.Run();
  } catch (std::exception& e) {
    LOG(ERROR) << "Unhandled exception reached the top of main: " << e.what() << ", device shutting down.";
    return 1;
  } catch (...) {
    LOG(ERROR) << "Unknown exception reached the top of main.\n";
    return 1;
  }
  return 0;
}

/// Remove all the GUI states from the tail of
/// the stack unless that's the only state on the stack.
void pruneGUI(std::vector<DriverState>& states)
{
  while (states.size() > 1 && states.back() == DriverState::GUI) {
    states.pop_back();
  }
}

struct WorkflowInfo {
  std::string executable;
  std::vector<std::string> args;
  std::vector<ConfigParamSpec> options;
};

// This is the handler for the parent inner loop.
int runStateMachine(DataProcessorSpecs const& workflow,
                    WorkflowInfo const& workflowInfo,
                    DataProcessorInfos const& previousDataProcessorInfos,
                    DriverControl& driverControl,
                    DriverInfo& driverInfo,
                    std::vector<DeviceMetricsInfo>& metricsInfos,
                    std::string frameworkId)
{
  DeviceSpecs deviceSpecs;
  DeviceInfos infos;
  DeviceControls controls;
  DeviceExecutions deviceExecutions;
  DataProcessorInfos dataProcessorInfos = previousDataProcessorInfos;
  auto resourceManager = std::make_unique<SimpleResourceManager>(driverInfo.startPort, driverInfo.portRange);

  void* window = nullptr;
  decltype(gui::getGUIDebugger(infos, deviceSpecs, dataProcessorInfos, metricsInfos, driverInfo, controls, driverControl)) debugGUICallback;

  // An empty frameworkId means this is the driver, so we initialise the GUI
  if (driverInfo.batch == false && frameworkId.empty()) {
    window = initGUI("O2 Framework debug GUI");
  }
  if (driverInfo.batch == false && window == nullptr) {
    LOG(WARN) << "Could not create GUI. Switching to batch mode. Do you have GLFW on your system?";
    driverInfo.batch = true;
  }
  bool guiQuitRequested = false;

  auto frameLast = std::chrono::high_resolution_clock::now();
  auto inputProcessingLast = frameLast;
  // FIXME: I should really have some way of exiting the
  // parent..
  DriverState current;
  DriverState previous;
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
      auto currentTime = std::chrono::steady_clock::now();
      std::chrono::duration<double> diff = currentTime - driverInfo.startTime;
      if ((graceful_exit == false) && (driverInfo.timeout > 0) && (diff.count() > driverInfo.timeout)) {
        LOG(INFO) << "Timout ellapsed. Requesting to quit.";
        graceful_exit = true;
      }
    }
    // Move to exit loop if sigint was sent we execute this only once.
    if (graceful_exit == true && driverInfo.sigintRequested == false) {
      driverInfo.sigintRequested = true;
      driverInfo.states.resize(0);
      driverInfo.states.push_back(DriverState::QUIT_REQUESTED);
      driverInfo.states.push_back(DriverState::GUI);
    }
    // If one of the children dies and sigint was not requested
    // we should decide what to do.
    if (sigchld_requested == true && driverInfo.sigchldRequested == false) {
      driverInfo.sigchldRequested = true;
      pruneGUI(driverInfo.states);
      driverInfo.states.push_back(DriverState::HANDLE_CHILDREN);
      driverInfo.states.push_back(DriverState::GUI);
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
        FD_ZERO(&(driverInfo.childFdset));

        /// After INIT we go into RUNNING and eventually to SCHEDULE from
        /// there and back into running. This is because the general case
        /// would be that we start an application and then we wait for
        /// resource offers from DDS or whatever resource manager we use.
        driverInfo.states.push_back(DriverState::RUNNING);
        driverInfo.states.push_back(DriverState::GUI);
        //        driverInfo.states.push_back(DriverState::REDEPLOY_GUI);
        LOG(INFO) << "O2 Data Processing Layer initialised. We brake for nobody.";
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
          dataProcessorInfos.push_back(
            DataProcessorInfo{
              device.id,
              workflowInfo.executable,
              workflowInfo.args,
              workflowInfo.options });
        }
        break;
      case DriverState::MATERIALISE_WORKFLOW:
        try {
          std::vector<ComputingResource> resources = resourceManager->getAvailableResources();
          DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow,
                                                            driverInfo.channelPolicies,
                                                            driverInfo.completionPolicies,
                                                            deviceSpecs,
                                                            resources);
          // This should expand nodes so that we can build a consistent DAG.
        } catch (std::runtime_error& e) {
          std::cerr << "Invalid workflow: " << e.what() << std::endl;
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
            return doChild(driverInfo.argc, driverInfo.argv, spec);
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
          debugGUICallback = gui::getGUIDebugger(infos, deviceSpecs, dataProcessorInfos, metricsInfos, driverInfo, controls, driverControl);
        }
        break;
      case DriverState::SCHEDULE: {
        // FIXME: for the moment modifying the topology means we rebuild completely
        //        all the devices and we restart them. This is also what DDS does at
        //        a larger scale. In principle one could try to do a delta and only
        //        restart the data processors which need to be restarted.
        LOG(INFO) << "Redeployment of configuration asked.";
        controls.resize(deviceSpecs.size());
        deviceExecutions.resize(deviceSpecs.size());

        DeviceSpecHelpers::prepareArguments(driverControl.defaultQuiet,
                                            driverControl.defaultStopped,
                                            dataProcessorInfos,
                                            deviceSpecs,
                                            deviceExecutions, controls);
        std::ostringstream forwardedStdin;
        WorkflowSerializationHelpers::dump(forwardedStdin, workflow, dataProcessorInfos);
        for (size_t di = 0; di < deviceSpecs.size(); ++di) {
          spawnDevice(forwardedStdin.str(),
                      deviceSpecs[di], driverInfo.socket2DeviceInfo, controls[di], deviceExecutions[di], infos,
                      driverInfo.maxFd, driverInfo.childFdset);
        }
        driverInfo.maxFd += 1;
        assert(infos.empty() == false);
        LOG(INFO) << "Redeployment of configuration done.";
      } break;
      case DriverState::RUNNING:
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
          driverInfo.states.push_back(DriverState::GUI);
        } else if (infos.size() != deviceSpecs.size()) {
          // If the number of deviceSpecs is different from
          // the DeviceInfos it means the speicification
          // does not match what is running, so we need to do
          // further scheduling.
          driverInfo.states.push_back(DriverState::RUNNING);
          driverInfo.states.push_back(DriverState::GUI);
          driverInfo.states.push_back(DriverState::REDEPLOY_GUI);
          driverInfo.states.push_back(DriverState::GUI);
          driverInfo.states.push_back(DriverState::SCHEDULE);
          driverInfo.states.push_back(DriverState::GUI);
        } else if (deviceSpecs.size() == 0) {
          LOG(INFO) << "No device resulting from the workflow. Quitting.";
          // If there are no deviceSpecs, we exit.
          driverInfo.states.push_back(DriverState::EXIT);
        } else {
          driverInfo.states.push_back(DriverState::RUNNING);
          driverInfo.states.push_back(DriverState::GUI);
        }
        {
          usleep(1000); // We wait for 1 millisecond between one processing
                        // and the other.
          auto inputProcessingStart = std::chrono::high_resolution_clock::now();
          auto inputProcessingLatency = inputProcessingStart - inputProcessingLast;
          processChildrenOutput(driverInfo, infos, deviceSpecs, controls, metricsInfos);
          auto inputProcessingEnd = std::chrono::high_resolution_clock::now();
          driverInfo.inputProcessingCost = std::chrono::duration_cast<std::chrono::milliseconds>(inputProcessingEnd - inputProcessingStart).count();
          driverInfo.inputProcessingLatency = std::chrono::duration_cast<std::chrono::milliseconds>(inputProcessingLatency).count();
          inputProcessingLast = inputProcessingStart;
        }
        break;
      case DriverState::GUI:
        if (window) {
          auto frameStart = std::chrono::high_resolution_clock::now();
          auto frameLatency = frameStart - frameLast;
          // We want to render at ~60 frames per second, so latency needs to be ~16ms
          if (std::chrono::duration_cast<std::chrono::milliseconds>(frameLatency).count() > 20) {
            guiQuitRequested = (pollGUI(window, debugGUICallback) == false);
            auto frameEnd = std::chrono::high_resolution_clock::now();
            driverInfo.frameCost = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart).count();
            driverInfo.frameLatency = std::chrono::duration_cast<std::chrono::milliseconds>(frameLatency).count();
            frameLast = frameStart;
          }
        }
        break;
      case DriverState::QUIT_REQUESTED:
        LOG(INFO) << "QUIT_REQUESTED" << std::endl;
        guiQuitRequested = true;
        killChildren(infos, SIGTERM);
        driverInfo.states.push_back(DriverState::HANDLE_CHILDREN);
        driverInfo.states.push_back(DriverState::GUI);
        break;
      case DriverState::HANDLE_CHILDREN:
        // I allow queueing of more sigchld only when
        // I process the previous call
        sigchld_requested = false;
        driverInfo.sigchldRequested = false;
        processChildrenOutput(driverInfo, infos, deviceSpecs, controls, metricsInfos);
        processSigChild(infos);
        if (areAllChildrenGone(infos) == true &&
            (guiQuitRequested || (checkIfCanExit(infos) == true) || graceful_exit)) {
          // We move to the exit, regardless of where we were
          driverInfo.states.resize(0);
          driverInfo.states.push_back(DriverState::EXIT);
          driverInfo.states.push_back(DriverState::GUI);
        } else if (areAllChildrenGone(infos) == false &&
                   (guiQuitRequested || checkIfCanExit(infos) == true || graceful_exit)) {
          driverInfo.states.push_back(DriverState::HANDLE_CHILDREN);
          driverInfo.states.push_back(DriverState::GUI);
        } else {
          driverInfo.states.push_back(DriverState::GUI);
        }
        break;
      case DriverState::EXIT:
        return calculateExitCode(infos);
      case DriverState::PERFORM_CALLBACKS:
        for (auto& callback : driverControl.callbacks) {
          callback(workflow, deviceSpecs, deviceExecutions, dataProcessorInfos);
        }
        driverControl.callbacks.clear();
        driverInfo.states.push_back(DriverState::GUI);
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
    control.callbacks = { [](WorkflowSpec const& workflow,
                             DeviceSpecs const& specs,
                             DeviceExecutions const&,
                             DataProcessorInfos&) {
      GraphvizHelpers::dumpDeviceSpec2Graphviz(std::cout, specs);
    } };
    control.forcedTransitions = {
      DriverState::EXIT,                    //
      DriverState::PERFORM_CALLBACKS,       //
      DriverState::IMPORT_CURRENT_WORKFLOW, //
      DriverState::MATERIALISE_WORKFLOW     //
    };
  } else if (varmap["dds"].as<bool>()) {
    // Dump a DDS representation of what I will do.
    // Notice that compared to DDS we need to schedule things,
    // because DDS needs to be able to have actual Executions in
    // order to provide a correct configuration.
    control.callbacks = { [](WorkflowSpec const& workflow,
                             DeviceSpecs const& specs,
                             DeviceExecutions const& executions,
                             DataProcessorInfos&) {
      dumpDeviceSpec2DDS(std::cout, specs, executions);
    } };
    control.forcedTransitions = {
      DriverState::EXIT,                    //
      DriverState::PERFORM_CALLBACKS,       //
      DriverState::SCHEDULE,                //
      DriverState::IMPORT_CURRENT_WORKFLOW, //
      DriverState::MATERIALISE_WORKFLOW     //
    };
  } else if (varmap["o2-control"].as<bool>()) {
    control.callbacks = { [](WorkflowSpec const& workflow,
                             DeviceSpecs const& specs,
                             DeviceExecutions const& executions,
                             DataProcessorInfos&) {
      dumpDeviceSpec2O2Control(std::cout, specs, executions);
    } };
    control.forcedTransitions = {
      DriverState::EXIT,                    //
      DriverState::PERFORM_CALLBACKS,       //
      DriverState::SCHEDULE,                //
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
      DriverState::IMPORT_CURRENT_WORKFLOW, //
      DriverState::MATERIALISE_WORKFLOW     //
    };
  } else if ((varmap["dump-workflow"].as<bool>() == true) || (varmap["run"].as<bool>() == false && varmap.count("id") == 0 && isOutputToPipe())) {
    control.callbacks = { [](WorkflowSpec const& workflow,
                             DeviceSpecs const devices,
                             DeviceExecutions const&,
                             DataProcessorInfos& dataProcessorInfos) {
      WorkflowSerializationHelpers::dump(std::cout, workflow, dataProcessorInfos);
      // FIXME: this is to avoid trailing garbage..
      exit(0);
    } };
    control.forcedTransitions = {
      DriverState::EXIT,                    //
      DriverState::PERFORM_CALLBACKS,       //
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
           std::vector<ConfigParamSpec> const& currentWorkflowOptions,
           o2::framework::ConfigContext& configContext)
{
  std::vector<std::string> currentArgs;
  for (size_t ai = 1; ai < argc; ++ai) {
    currentArgs.push_back(argv[ai]);
  }

  WorkflowInfo currentWorkflow{
    argv[0],
    currentArgs,
    currentWorkflowOptions
  };

  enum TerminationPolicy policy;
  bpo::options_description executorOptions("Executor options");
  const char* helpDescription = "print help: short, full, executor, or processor name";
  executorOptions.add_options()                                                                             //
    ("help,h", bpo::value<std::string>()->implicit_value("short"), helpDescription)                         //
    ("quiet,q", bpo::value<bool>()->zero_tokens()->default_value(false), "quiet operation")                 //
    ("stop,s", bpo::value<bool>()->zero_tokens()->default_value(false), "stop before device start")         //
    ("single-step", bpo::value<bool>()->zero_tokens()->default_value(false), "start in single step mode")   //
    ("batch,b", bpo::value<bool>()->zero_tokens()->default_value(false), "batch processing mode")           //
    ("start-port,p", bpo::value<unsigned short>()->default_value(22000), "start port to allocate")          //
    ("port-range,pr", bpo::value<unsigned short>()->default_value(1000), "ports in range")                  //
    ("completion-policy,c", bpo::value<TerminationPolicy>(&policy)->default_value(TerminationPolicy::QUIT), //
     "what to do when processing is finished: quit, wait")                                                  //
    ("graphviz,g", bpo::value<bool>()->zero_tokens()->default_value(false), "produce graph output")         //
    ("timeout,t", bpo::value<double>()->default_value(0), "timeout after which to exit")                    //
    ("dds,D", bpo::value<bool>()->zero_tokens()->default_value(false), "create DDS configuration")          //
    ("dump-workflow", bpo::value<bool>()->zero_tokens()->default_value(false), "dump workflow as JSON")     //
    ("run", bpo::value<bool>()->zero_tokens()->default_value(false), "run workflow merged so far")          //
    ("o2-control,o2", bpo::value<bool>()->zero_tokens()->default_value(false), "create O2 Control configuration");
  // some of the options must be forwarded by default to the device
  executorOptions.add(DeviceSpecHelpers::getForwardedDeviceOptions());

  gHiddenDeviceOptions.add_options()                                                    //
    ((std::string("id") + ",i").c_str(), bpo::value<std::string>(),                     //
     "device id for child spawning")                                                    //
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

  WorkflowHelpers::injectServiceDevices(physicalWorkflow);
  std::stable_sort(physicalWorkflow.begin(), physicalWorkflow.end(), [&rankIndex](DataProcessorSpec const& a, DataProcessorSpec const& b) {
    return rankIndex[a.name] < rankIndex[b.name];
  });

  // Use the hidden options as veto, all config specs matching a definition
  // in the hidden options are skipped in order to avoid duplicate definitions
  // in the main parser. Note: all config specs are forwarded to devices
  visibleOptions.add(ConfigParamsHelper::prepareOptionDescriptions(physicalWorkflow, currentWorkflowOptions, gHiddenDeviceOptions));

  bpo::options_description od;
  od.add(visibleOptions);
  od.add(gHiddenDeviceOptions);

  // FIXME: decide about the policy for handling unrecognized arguments
  // command_line_parser with option allow_unregistered() can be used
  bpo::variables_map varmap;
  try {
    bpo::store(bpo::parse_command_line(argc, argv, od), varmap);
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    exit(1);
  }

  if (varmap.count("help")) {
    printHelp(varmap, executorOptions, physicalWorkflow, currentWorkflowOptions);
    exit(0);
  }
  DriverControl driverControl;
  initialiseDriverControl(varmap, driverControl);

  DriverInfo driverInfo;
  driverInfo.maxFd = 0;
  driverInfo.states.reserve(10);
  driverInfo.sigintRequested = false;
  driverInfo.sigchldRequested = false;
  driverInfo.channelPolicies = channelPolicies;
  driverInfo.completionPolicies = completionPolicies;
  driverInfo.argc = argc;
  driverInfo.argv = argv;
  driverInfo.batch = varmap["batch"].as<bool>();
  driverInfo.terminationPolicy = varmap["completion-policy"].as<TerminationPolicy>();
  driverInfo.startTime = std::chrono::steady_clock::now();
  driverInfo.timeout = varmap["timeout"].as<double>();
  driverInfo.startPort = varmap["start-port"].as<unsigned short>();
  driverInfo.portRange = varmap["port-range"].as<unsigned short>();
  // FIXME: should use the whole dataProcessorInfos, actually...
  driverInfo.processorInfo = dataProcessorInfos;
  driverInfo.configContext = &configContext;

  std::string frameworkId;
  // If the id is set, this means this is a device,
  // otherwise this is the driver.
  FreePortFinder finder(driverInfo.startPort - 1,
                        65535 - driverInfo.portRange,
                        driverInfo.portRange);
  if (varmap.count("id")) {
    frameworkId = varmap["id"].as<std::string>();
  } else {
    finder.scan();
    driverInfo.startPort = finder.port();
    driverInfo.portRange = finder.range();
  }
  return runStateMachine(physicalWorkflow,
                         currentWorkflow,
                         dataProcessorInfos,
                         driverControl,
                         driverInfo,
                         gDeviceMetricsInfos,
                         frameworkId);
}
