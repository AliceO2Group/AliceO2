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
#include <stdexcept>
#include "Framework/BoostOptionsRetriever.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/ChannelConfigurationPolicy.h"
#include "Framework/ChannelMatching.h"
#include "Framework/ConfigParamsHelper.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigContext.h"
#include "Framework/ComputingQuotaEvaluator.h"
#include "CommonDriverServices.h"
#include "Framework/DataProcessingDevice.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Plugins.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceExecution.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceMetricsHelper.h"
#include "Framework/DeviceConfigInfo.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceState.h"
#include "DeviceStateHelpers.h"
#include "Framework/DevicesManager.h"
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
#include "Framework/CommandInfo.h"
#include "Framework/RunningWorkflowInfo.h"
#include "Framework/TopologyPolicy.h"
#include "Framework/WorkflowSpecNode.h"
#include "Framework/GuiCallbackContext.h"
#include "Framework/DeviceContext.h"
#include "ControlServiceHelpers.h"
#include "ProcessingPoliciesHelpers.h"
#include "DriverServerContext.h"
#include "HTTPParser.h"
#include "DPLWebSocket.h"
#include "ArrowSupport.h"

#include "ComputingResourceHelpers.h"
#include "DataProcessingStatus.h"
#include "DDSConfigHelpers.h"
#include "O2ControlHelpers.h"
#include "DeviceSpecHelpers.h"
#include "GraphvizHelpers.h"
#include "MermaidHelpers.h"
#include "PropertyTreeHelpers.h"
#include "SimpleResourceManager.h"
#include "WorkflowSerializationHelpers.h"

#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>
#include <Monitoring/MonitoringFactory.h>
#include "ResourcesMonitoringHelper.h"

#include <fairmq/Device.h>
#include <fairmq/DeviceRunner.h>
#include <fairmq/shmem/Monitor.h>
#include <fairmq/ProgOptions.h>

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
// This is to allow C++20 aggregate initialisation
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#if defined(__linux__) && __has_include(<sched.h>)
#include <sched.h>
#elif __has_include(<linux/getcpu.h>)
#include <linux/getcpu.h>
#elif __has_include(<cpuid.h>) && (__x86_64__ || __i386__)
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

using namespace o2::framework;
namespace bpo = boost::program_options;
using DataProcessorInfos = std::vector<DataProcessorInfo>;
using DeviceExecutions = std::vector<DeviceExecution>;
using DeviceSpecs = std::vector<DeviceSpec>;
using DeviceInfos = std::vector<DeviceInfo>;
using DeviceControls = std::vector<DeviceControl>;
using DataProcessorSpecs = std::vector<DataProcessorSpec>;

std::vector<DeviceMetricsInfo> gDeviceMetricsInfos;

// FIXME: probably find a better place
// these are the device options added by the framework, but they can be
// overloaded in the config spec
bpo::options_description gHiddenDeviceOptions("Hidden child options");

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
namespace
{
int calculateExitCode(DriverInfo& driverInfo, DeviceSpecs& deviceSpecs, DeviceInfos& infos)
{
  std::regex regexp(R"(^\[([\d+:]*)\]\[\w+\] )");
  if (!driverInfo.lastError.empty()) {
    LOGP(error, "SEVERE: DPL driver encountered an error while running.\n{}",
         driverInfo.lastError);
    return 1;
  }
  for (size_t di = 0; di < deviceSpecs.size(); ++di) {
    auto& info = infos[di];
    auto& spec = deviceSpecs[di];
    if (info.maxLogLevel >= driverInfo.minFailureLevel) {
      LOGP(error, "SEVERE: Device {} ({}) had at least one message above severity {}: {}",
           spec.name,
           info.pid,
           (int)info.minFailureLevel,
           std::regex_replace(info.firstSevereError, regexp, ""));
      return 1;
    }
    if (info.exitStatus != 0) {
      LOGP(error, "SEVERE: Device {} ({}) returned with {}",
           spec.name,
           info.pid,
           info.exitStatus);
      return info.exitStatus;
    }
  }
  return 0;
}
} // namespace

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
volatile sig_atomic_t double_sigint = false;

static void handle_sigint(int)
{
  if (graceful_exit == false) {
    graceful_exit = true;
  } else {
    forceful_exit = true;
    // We keep track about forceful exiting via
    // a double SIGINT, so that we do not print
    // any extra message. This means that if the
    // forceful_exit is set by the timer, we will
    // get an error message about each child which
    // did not gracefully exited.
    double_sigint = true;
  }
}

/// Helper to invoke shared memory cleanup
void cleanupSHM(std::string const& uniqueWorkflowId)
{
  using namespace fair::mq::shmem;
  fair::mq::shmem::Monitor::Cleanup(SessionId{"dpl_" + uniqueWorkflowId}, false);
}

static void handle_sigchld(int) { sigchld_requested = true; }

void spawnRemoteDevice(std::string const&,
                       DeviceSpec const& spec,
                       DeviceControl&,
                       DeviceExecution&,
                       std::vector<DeviceInfo>& deviceInfos)
{
  LOG(info) << "Starting " << spec.id << " as remote device";
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
  info.outputsViewIndex = Metric2DViewIndex{"output_matchers", 0, 0, {}};
  info.inputChannelMetricsViewIndex = Metric2DViewIndex{"oldest_possible_timeslice", 0, 0, {}};
  info.outputChannelMetricsViewIndex = Metric2DViewIndex{"oldest_possible_output", 0, 0, {}};
  // FIXME: use uv_now.
  info.lastSignal = uv_hrtime() - 10000000;

  deviceInfos.emplace_back(info);
  // Let's add also metrics information for the given device
  gDeviceMetricsInfos.emplace_back(DeviceMetricsInfo{});
}

struct DeviceLogContext {
  int fd;
  int index;
  uv_loop_t* loop;
  std::vector<DeviceInfo>* infos;
};

void log_callback(uv_poll_t* handle, int status, int events)
{
  auto* logContext = reinterpret_cast<DeviceLogContext*>(handle->data);
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

void close_websocket(uv_handle_t* handle)
{
  LOG(debug) << "Handle is being closed";
  delete (WSDPLHandler*)handle->data;
}

void websocket_callback(uv_stream_t* stream, ssize_t nread, const uv_buf_t* buf)
{
  auto* handler = (WSDPLHandler*)stream->data;
  if (nread == 0) {
    return;
  }
  if (nread == UV_EOF) {
    if (buf->base) {
      free(buf->base);
    }
    uv_read_stop(stream);
    uv_close((uv_handle_t*)stream, close_websocket);
    return;
  }
  if (nread < 0) {
    // FIXME: should I close?
    LOG(error) << "websocket_callback: Error while reading from websocket";
    if (buf->base) {
      free(buf->base);
    }
    uv_read_stop(stream);
    uv_close((uv_handle_t*)stream, close_websocket);
    return;
  }
  try {
    LOG(debug3) << "Parsing request with " << handler << " with " << nread << " bytes";
    parse_http_request(buf->base, nread, handler);
    if (buf->base) {
      free(buf->base);
    }
  } catch (WSError& e) {
    LOG(error) << "Error while parsing request: " << e.message;
    handler->error(e.code, e.message.c_str());
  }
}

static void my_alloc_cb(uv_handle_t*, size_t suggested_size, uv_buf_t* buf)
{
  buf->base = (char*)malloc(suggested_size);
  buf->len = suggested_size;
}

/// An handler for a websocket message stream.
struct ControlWebSocketHandler : public WebSocketHandler {
  ControlWebSocketHandler(DriverServerContext& context)
    : mContext{context}
  {
  }
  ~ControlWebSocketHandler() override = default;

  /// Invoked at the end of the headers.
  /// as a special header we have "x-dpl-pid" which devices can use
  /// to identify themselves.
  /// FIXME: No effort is done to guarantee their identity. Maybe each device
  ///        should be started with a unique secret if we wanted to provide
  ///        some secutity.
  void headers(std::map<std::string, std::string> const& headers) override
  {
    if (headers.count("x-dpl-pid")) {
      auto s = headers.find("x-dpl-pid");
      this->mPid = std::stoi(s->second);
      for (size_t di = 0; di < mContext.infos->size(); ++di) {
        if ((*mContext.infos)[di].pid == mPid) {
          mIndex = di;
          return;
        }
      }
    }
  }
  /// FIXME: not implemented by the backend.
  void beginFragmentation() override {}

  /// Invoked when a frame it's parsed. Notice you do not own the data and you must
  /// not free the memory.
  void frame(char const* frame, size_t s) override
  {
    bool hasNewMetric = false;
    auto updateMetricsViews = Metric2DViewIndex::getUpdater({&(*mContext.infos)[mIndex].dataRelayerViewIndex,
                                                             &(*mContext.infos)[mIndex].variablesViewIndex,
                                                             &(*mContext.infos)[mIndex].queriesViewIndex,
                                                             &(*mContext.infos)[mIndex].outputsViewIndex,
                                                             &(*mContext.infos)[mIndex].inputChannelMetricsViewIndex});

    auto newMetricCallback = [&updateMetricsViews, &hasNewMetric](std::string const& name, MetricInfo const& metric, int value, size_t metricIndex) {
      updateMetricsViews(name, metric, value, metricIndex);
      hasNewMetric = true;
    };
    std::string token(frame, s);
    std::match_results<std::string_view::const_iterator> match;
    ParsedConfigMatch configMatch;
    ParsedMetricMatch metricMatch;

    auto doParseConfig = [](std::string_view const& token, ParsedConfigMatch& configMatch, DeviceInfo& info) -> bool {
      if (DeviceConfigHelper::parseConfig(token, configMatch)) {
        DeviceConfigHelper::processConfig(configMatch, info);
        return true;
      }
      return false;
    };
    LOG(debug3) << "Data received: " << std::string_view(frame, s);
    if (DeviceMetricsHelper::parseMetric(token, metricMatch)) {
      // We use this callback to cache which metrics are needed to provide a
      // the DataRelayer view.
      assert(mContext.metrics);
      DeviceMetricsHelper::processMetric(metricMatch, (*mContext.metrics)[mIndex], newMetricCallback);
      didProcessMetric = true;
      didHaveNewMetric |= hasNewMetric;
    } else if (ControlServiceHelpers::parseControl(token, match) && mContext.infos) {
      ControlServiceHelpers::processCommand(*mContext.infos, mPid, match[1].str(), match[2].str());
    } else if (doParseConfig(token, configMatch, (*mContext.infos)[mIndex]) && mContext.infos) {
      LOG(debug2) << "Found configuration information for pid " << mPid;
    } else {
      LOG(error) << "Unexpected control data: " << std::string_view(frame, s);
    }
  }

  /// FIXME: not implemented
  void endFragmentation() override{};
  /// FIXME: not implemented
  void control(char const*, size_t) override{};

  /// Invoked at the beginning of some incoming data. We simply
  /// reset actions which need to happen on a per chunk basis.
  void beginChunk() override
  {
    didProcessMetric = false;
    didHaveNewMetric = false;
  }

  /// Invoked after we have processed all the available incoming data.
  /// In this particular case we must handle the metric callbacks, if
  /// needed.
  void endChunk() override
  {
    if (!didProcessMetric) {
      return;
    }
    size_t timestamp = uv_now(mContext.loop);
    for (auto& callback : *mContext.metricProcessingCallbacks) {
      callback(mContext.registry, *mContext.metrics, *mContext.specs, *mContext.infos, mContext.driver->metrics, timestamp);
    }
    for (auto& metricsInfo : *mContext.metrics) {
      std::fill(metricsInfo.changed.begin(), metricsInfo.changed.end(), false);
    }
  }

  /// The driver context were we want to accumulate changes
  /// which we got from the websocket.
  DriverServerContext& mContext;
  /// The pid of the remote process actually associated to this
  /// handler. Notice that this information comes as part of
  /// the HTTP headers via x-dpl-pid.
  pid_t mPid = 0;
  /// The index of the remote process associated to this handler.
  size_t mIndex = (size_t)-1;
  /// Wether any frame operation between beginChunk and endChunk
  /// actually processed some metric.
  bool didProcessMetric = false;
  bool didHaveNewMetric = false;
};

/// A callback for the rest engine
void ws_connect_callback(uv_stream_t* server, int status)
{
  auto* serverContext = reinterpret_cast<DriverServerContext*>(server->data);
  if (status < 0) {
    LOGF(error, "New connection error %s\n", uv_strerror(status));
    // error!
    return;
  }

  auto* client = (uv_tcp_t*)malloc(sizeof(uv_tcp_t));
  uv_tcp_init(serverContext->loop, client);
  if (uv_accept(server, (uv_stream_t*)client) == 0) {
    client->data = new WSDPLHandler((uv_stream_t*)client, serverContext);
    uv_read_start((uv_stream_t*)client, (uv_alloc_cb)my_alloc_cb, websocket_callback);
  } else {
    uv_close((uv_handle_t*)client, nullptr);
  }
}

struct StreamConfigContext {
  std::string configuration;
  int fd;
};

void stream_config(uv_work_t* req)
{
  auto* context = (StreamConfigContext*)req->data;
  size_t result = write(context->fd, context->configuration.data(), context->configuration.size());
  if (result != context->configuration.size()) {
    LOG(error) << "Unable to pass configuration to children";
  }
  {
    auto error = fsync(context->fd);
    switch (error) {
      case EBADF:
        LOGP(error, "EBADF while flushing child stdin");
        break;
      case EINVAL:
        LOGP(error, "EINVAL while flushing child stdin");
        break;
      case EINTR:
        LOGP(error, "EINTR while flushing child stdin");
        break;
      case EIO:
        LOGP(error, "EIO while flushing child stdin");
        break;
      default:;
    }
  }
  {
    auto error = close(context->fd); // Not allowing further communication...
    switch (error) {
      case EBADF:
        LOGP(error, "EBADF while closing child stdin");
        break;
      case EINTR:
        LOGP(error, "EINTR while closing child stdin");
        break;
      case EIO:
        LOGP(error, "EIO while closing child stdin");
        break;
      default:;
    }
  }
}

struct DeviceRef {
  int index;
};

struct DeviceStdioContext {
  int childstdin[2];
  int childstdout[2];
};

void handleSignals()
{
  struct sigaction sa_handle_int;
  sa_handle_int.sa_handler = handle_sigint;
  sigemptyset(&sa_handle_int.sa_mask);
  sa_handle_int.sa_flags = SA_RESTART;
  if (sigaction(SIGINT, &sa_handle_int, nullptr) == -1) {
    perror("Unable to install signal handler");
    exit(1);
  }
  struct sigaction sa_handle_term;
  sa_handle_term.sa_handler = handle_sigint;
  sigemptyset(&sa_handle_term.sa_mask);
  sa_handle_term.sa_flags = SA_RESTART;
  if (sigaction(SIGTERM, &sa_handle_int, nullptr) == -1) {
    perror("Unable to install signal handler");
    exit(1);
  }
}

void handleChildrenStdio(uv_loop_t* loop,
                         std::string const& forwardedStdin,
                         std::vector<DeviceInfo>& deviceInfos,
                         std::vector<DeviceStdioContext>& childFds,
                         std::vector<uv_poll_t*>& handles)
{
  for (size_t i = 0; i < childFds.size(); ++i) {
    auto& childstdin = childFds[i].childstdin;
    auto& childstdout = childFds[i].childstdout;

    auto* req = (uv_work_t*)malloc(sizeof(uv_work_t));
    req->data = new StreamConfigContext{forwardedStdin, childstdin[1]};
    uv_queue_work(loop, req, stream_config, nullptr);

    // Setting them to non-blocking to avoid haing the driver hang when
    // reading from child.
    int resultCode = fcntl(childstdout[0], F_SETFL, O_NONBLOCK);
    if (resultCode == -1) {
      LOGP(error, "Error while setting the socket to non-blocking: {}", strerror(errno));
    }

    /// Add pollers for stdout and stderr
    auto addPoller = [&handles, &deviceInfos, &loop](int index, int fd) {
      auto* context = new DeviceLogContext{};
      context->index = index;
      context->fd = fd;
      context->loop = loop;
      context->infos = &deviceInfos;
      handles.push_back((uv_poll_t*)malloc(sizeof(uv_poll_t)));
      auto handle = handles.back();
      handle->data = context;
      uv_poll_init(loop, handle, fd);
      uv_poll_start(handle, UV_READABLE, log_callback);
    };

    addPoller(i, childstdout[0]);
  }
}

void handle_crash(int sig)
{
  // dump demangled stack trace
  void* array[1024];

  int size = backtrace(array, 1024);

  {
    char const* msg = "*** Program crashed (Segmentation fault, FPE, BUS, ABRT, KILL, Unhandled Exception, ...)\nBacktrace by DPL:\n";
    auto retVal = write(STDERR_FILENO, msg, strlen(msg));
    msg = "UNKNOWN SIGNAL\n";
    if (sig == SIGSEGV) {
      msg = "SEGMENTATION FAULT\n";
    } else if (sig == SIGABRT) {
      msg = "ABRT\n";
    } else if (sig == SIGBUS) {
      msg = "BUS ERROR\n";
    } else if (sig == SIGILL) {
      msg = "ILLEGAL INSTRUCTION\n";
    } else if (sig == SIGFPE) {
      msg = "FLOATING POINT EXCEPTION\n";
    }
    retVal = write(STDERR_FILENO, msg, strlen(msg));
    (void)retVal;
  }
  demangled_backtrace_symbols(array, size, STDERR_FILENO);
  {
    char const* msg = "Backtrace complete.\n";
    int len = strlen(msg); /* the byte length of the string */

    auto retVal = write(STDERR_FILENO, msg, len);
    (void)retVal;
    fsync(STDERR_FILENO);
  }
  _exit(1);
}

/// This will start a new device by forking and executing a
/// new child
void spawnDevice(DeviceRef ref,
                 std::vector<DeviceSpec> const& specs,
                 DriverInfo& driverInfo,
                 std::vector<DeviceControl>&,
                 std::vector<DeviceExecution>& executions,
                 std::vector<DeviceInfo>& deviceInfos,
                 ServiceRegistryRef serviceRegistry,
                 boost::program_options::variables_map& varmap,
                 std::vector<DeviceStdioContext>& childFds,
                 unsigned parentCPU,
                 unsigned parentNode)
{
  // FIXME: this might not work when more than one DPL driver on the same
  // machine. Hopefully we do not care.
  // Not how the first port is actually used to broadcast clients.
  auto& spec = specs[ref.index];
  auto& execution = executions[ref.index];

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
    // We also close all the filedescriptors for our sibilings.
    struct rlimit rlim;
    getrlimit(RLIMIT_NOFILE, &rlim);
    // We close all FD, but the one which are actually
    // used to communicate with the driver.
    int rlim_cur = rlim.rlim_cur;
    for (int i = 0; i < rlim_cur; ++i) {
      if (childFds[ref.index].childstdin[0] == i) {
        continue;
      }
      if (childFds[ref.index].childstdout[1] == i) {
        continue;
      }
      close(i);
    }
    dup2(childFds[ref.index].childstdin[0], STDIN_FILENO);
    dup2(childFds[ref.index].childstdout[1], STDOUT_FILENO);
    dup2(childFds[ref.index].childstdout[1], STDERR_FILENO);

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
  close(childFds[ref.index].childstdin[0]);
  close(childFds[ref.index].childstdout[1]);
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

  LOG(info) << "Starting " << spec.id << " on pid " << id;
  DeviceInfo info;
  info.pid = id;
  info.active = true;
  info.readyToQuit = false;
  info.historySize = 1000;
  info.historyPos = 0;
  info.maxLogLevel = LogParsingHelpers::LogLevel::Debug;
  info.minFailureLevel = driverInfo.minFailureLevel;
  info.dataRelayerViewIndex = Metric2DViewIndex{"data_relayer", 0, 0, {}};
  info.variablesViewIndex = Metric2DViewIndex{"matcher_variables", 0, 0, {}};
  info.queriesViewIndex = Metric2DViewIndex{"data_queries", 0, 0, {}};
  info.outputsViewIndex = Metric2DViewIndex{"output_matchers", 0, 0, {}};
  info.inputChannelMetricsViewIndex = Metric2DViewIndex{"oldest_possible_timeslice", 0, 0, {}};
  info.outputChannelMetricsViewIndex = Metric2DViewIndex{"oldest_possible_output", 0, 0, {}};
  info.tracyPort = driverInfo.tracyPort;
  info.lastSignal = uv_hrtime() - 10000000;

  deviceInfos.emplace_back(info);
  // Let's add also metrics information for the given device
  gDeviceMetricsInfos.emplace_back(DeviceMetricsInfo{});
}

struct LogProcessingState {
  bool didProcessLog = false;
  bool didProcessControl = true;
  bool didProcessConfig = true;
  bool didProcessMetric = false;
  bool hasNewMetric = false;
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
  std::match_results<std::string_view::const_iterator> match;
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
                                     &info.queriesViewIndex,
                                     &info.outputsViewIndex});

    auto newMetricCallback = [&updateMetricsViews, &hasNewMetric](std::string const& name, MetricInfo const& metric, int value, size_t metricIndex) {
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
      } else if (logLevel == LogParsingHelpers::LogLevel::Info && ControlServiceHelpers::parseControl(token, match)) {
        ControlServiceHelpers::processCommand(infos, info.pid, match[1].str(), match[2].str());
        result.didProcessControl = true;
      } else if (logLevel == LogParsingHelpers::LogLevel::Info && DeviceConfigHelper::parseConfig(token.substr(16), configMatch)) {
        DeviceConfigHelper::processConfig(configMatch, info);
        result.didProcessConfig = true;
      } else if (!control.quiet && (token.find(control.logFilter) != std::string::npos) &&
                 logLevel >= control.logLevel) {
        assert(info.historyPos >= 0);
        assert(info.historyPos < info.history.size());
        info.history[info.historyPos] = token;
        info.historyLevel[info.historyPos] = logLevel;
        info.historyPos = (info.historyPos + 1) % info.history.size();
        fmt::print("[{}:{}]: {}\n", info.pid, spec.id, token);
        result.didProcessLog = true;
      }
      // We keep track of the maximum log error a
      // device has seen.
      bool maxLogLevelIncreased = false;
      if (logLevel > info.maxLogLevel && logLevel > LogParsingHelpers::LogLevel::Info &&
          logLevel != LogParsingHelpers::LogLevel::Unknown) {
        info.maxLogLevel = logLevel;
        maxLogLevelIncreased = true;
      }
      if (logLevel >= driverInfo.minFailureLevel) {
        info.lastError = token;
        if (info.firstSevereError.empty() || maxLogLevelIncreased) {
          info.firstSevereError = token;
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
  }
  return result;
}

// Process all the sigchld which are pending
// @return wether or not a given child exited with an error condition.
bool processSigChild(DeviceInfos& infos, DeviceSpecs& specs)
{
  bool hasError = false;
  while (true) {
    int status;
    pid_t pid = waitpid((pid_t)(-1), &status, WNOHANG);
    if (pid > 0) {
      int es = WEXITSTATUS(status);

      if (WIFEXITED(status) == false || es != 0) {
        es = WIFEXITED(status) ? es : 128 + es;
        // Look for the name associated to the pid in the infos
        std::string id = "unknown";
        assert(specs.size() == infos.size());
        for (size_t ii = 0; ii < infos.size(); ++ii) {
          if (infos[ii].pid == pid) {
            id = specs[ii].id;
          }
        }
        // No need to print anything if the user
        // force quitted doing a double Ctrl-C.
        if (double_sigint) {
        } else if (forceful_exit) {
          LOGP(error, "pid {} ({}) was forcefully terminated after being requested to quit", pid, id);
        } else {
          LOGP(error, "pid {} ({}) crashed with {}", pid, id, es);
        }
        hasError |= true;
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

void doDPLException(RuntimeErrorRef& e, char const* processName)
{
  auto& err = o2::framework::error_from_ref(e);
  if (err.maxBacktrace != 0) {
    LOGP(fatal,
         "Unhandled o2::framework::runtime_error reached the top of main of {}, device shutting down."
         " Reason: {}"
         "\n Backtrace follow: \n",
         processName, err.what);
    demangled_backtrace_symbols(err.backtrace, err.maxBacktrace, STDERR_FILENO);
  } else {
    LOGP(fatal,
         "Unhandled o2::framework::runtime_error reached the top of main of {}, device shutting down."
         " Reason: {}"
         "\n Recompile with DPL_ENABLE_BACKTRACE=1 to get more information.",
         processName, err.what);
  }
}

void doUnknownException(std::string const& s, char const* processName)
{
  if (s.empty()) {
    LOGP(fatal, "unknown error while setting up workflow in {}.", processName);
  } else {
    LOGP(fatal, "error while setting up workflow in {}: {}", processName, s);
  }
}

[[maybe_unused]] AlgorithmSpec dryRun(DeviceSpec const& spec)
{
  return AlgorithmSpec{adaptStateless(
    [&routes = spec.outputs](DataAllocator& outputs) {
      LOG(info) << "Dry run enforced. Creating dummy messages to simulate computation happended";
      for (auto& route : routes) {
        auto concrete = DataSpecUtils::asConcreteDataMatcher(route.matcher);
        outputs.make<int>(Output{concrete.origin, concrete.description, concrete.subSpec}, 2);
      }
    })};
}

void doDefaultWorkflowTerminationHook()
{
  // LOG(info) << "Process " << getpid() << " is exiting.";
}

int doChild(int argc, char** argv, ServiceRegistry& serviceRegistry,
            RunningWorkflowInfo const& runningWorkflow,
            RunningDeviceRef ref,
            ProcessingPolicies processingPolicies,
            std::string const& defaultDriverClient,
            uv_loop_t* loop)
{
  fair::Logger::SetConsoleColor(false);
  fair::Logger::OnFatal([]() { throw runtime_error("Fatal error"); });
  DeviceSpec const& spec = runningWorkflow.devices[ref.index];
  LOG(info) << "Spawing new device " << spec.id << " in process with pid " << getpid();

  fair::mq::DeviceRunner runner{argc, argv};

  // Populate options from the command line. Notice that only the options
  // declared in the workflow definition are allowed.
  runner.AddHook<fair::mq::hooks::SetCustomCmdLineOptions>([&spec, defaultDriverClient](fair::mq::DeviceRunner& r) {
    std::string defaultExitTransitionTimeout = "0";
    std::string defaultInfologgerMode = "";
    if (getenv("DDS_SESSION_ID") != nullptr) {
      defaultExitTransitionTimeout = "20";
      defaultInfologgerMode = "infoLoggerD";
    } else if (getenv("OCC_CONTROL_PORT") != nullptr) {
      defaultExitTransitionTimeout = "20";
    }
    boost::program_options::options_description optsDesc;
    ConfigParamsHelper::populateBoostProgramOptions(optsDesc, spec.options, gHiddenDeviceOptions);
    optsDesc.add_options()("monitoring-backend", bpo::value<std::string>()->default_value("default"), "monitoring backend info")                                                           //
      ("driver-client-backend", bpo::value<std::string>()->default_value(defaultDriverClient), "backend for device -> driver communicataon: stdout://: use stdout, ws://: use websockets") //
      ("infologger-severity", bpo::value<std::string>()->default_value(""), "minimum FairLogger severity to send to InfoLogger")                                                           //
      ("dpl-tracing-flags", bpo::value<std::string>()->default_value(""), "pipe `|` separate list of events to be traced")                                                                 //
      ("expected-region-callbacks", bpo::value<std::string>()->default_value("0"), "how many region callbacks we are expecting")                                                           //
      ("exit-transition-timeout", bpo::value<std::string>()->default_value(defaultExitTransitionTimeout), "how many second to wait before switching from RUN to READY")                    //
      ("timeframes-rate-limit", bpo::value<std::string>()->default_value("0"), "how many timeframe can be in fly at the same moment (0 disables)")                                         //
      ("configuration,cfg", bpo::value<std::string>()->default_value("command-line"), "configuration backend")                                                                             //
      ("infologger-mode", bpo::value<std::string>()->default_value(defaultInfologgerMode), "O2_INFOLOGGER_MODE override");
    r.fConfig.AddToCmdLineOptions(optsDesc, true);
  });

  // This is to control lifetime. All these services get destroyed
  // when the runner is done.
  std::unique_ptr<SimpleRawDeviceService> simpleRawDeviceService;
  std::unique_ptr<DeviceState> deviceState;
  std::unique_ptr<ComputingQuotaEvaluator> quotaEvaluator;
  std::unique_ptr<FairMQDeviceProxy> deviceProxy;
  std::unique_ptr<DeviceContext> deviceContext;

  auto afterConfigParsingCallback = [&simpleRawDeviceService,
                                     &runningWorkflow,
                                     ref,
                                     &spec,
                                     &quotaEvaluator,
                                     &serviceRegistry,
                                     &deviceState,
                                     &deviceProxy,
                                     &processingPolicies,
                                     &deviceContext,
                                     &loop](fair::mq::DeviceRunner& r) {
    ServiceRegistryRef serviceRef = {serviceRegistry};
    simpleRawDeviceService = std::make_unique<SimpleRawDeviceService>(nullptr, spec);
    serviceRef.registerService(ServiceRegistryHelpers::handleForService<RawDeviceService>(simpleRawDeviceService.get()));

    deviceState = std::make_unique<DeviceState>();
    deviceState->loop = loop;
    deviceState->tracingFlags = DeviceStateHelpers::parseTracingFlags(r.fConfig.GetPropertyAsString("dpl-tracing-flags"));
    serviceRef.registerService(ServiceRegistryHelpers::handleForService<DeviceState>(deviceState.get()));

    quotaEvaluator = std::make_unique<ComputingQuotaEvaluator>(uv_now(loop));
    serviceRef.registerService(ServiceRegistryHelpers::handleForService<ComputingQuotaEvaluator>(quotaEvaluator.get()));

    deviceContext = std::make_unique<DeviceContext>();
    serviceRef.registerService(ServiceRegistryHelpers::handleForService<DeviceSpec const>(&spec));
    serviceRef.registerService(ServiceRegistryHelpers::handleForService<RunningWorkflowInfo const>(&runningWorkflow));
    serviceRef.registerService(ServiceRegistryHelpers::handleForService<DeviceContext>(deviceContext.get()));

    // The decltype stuff is to be able to compile with both new and old
    // FairMQ API (one which uses a shared_ptr, the other one a unique_ptr.
    decltype(r.fDevice) device;
    device = make_matching<decltype(device), DataProcessingDevice>(ref, serviceRegistry, processingPolicies);

    serviceRef.get<RawDeviceService>().setDevice(device.get());
    r.fDevice = std::move(device);
    fair::Logger::SetConsoleColor(false);

    /// Create all the requested services and initialise them
    for (auto& service : spec.services) {
      LOG(debug) << "Declaring service " << service.name;
      serviceRegistry.declareService(service, *deviceState.get(), r.fConfig);
    }
    if (ResourcesMonitoringHelper::isResourcesMonitoringEnabled(spec.resourceMonitoringInterval)) {
      serviceRef.get<Monitoring>().enableProcessMonitoring(spec.resourceMonitoringInterval, {PmMeasurement::Cpu, PmMeasurement::Mem, PmMeasurement::Smaps});
    }
  };

  runner.AddHook<fair::mq::hooks::InstantiateDevice>(afterConfigParsingCallback);

  auto result = runner.Run();
  serviceRegistry.preExitCallbacks();
  return result;
}

struct WorkflowInfo {
  std::string executable;
  std::vector<std::string> args;
  std::vector<ConfigParamSpec> options;
};

void gui_callback(uv_timer_s* ctx)
{
  auto* gui = reinterpret_cast<GuiCallbackContext*>(ctx->data);
  if (gui->plugin == nullptr) {
    return;
  }
  void* draw_data = nullptr;

  uint64_t frameStart = uv_hrtime();
  uint64_t frameLatency = frameStart - gui->frameLast;

  // if less than 15ms have passed reuse old frame
  if (frameLatency / 1000000 > 15) {
    if (!gui->plugin->pollGUIPreRender(gui->window, (float)frameLatency / 1000000000.0f)) {
      *(gui->guiQuitRequested) = true;
      return;
    }
    draw_data = gui->plugin->pollGUIRender(gui->callback);
    gui->plugin->pollGUIPostRender(gui->window, draw_data);
  } else {
    draw_data = gui->lastFrame;
  }

  if (frameLatency / 1000000 > 15) {
    uint64_t frameEnd = uv_hrtime();
    *(gui->frameCost) = (frameEnd - frameStart) / 1000000.f;
    *(gui->frameLatency) = frameLatency / 1000000.f;
    gui->frameLast = frameStart;
  }
}

/// Force single stepping of the children
void single_step_callback(uv_timer_s* ctx)
{
  auto* infos = reinterpret_cast<DeviceInfos*>(ctx->data);
  killChildren(*infos, SIGUSR1);
}

void force_exit_callback(uv_timer_s* ctx)
{
  auto* infos = reinterpret_cast<DeviceInfos*>(ctx->data);
  killChildren(*infos, SIGKILL);
}

std::vector<std::regex> getDumpableMetrics()
{
  auto performanceMetrics = o2::monitoring::ProcessMonitor::getAvailableMetricsNames();
  auto dumpableMetrics = std::vector<std::regex>{};
  for (const auto& metric : performanceMetrics) {
    dumpableMetrics.emplace_back(metric);
  }
  dumpableMetrics.emplace_back("^arrow-bytes-delta$");
  dumpableMetrics.emplace_back("^aod-bytes-read-uncompressed$");
  dumpableMetrics.emplace_back("^aod-bytes-read-compressed$");
  dumpableMetrics.emplace_back("^aod-file-read-info$");
  dumpableMetrics.emplace_back("^table-bytes-.*");
  dumpableMetrics.emplace_back("^total-timeframes.*");
  return dumpableMetrics;
}

void dumpMetricsCallback(uv_timer_t* handle)
{
  auto* context = (DriverServerContext*)handle->data;

  static auto performanceMetrics = getDumpableMetrics();
  ResourcesMonitoringHelper::dumpMetricsToJSON(*(context->metrics),
                                               context->driver->metrics, *(context->specs), performanceMetrics);
}

// This is the handler for the parent inner loop.
int runStateMachine(DataProcessorSpecs const& workflow,
                    WorkflowInfo const& workflowInfo,
                    DataProcessorInfos const& previousDataProcessorInfos,
                    CommandInfo const& commandInfo,
                    DriverControl& driverControl,
                    DriverInfo& driverInfo,
                    std::vector<DeviceMetricsInfo>& metricsInfos,
                    boost::program_options::variables_map& varmap,
                    std::vector<ServiceSpec>& driverServices,
                    std::string frameworkId)
{
  RunningWorkflowInfo runningWorkflow{
    .uniqueWorkflowId = driverInfo.uniqueWorkflowId,
    .shmSegmentId = (int16_t)atoi(varmap["shm-segment-id"].as<std::string>().c_str())};
  DeviceInfos infos;
  DeviceControls controls;
  auto* devicesManager = new DevicesManager{.controls = controls, .infos = infos, .specs = runningWorkflow.devices, .messages = {}};
  DeviceExecutions deviceExecutions;
  DataProcessorInfos dataProcessorInfos = previousDataProcessorInfos;

  std::vector<uv_poll_t*> pollHandles;
  std::vector<DeviceStdioContext> childFds;

  std::vector<ComputingResource> resources;

  if (driverInfo.resources != "") {
    resources = ComputingResourceHelpers::parseResources(driverInfo.resources);
  } else {
    resources = {ComputingResourceHelpers::getLocalhostResource()};
  }

  auto resourceManager = std::make_unique<SimpleResourceManager>(resources);

  DebugGUI* debugGUI = nullptr;
  void* window = nullptr;
  decltype(debugGUI->getGUIDebugger(infos, runningWorkflow.devices, dataProcessorInfos, metricsInfos, driverInfo, controls, driverControl)) debugGUICallback;

  // An empty frameworkId means this is the driver, so we initialise the GUI
  auto initDebugGUI = []() -> DebugGUI* {
    uv_lib_t supportLib;
    int result = 0;
#ifdef __APPLE__
    result = uv_dlopen("libO2FrameworkGUISupport.dylib", &supportLib);
#else
    result = uv_dlopen("libO2FrameworkGUISupport.so", &supportLib);
#endif
    if (result == -1) {
      LOG(error) << uv_dlerror(&supportLib);
      return nullptr;
    }
    DPLPluginHandle* (*dpl_plugin_callback)(DPLPluginHandle*);

    result = uv_dlsym(&supportLib, "dpl_plugin_callback", (void**)&dpl_plugin_callback);
    if (result == -1) {
      LOG(error) << uv_dlerror(&supportLib);
      return nullptr;
    }
    DPLPluginHandle* pluginInstance = dpl_plugin_callback(nullptr);
    return PluginManager::getByName<DebugGUI>(pluginInstance, "ImGUIDebugGUI");
  };

  // We initialise this in the driver, because different drivers might have
  // different versions of the service
  ServiceRegistry serviceRegistry;

  if ((driverInfo.batch == false || getenv("DPL_DRIVER_REMOTE_GUI") != nullptr) && frameworkId.empty()) {
    debugGUI = initDebugGUI();
    if (debugGUI) {
      if (driverInfo.batch == false) {
        window = debugGUI->initGUI("O2 Framework debug GUI", serviceRegistry);
      } else {
        window = debugGUI->initGUI(nullptr, serviceRegistry);
      }
    }
  } else if (getenv("DPL_DEVICE_REMOTE_GUI") && !frameworkId.empty()) {
    debugGUI = initDebugGUI();
    // We never run the GUI on desktop for devices. All
    // you can do is to connect to the remote version.
    // this is done to avoid having a proliferation of
    // GUIs popping up when the variable is set globally.
    // FIXME: maybe this is not what we want, but it should
    //        be ok for now.
    if (debugGUI) {
      window = debugGUI->initGUI(nullptr, serviceRegistry);
    }
  }
  if (driverInfo.batch == false && window == nullptr && frameworkId.empty()) {
    LOG(warn) << "Could not create GUI. Switching to batch mode. Do you have GLFW on your system?";
    driverInfo.batch = true;
  }
  bool guiQuitRequested = false;
  bool hasError = false;

  // FIXME: I should really have some way of exiting the
  // parent..
  DriverState current;
  DriverState previous;

  uv_loop_t* loop = uv_loop_new();

  uv_timer_t* gui_timer = nullptr;

  if (!driverInfo.batch) {
    gui_timer = (uv_timer_t*)malloc(sizeof(uv_timer_t));
    uv_timer_init(loop, gui_timer);
  }

  std::vector<ServiceMetricHandling> metricProcessingCallbacks;
  std::vector<ServicePreSchedule> preScheduleCallbacks;
  std::vector<ServicePostSchedule> postScheduleCallbacks;
  std::vector<ServiceDriverInit> driverInitCallbacks;
  for (auto& service : driverServices) {
    if (service.driverStartup == nullptr) {
      continue;
    }
    service.driverStartup(serviceRegistry, varmap);
  }

  ServiceRegistryRef ref{serviceRegistry};
  ref.registerService(ServiceRegistryHelpers::handleForService<DevicesManager>(devicesManager));

  GuiCallbackContext guiContext;
  guiContext.plugin = debugGUI;
  guiContext.frameLast = uv_hrtime();
  guiContext.frameLatency = &driverInfo.frameLatency;
  guiContext.frameCost = &driverInfo.frameCost;
  guiContext.guiQuitRequested = &guiQuitRequested;

  // This is to make sure we can process metrics, commands, configuration
  // changes coming from websocket (or even via any standard uv_stream_t, I guess).
  DriverServerContext serverContext{
    .registry = {serviceRegistry},
    .loop = loop,
    .controls = &controls,
    .infos = &infos,
    .specs = &runningWorkflow.devices,
    .metrics = &metricsInfos,
    .metricProcessingCallbacks = &metricProcessingCallbacks,
    .driver = &driverInfo,
    .gui = &guiContext,
    .isDriver = frameworkId.empty()};

  uv_tcp_t serverHandle;
  serverHandle.data = &serverContext;
  uv_tcp_init(loop, &serverHandle);

  driverInfo.port = 8080 + (getpid() % 30000);

  if (getenv("DPL_REMOTE_GUI_PORT")) {
    try {
      driverInfo.port = stoi(std::string(getenv("DPL_REMOTE_GUI_PORT")));
    } catch (std::invalid_argument) {
      LOG(error) << "DPL_REMOTE_GUI_PORT not a valid integer";
    } catch (std::out_of_range) {
      LOG(error) << "DPL_REMOTE_GUI_PORT out of range (integer)";
    }
    if (driverInfo.port < 1024 || driverInfo.port > 65535) {
      LOG(error) << "DPL_REMOTE_GUI_PORT out of range (1024-65535)";
    }
  }

  int result = 0;
  struct sockaddr_in* serverAddr = nullptr;

  // Do not offer websocket endpoint for devices
  // FIXME: this was blocking david's workflows. For now
  //        there is no point in any case to have devices
  //        offering a web based API, but it might make sense in
  //        the future to inspect them via some web based interface.
  if (frameworkId.empty()) {
    do {
      free(serverAddr);
      if (driverInfo.port > 64000) {
        throw runtime_error_f("Unable to find a free port for the driver. Last attempt returned %d", result);
      }
      serverAddr = (sockaddr_in*)malloc(sizeof(sockaddr_in));
      uv_ip4_addr("0.0.0.0", driverInfo.port, serverAddr);
      auto bindResult = uv_tcp_bind(&serverHandle, (const struct sockaddr*)serverAddr, 0);
      if (bindResult != 0) {
        driverInfo.port++;
        usleep(1000);
        continue;
      }
      result = uv_listen((uv_stream_t*)&serverHandle, 100, ws_connect_callback);
      if (result != 0) {
        driverInfo.port++;
        usleep(1000);
        continue;
      }
    } while (result != 0);
  } else if (getenv("DPL_DEVICE_REMOTE_GUI") && !frameworkId.empty()) {
    do {
      free(serverAddr);
      if (driverInfo.port > 64000) {
        throw runtime_error_f("Unable to find a free port for the driver. Last attempt returned %d", result);
      }
      serverAddr = (sockaddr_in*)malloc(sizeof(sockaddr_in));
      uv_ip4_addr("0.0.0.0", driverInfo.port, serverAddr);
      auto bindResult = uv_tcp_bind(&serverHandle, (const struct sockaddr*)serverAddr, 0);
      if (bindResult != 0) {
        driverInfo.port++;
        usleep(1000);
        continue;
      }
      result = uv_listen((uv_stream_t*)&serverHandle, 100, ws_connect_callback);
      if (result != 0) {
        driverInfo.port++;
        usleep(1000);
        continue;
      }
      LOG(info) << "Device GUI port: " << driverInfo.port << " " << frameworkId;
    } while (result != 0);
  }

  uv_timer_t force_step_timer;
  uv_timer_init(loop, &force_step_timer);
  uv_timer_t force_exit_timer;
  uv_timer_init(loop, &force_exit_timer);

  bool guiDeployedOnce = false;
  bool once = false;

  uv_timer_t metricDumpTimer;
  metricDumpTimer.data = &serverContext;

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
        LOG(info) << "Timout ellapsed. Requesting to quit.";
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
        LOGP(info, "Initialising O2 Data Processing Layer. Driver PID: {}.", getpid());
        LOGP(info, "Driver listening on port: {}", driverInfo.port);

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
        for (auto& callback : driverInitCallbacks) {
          callback(serviceRegistry, varmap);
        }
        driverInfo.states.push_back(DriverState::RUNNING);
        //        driverInfo.states.push_back(DriverState::REDEPLOY_GUI);
        LOG(info) << "O2 Data Processing Layer initialised. We brake for nobody.";
#ifdef NDEBUG
        LOGF(info, "Optimised build. O2DEBUG / LOG(debug) / LOGF(debug) / assert statement will not be shown.");
#endif
        break;
      case DriverState::IMPORT_CURRENT_WORKFLOW:
        // This state is needed to fill the metadata structure
        // which contains how to run the current workflow
        dataProcessorInfos = previousDataProcessorInfos;
        for (auto const& device : runningWorkflow.devices) {
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
          if (driverInfo.batch == true && varmap["dds"].as<std::string>().empty() && !varmap["dump-workflow"].as<bool>() && workflowState == WorkflowParsingState::Empty) {
            LOGP(error, "Empty workflow provided while running in batch mode.");
            return 1;
          }

          /// extract and apply process switches
          /// prune device inputs
          auto altered_workflow = workflow;

          auto confNameFromParam = [](std::string const& paramName) {
            std::regex name_regex(R"(^control:([\w-]+)\/(\w+))");
            auto match = std::sregex_token_iterator(paramName.begin(), paramName.end(), name_regex, 0);
            if (match == std::sregex_token_iterator()) {
              throw runtime_error_f("Malformed process control spec: %s", paramName.c_str());
            }
            std::string task = std::sregex_token_iterator(paramName.begin(), paramName.end(), name_regex, 1)->str();
            std::string conf = std::sregex_token_iterator(paramName.begin(), paramName.end(), name_regex, 2)->str();
            return std::pair{task, conf};
          };
          bool altered = false;
          for (auto& device : altered_workflow) {
            // ignore internal devices
            if (device.name.find("internal") != std::string::npos) {
              continue;
            }
            // ignore devices with no inputs
            if (device.inputs.empty() == true) {
              continue;
            }
            // ignore devices with no metadata in inputs
            auto hasMetadata = std::any_of(device.inputs.begin(), device.inputs.end(), [](InputSpec const& spec) {
              return spec.metadata.empty() == false;
            });
            if (!hasMetadata) {
              continue;
            }
            // ignore devices with no control options
            auto hasControls = std::any_of(device.inputs.begin(), device.inputs.end(), [](InputSpec const& spec) {
              return std::any_of(spec.metadata.begin(), spec.metadata.end(), [](ConfigParamSpec const& param) {
                return param.type == VariantType::Bool && param.name.find("control:") != std::string::npos;
              });
            });
            if (!hasControls) {
              continue;
            }

            LOGP(debug, "Adjusting device {}", device.name.c_str());

            auto configStore = DeviceConfigurationHelpers::getConfiguration(serviceRegistry, device.name.c_str(), device.options);
            if (configStore != nullptr) {
              auto reg = std::make_unique<ConfigParamRegistry>(std::move(configStore));
              for (auto& input : device.inputs) {
                for (auto& param : input.metadata) {
                  if (param.type == VariantType::Bool && param.name.find("control:") != std::string::npos) {
                    if (param.name != "control:default" && param.name != "control:spawn" && param.name != "control:build") {
                      auto confName = confNameFromParam(param.name).second;
                      param.defaultValue = reg->get<bool>(confName.c_str());
                    }
                  }
                }
              }
            }
            /// FIXME: use commandline arguments as alternative
            LOGP(debug, "Original inputs: ");
            for (auto& input : device.inputs) {
              LOGP(debug, "-> {}", input.binding);
            }
            auto end = device.inputs.end();
            auto new_end = std::remove_if(device.inputs.begin(), device.inputs.end(), [](InputSpec& input) {
              auto requested = false;
              auto hasControls = false;
              for (auto& param : input.metadata) {
                if (param.type != VariantType::Bool) {
                  continue;
                }
                if (param.name.find("control:") != std::string::npos) {
                  hasControls = true;
                  if (param.defaultValue.get<bool>() == true) {
                    requested = true;
                    break;
                  }
                }
              }
              if (hasControls) {
                return !requested;
              }
              return false;
            });
            device.inputs.erase(new_end, end);
            LOGP(debug, "Adjusted inputs: ");
            for (auto& input : device.inputs) {
              LOGP(debug, "-> {}", input.binding);
            }
            altered = true;
          }
          WorkflowHelpers::adjustTopology(altered_workflow, *driverInfo.configContext);
          if (altered) {
            WorkflowSpecNode node{altered_workflow};
            for (auto& service : driverServices) {
              if (service.adjustTopology == nullptr) {
                continue;
              }
              service.adjustTopology(node, *driverInfo.configContext);
            }
          }

          // These allow services customization via an environment variable
          OverrideServiceSpecs overrides = ServiceSpecHelpers::parseOverrides(getenv("DPL_OVERRIDE_SERVICES"));
          DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(altered_workflow,
                                                            driverInfo.channelPolicies,
                                                            driverInfo.completionPolicies,
                                                            driverInfo.dispatchPolicies,
                                                            driverInfo.resourcePolicies,
                                                            driverInfo.callbacksPolicies,
                                                            driverInfo.sendingPolicies,
                                                            runningWorkflow.devices,
                                                            *resourceManager,
                                                            driverInfo.uniqueWorkflowId,
                                                            *driverInfo.configContext,
                                                            !varmap["no-IPC"].as<bool>(),
                                                            driverInfo.resourcesMonitoringInterval,
                                                            varmap["channel-prefix"].as<std::string>(),
                                                            overrides);
          metricProcessingCallbacks.clear();
          for (auto& device : runningWorkflow.devices) {
            for (auto& service : device.services) {
              if (service.metricHandling) {
                metricProcessingCallbacks.push_back(service.metricHandling);
              }
            }
          }
          preScheduleCallbacks.clear();
          for (auto& device : runningWorkflow.devices) {
            for (auto& service : device.services) {
              if (service.preSchedule) {
                preScheduleCallbacks.push_back(service.preSchedule);
              }
            }
          }
          postScheduleCallbacks.clear();
          for (auto& device : runningWorkflow.devices) {
            for (auto& service : device.services) {
              if (service.postSchedule) {
                postScheduleCallbacks.push_back(service.postSchedule);
              }
            }
          }
          driverInitCallbacks.clear();
          for (auto& device : runningWorkflow.devices) {
            for (auto& service : device.services) {
              if (service.driverInit) {
                driverInitCallbacks.push_back(service.driverInit);
              }
            }
          }

          // This should expand nodes so that we can build a consistent DAG.
        } catch (std::runtime_error& e) {
          LOGP(error, "invalid workflow in {}: {}", driverInfo.argv[0], e.what());
          return 1;
        } catch (o2::framework::RuntimeErrorRef ref) {
          auto& err = o2::framework::error_from_ref(ref);
#ifdef DPL_ENABLE_BACKTRACE
          demangled_backtrace_symbols(err.backtrace, err.maxBacktrace, STDERR_FILENO);
#endif
          LOGP(error, "invalid workflow in {}: {}", driverInfo.argv[0], err.what);
          return 1;
        } catch (...) {
          LOGP(error, "invalid workflow in {}: Unknown error while materialising workflow", driverInfo.argv[0]);
          return 1;
        }
        break;
      case DriverState::DO_CHILD:
        // We do not start the process if by default we are stopped.
        if (driverControl.defaultStopped) {
          kill(getpid(), SIGSTOP);
        }
        for (size_t di = 0; di < runningWorkflow.devices.size(); di++) {
          RunningDeviceRef ref{di};
          if (runningWorkflow.devices[di].id == frameworkId) {
            return doChild(driverInfo.argc, driverInfo.argv,
                           serviceRegistry,
                           runningWorkflow, ref,
                           driverInfo.processingPolicies,
                           driverInfo.defaultDriverClient,
                           loop);
          }
        }
        {
          std::ostringstream ss;
          for (auto& processor : workflow) {
            ss << " - " << processor.name << "\n";
          }
          for (auto& spec : runningWorkflow.devices) {
            ss << " - " << spec.name << "(" << spec.id << ")"
               << "\n";
          }
          driverInfo.lastError = fmt::format(
            "Unable to find component with id {}."
            " Available options:\n{}",
            frameworkId, ss.str());
          driverInfo.states.push_back(DriverState::QUIT_REQUESTED);
        }
        break;
      case DriverState::REDEPLOY_GUI:
        // The callback for the GUI needs to be recalculated every time
        // the deployed configuration changes, e.g. a new device
        // has been added to the topology.
        // We need to recreate the GUI callback every time we reschedule
        // because getGUIDebugger actually recreates the GUI state.
        // Notice also that we need the actual gui_timer only for the
        // case the GUI runs in interactive mode, however we deploy the
        // GUI in both interactive and non-interactive mode, if the
        // DPL_DRIVER_REMOTE_GUI environment variable is set.
        if (!driverInfo.batch || getenv("DPL_DRIVER_REMOTE_GUI")) {
          if (gui_timer) {
            uv_timer_stop(gui_timer);
          }

          auto callback = debugGUI->getGUIDebugger(infos, runningWorkflow.devices, dataProcessorInfos, metricsInfos, driverInfo, controls, driverControl);
          guiContext.callback = [&serviceRegistry, &driverServices, &debugGUI, &infos, &runningWorkflow, &dataProcessorInfos, &metricsInfos, &driverInfo, &controls, &driverControl, callback]() {
            callback();
            for (auto& service : driverServices) {
              if (service.postRenderGUI) {
                service.postRenderGUI(serviceRegistry);
              }
            }
          };
          guiContext.window = window;

          if (gui_timer) {
            gui_timer->data = &guiContext;
            uv_timer_start(gui_timer, gui_callback, 0, 20);
          }
          guiDeployedOnce = true;
        }
        break;
      case DriverState::MERGE_CONFIGS: {
        try {
          controls.resize(runningWorkflow.devices.size());
          /// Set the default value for tracingFlags of each control
          /// to the command line value --dpl-tracing-flags
          if (varmap.count("dpl-tracing-flags")) {
            for (auto& control : controls) {
              auto tracingFlags = DeviceStateHelpers::parseTracingFlags(varmap["dpl-tracing-flags"].as<std::string>());
              control.tracingFlags = tracingFlags;
            }
          }
          deviceExecutions.resize(runningWorkflow.devices.size());

          // Options which should be uniform across all
          // the subworkflow invokations.
          const auto uniformOptions = {
            "--aod-file",
            "--aod-memory-rate-limit",
            "--aod-writer-json",
            "--aod-writer-ntfmerge",
            "--aod-writer-resdir",
            "--aod-writer-resfile",
            "--aod-writer-resmode",
            "--aod-writer-maxfilesize",
            "--aod-writer-keep",
            "--aod-parent-access-level",
            "--aod-parent-base-path-replacement",
            "--driver-client-backend",
            "--fairmq-ipc-prefix",
            "--readers",
            "--resources-monitoring",
            "--resources-monitoring-dump-interval",
            "--time-limit",
          };

          for (auto& option : uniformOptions) {
            DeviceSpecHelpers::reworkHomogeneousOption(dataProcessorInfos, option, nullptr);
          }

          DeviceSpecHelpers::reworkShmSegmentSize(dataProcessorInfos);
          DeviceSpecHelpers::prepareArguments(driverControl.defaultQuiet,
                                              driverControl.defaultStopped,
                                              driverInfo.processingPolicies.termination == TerminationPolicy::WAIT,
                                              driverInfo.port,
                                              dataProcessorInfos,
                                              runningWorkflow.devices,
                                              deviceExecutions,
                                              controls,
                                              driverInfo.uniqueWorkflowId);
        } catch (o2::framework::RuntimeErrorRef& ref) {
          auto& err = o2::framework::error_from_ref(ref);
          LOGP(error, "unable to merge configurations in {}: {}", driverInfo.argv[0], err.what);
#ifdef DPL_ENABLE_BACKTRACE
          std::cerr << "\nStacktrace follows:\n\n";
          demangled_backtrace_symbols(err.backtrace, err.maxBacktrace, STDERR_FILENO);
#endif
          return 1;
        }
      } break;
      case DriverState::SCHEDULE: {
        // FIXME: for the moment modifying the topology means we rebuild completely
        //        all the devices and we restart them. This is also what DDS does at
        //        a larger scale. In principle one could try to do a delta and only
        //        restart the data processors which need to be restarted.
        LOG(info) << "Redeployment of configuration asked.";
        std::ostringstream forwardedStdin;
        WorkflowSerializationHelpers::dump(forwardedStdin, workflow, dataProcessorInfos, commandInfo);
        infos.reserve(runningWorkflow.devices.size());

        // This is guaranteed to be a single CPU.
        unsigned parentCPU = -1;
        unsigned parentNode = -1;
#if defined(__linux__) && __has_include(<sched.h>)
        parentCPU = sched_getcpu();
#elif __has_include(<linux/getcpu.h>)
        getcpu(&parentCPU, &parentNode, nullptr);
#elif __has_include(<cpuid.h>) && (__x86_64__ || __i386__)
        // FIXME: this is a last resort as it is apparently buggy
        //        on some Intel CPUs.
        GETCPU(parentCPU);
#endif
        for (auto& callback : preScheduleCallbacks) {
          callback(serviceRegistry, varmap);
        }
        childFds.resize(runningWorkflow.devices.size());
        for (int di = 0; di < (int)runningWorkflow.devices.size(); ++di) {
          auto& context = childFds[di];
          createPipes(context.childstdin);
          createPipes(context.childstdout);
          if (runningWorkflow.devices[di].resource.hostname != driverInfo.deployHostname) {
            spawnRemoteDevice(forwardedStdin.str(),
                              runningWorkflow.devices[di], controls[di], deviceExecutions[di], infos);
          } else {
            DeviceRef ref{di};
            spawnDevice(ref,
                        runningWorkflow.devices, driverInfo,
                        controls, deviceExecutions, infos,
                        serviceRegistry, varmap,
                        childFds, parentCPU, parentNode);
          }
        }
        handleSignals();
        handleChildrenStdio(loop, forwardedStdin.str(), infos, childFds, pollHandles);
        for (auto& callback : postScheduleCallbacks) {
          callback(serviceRegistry, varmap);
        }
        assert(infos.empty() == false);

        // In case resource monitoring is requested, we dump metrics to disk
        // every 3 minutes.
        if (driverInfo.resourcesMonitoringDumpInterval && ResourcesMonitoringHelper::isResourcesMonitoringEnabled(driverInfo.resourcesMonitoringInterval)) {
          uv_timer_init(loop, &metricDumpTimer);
          uv_timer_start(&metricDumpTimer, dumpMetricsCallback,
                         driverInfo.resourcesMonitoringDumpInterval * 1000,
                         driverInfo.resourcesMonitoringDumpInterval * 1000);
        }
        LOG(info) << "Redeployment of configuration done.";
      } break;
      case DriverState::RUNNING:
        // Run any pending libUV event loop, block if
        // any, so that we do not consume CPU time when the driver is
        // idle.
        devicesManager->flush();
        uv_run(loop, once ? UV_RUN_ONCE : UV_RUN_NOWAIT);
        once = true;
        // Calculate what we should do next and eventually
        // show the GUI
        if (guiQuitRequested ||
            (driverInfo.processingPolicies.termination == TerminationPolicy::QUIT && (checkIfCanExit(infos) == true))) {
          // Something requested to quit. This can be a user
          // interaction with the GUI or (if --completion-policy=quit)
          // it could mean that the workflow does not have anything else to do.
          // Let's update the GUI one more time and then EXIT.
          LOG(info) << "Quitting";
          driverInfo.states.push_back(DriverState::QUIT_REQUESTED);
        } else if (infos.size() != runningWorkflow.devices.size()) {
          // If the number of devices is different from
          // the DeviceInfos it means the speicification
          // does not match what is running, so we need to do
          // further scheduling.
          driverInfo.states.push_back(DriverState::RUNNING);
          driverInfo.states.push_back(DriverState::REDEPLOY_GUI);
          driverInfo.states.push_back(DriverState::SCHEDULE);
          driverInfo.states.push_back(DriverState::MERGE_CONFIGS);
        } else if (runningWorkflow.devices.empty() && driverInfo.batch == true) {
          LOG(info) << "No device resulting from the workflow. Quitting.";
          // If there are no deviceSpecs, we exit.
          driverInfo.states.push_back(DriverState::EXIT);
        } else if (runningWorkflow.devices.empty() && driverInfo.batch == false && !guiDeployedOnce) {
          // In case of an empty workflow, we need to deploy the GUI at least once.
          driverInfo.states.push_back(DriverState::RUNNING);
          driverInfo.states.push_back(DriverState::REDEPLOY_GUI);
        } else {
          driverInfo.states.push_back(DriverState::RUNNING);
        }
        {
          auto outputProcessing = processChildrenOutput(driverInfo, infos, runningWorkflow.devices, controls, metricsInfos);
          if (outputProcessing.didProcessMetric) {
            size_t timestamp = uv_now(loop);
            for (auto& callback : metricProcessingCallbacks) {
              callback(serviceRegistry, metricsInfos, runningWorkflow.devices, infos, driverInfo.metrics, timestamp);
            }
            for (auto& metricsInfo : metricsInfos) {
              std::fill(metricsInfo.changed.begin(), metricsInfo.changed.end(), false);
            }
          }
        }
        break;
      case DriverState::QUIT_REQUESTED:
        LOG(info) << "QUIT_REQUESTED";
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
        uv_run(loop, once ? UV_RUN_ONCE : UV_RUN_NOWAIT);
        once = true;
        // I allow queueing of more sigchld only when
        // I process the previous call
        if (forceful_exit == true) {
          static bool forcefulExitMessage = true;
          if (forcefulExitMessage) {
            LOG(info) << "Forceful exit requested.";
            forcefulExitMessage = false;
          }
          killChildren(infos, SIGCONT);
          killChildren(infos, SIGKILL);
        }
        sigchld_requested = false;
        driverInfo.sigchldRequested = false;
        auto outputProcessing = processChildrenOutput(driverInfo, infos, runningWorkflow.devices, controls, metricsInfos);
        if (outputProcessing.didProcessMetric) {
          size_t timestamp = uv_now(loop);
          for (auto& callback : metricProcessingCallbacks) {
            callback(serviceRegistry, metricsInfos, runningWorkflow.devices, infos, driverInfo.metrics, timestamp);
          }
        }
        hasError = processSigChild(infos, runningWorkflow.devices);
        bool allChildrenGone = areAllChildrenGone(infos);
        bool canExit = checkIfCanExit(infos);
        bool supposedToQuit = (guiQuitRequested || canExit || graceful_exit);

        if (allChildrenGone && (supposedToQuit || driverInfo.processingPolicies.termination == TerminationPolicy::QUIT)) {
          // We move to the exit, regardless of where we were
          driverInfo.states.resize(0);
          driverInfo.states.push_back(DriverState::EXIT);
        } else if (hasError && driverInfo.processingPolicies.error == TerminationPolicy::QUIT && !supposedToQuit) {
          graceful_exit = 1;
          force_exit_timer.data = &infos;
          static bool forceful_timer_started = false;
          if (forceful_timer_started == false) {
            forceful_timer_started = true;
            uv_timer_start(&force_exit_timer, force_exit_callback, 15000, 3000);
          }
          driverInfo.states.push_back(DriverState::QUIT_REQUESTED);
        } else if (allChildrenGone == false && supposedToQuit) {
          driverInfo.states.push_back(DriverState::HANDLE_CHILDREN);
        } else {
        }
      } break;
      case DriverState::EXIT: {
        if (ResourcesMonitoringHelper::isResourcesMonitoringEnabled(driverInfo.resourcesMonitoringInterval)) {
          if (driverInfo.resourcesMonitoringDumpInterval) {
            uv_timer_stop(&metricDumpTimer);
          }
          LOG(info) << "Dumping performance metrics to performanceMetrics.json file";
          dumpMetricsCallback(&metricDumpTimer);
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
        assert(infos.size() == runningWorkflow.devices.size());
        for (size_t di = 0; di < infos.size(); ++di) {
          auto& info = infos[di];
          auto& spec = runningWorkflow.devices[di];
          finalConfig.put_child(spec.name, info.currentConfig);
        }
        LOG(info) << "Dumping used configuration in dpl-config.json";

        std::ofstream outDPLConfigFile("dpl-config.json", std::ios::out);
        if (outDPLConfigFile.is_open()) {
          boost::property_tree::write_json(outDPLConfigFile, finalConfig);
        } else {
          LOGP(warning, "Could not write out final configuration file. Read only run folder?");
        }
        if (driverInfo.noSHMCleanup) {
          LOGP(warning, "Not cleaning up shared memory.");
        } else {
          cleanupSHM(driverInfo.uniqueWorkflowId);
        }
        return calculateExitCode(driverInfo, runningWorkflow.devices, infos);
      }
      case DriverState::PERFORM_CALLBACKS:
        for (auto& callback : driverControl.callbacks) {
          callback(workflow, runningWorkflow.devices, deviceExecutions, dataProcessorInfos, commandInfo);
        }
        driverControl.callbacks.clear();
        break;
      default:
        LOG(error) << "Driver transitioned in an unknown state("
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

bool isInputConfig()
{
  struct stat s;
  int r = fstat(STDIN_FILENO, &s);
  // If stdin cannot be statted, we assume the shell is some sort of
  // non-interactive container thing
  if (r < 0) {
    return false;
  }
  // If stdin is a pipe or a file, we try to fetch configuration from there
  return ((s.st_mode & S_IFIFO) != 0 || (s.st_mode & S_IFREG) != 0);
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

void overrideLabels(ConfigContext& ctx, WorkflowSpec& workflow)
{
  struct LabelsSpec {
    std::string_view matcher;
    std::vector<std::string> labels;
  };
  std::vector<LabelsSpec> specs;

  auto labelsString = ctx.options().get<std::string>("labels");
  if (labelsString.empty()) {
    return;
  }
  std::string_view sv{labelsString};

  size_t specStart = 0;
  size_t specEnd = 0;
  constexpr char specDelim = ',';
  constexpr char labelDelim = ':';
  do {
    specEnd = sv.find(specDelim, specStart);
    auto token = sv.substr(specStart, specEnd == std::string_view::npos ? std::string_view::npos : specEnd - specStart);
    if (token.empty()) {
      throw std::runtime_error("bad labels definition. Syntax <processor>:<label>[:<label>][,<processor>:<label>[:<label>]");
    }

    size_t labelDelimPos = token.find(labelDelim);
    if (labelDelimPos == 0 || labelDelimPos == std::string_view::npos) {
      throw std::runtime_error("bad labels definition. Syntax <processor>:<label>[:<label>][,<processor>:<label>[:<label>]");
    }
    LabelsSpec spec{.matcher = token.substr(0, labelDelimPos), .labels = {}};

    size_t labelEnd = labelDelimPos + 1;
    do {
      size_t labelStart = labelDelimPos + 1;
      labelEnd = token.find(labelDelim, labelStart);
      auto label = labelEnd == std::string_view::npos ? token.substr(labelStart) : token.substr(labelStart, labelEnd - labelStart);
      if (label.empty()) {
        throw std::runtime_error("bad labels definition. Syntax <processor>:<label>[:<label>][,<processor>:<label>[:<label>]");
      }
      spec.labels.emplace_back(label);
      labelDelimPos = labelEnd;
    } while (labelEnd != std::string_view::npos);

    specs.push_back(spec);
    specStart = specEnd + 1;
  } while (specEnd != std::string_view::npos);

  if (labelsString.empty() == false && specs.empty() == true) {
    throw std::runtime_error("bad labels definition. Syntax <processor>:<label>[:<label>][,<processor>:<label>[:<label>]");
  }

  for (auto& spec : specs) {
    for (auto& processor : workflow) {
      if (processor.name == spec.matcher) {
        for (const auto& label : spec.labels) {
          if (std::find_if(processor.labels.begin(), processor.labels.end(),
                           [label](const auto& procLabel) { return procLabel.value == label; }) == processor.labels.end()) {
            processor.labels.push_back({label});
          }
        }
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
    control.callbacks = {[](WorkflowSpec const&,
                            DeviceSpecs const& specs,
                            DeviceExecutions const&,
                            DataProcessorInfos&,
                            CommandInfo const&) {
      GraphvizHelpers::dumpDeviceSpec2Graphviz(std::cout, specs);
    }};
    control.forcedTransitions = {
      DriverState::EXIT,                    //
      DriverState::PERFORM_CALLBACKS,       //
      DriverState::MERGE_CONFIGS,           //
      DriverState::IMPORT_CURRENT_WORKFLOW, //
      DriverState::MATERIALISE_WORKFLOW     //
    };
  } else if (!varmap["dds"].as<std::string>().empty()) {
    // Dump a DDS representation of what I will do.
    // Notice that compared to DDS we need to schedule things,
    // because DDS needs to be able to have actual Executions in
    // order to provide a correct configuration.
    control.callbacks = {[filename = varmap["dds"].as<std::string>(),
                          workflowSuffix = varmap["dds-workflow-suffix"]](WorkflowSpec const& workflow,
                                                                          DeviceSpecs const& specs,
                                                                          DeviceExecutions const& executions,
                                                                          DataProcessorInfos& dataProcessorInfos,
                                                                          CommandInfo const& commandInfo) {
      if (filename == "-") {
        DDSConfigHelpers::dumpDeviceSpec2DDS(std::cout, workflowSuffix.as<std::string>(), workflow, dataProcessorInfos, specs, executions, commandInfo);
      } else {
        std::ofstream out(filename);
        DDSConfigHelpers::dumpDeviceSpec2DDS(out, workflowSuffix.as<std::string>(), workflow, dataProcessorInfos, specs, executions, commandInfo);
      }
    }};
    control.forcedTransitions = {
      DriverState::EXIT,                    //
      DriverState::PERFORM_CALLBACKS,       //
      DriverState::MERGE_CONFIGS,           //
      DriverState::IMPORT_CURRENT_WORKFLOW, //
      DriverState::MATERIALISE_WORKFLOW     //
    };
  } else if (!varmap["o2-control"].as<std::string>().empty() or !varmap["mermaid"].as<std::string>().empty()) {
    // Dump the workflow in o2-control and/or mermaid format
    control.callbacks = {[filename = varmap["mermaid"].as<std::string>(),
                          workflowName = varmap["o2-control"].as<std::string>()](WorkflowSpec const&,
                                                                                 DeviceSpecs const& specs,
                                                                                 DeviceExecutions const& executions,
                                                                                 DataProcessorInfos&,
                                                                                 CommandInfo const& commandInfo) {
      if (!workflowName.empty()) {
        dumpDeviceSpec2O2Control(workflowName, specs, executions, commandInfo);
      }
      if (!filename.empty()) {
        if (filename == "-") {
          MermaidHelpers::dumpDeviceSpec2Mermaid(std::cout, specs);
        } else {
          std::ofstream output(filename);
          MermaidHelpers::dumpDeviceSpec2Mermaid(output, specs);
        }
      }
    }};
    control.forcedTransitions = {
      DriverState::EXIT,                    //
      DriverState::PERFORM_CALLBACKS,       //
      DriverState::MERGE_CONFIGS,           //
      DriverState::IMPORT_CURRENT_WORKFLOW, //
      DriverState::MATERIALISE_WORKFLOW     //
    };

  } else if (varmap.count("id")) {
    // Add our own stacktrace dumping
    if (varmap["stacktrace-on-signal"].as<std::string>() == "simple" && (getenv("O2_NO_CATCHALL_EXCEPTIONS") == nullptr || strcmp(getenv("O2_NO_CATCHALL_EXCEPTIONS"), "0") == 0)) {
      LOGP(info, "Instrumenting crash signals");
      signal(SIGSEGV, handle_crash);
      signal(SIGABRT, handle_crash);
      signal(SIGBUS, handle_crash);
      signal(SIGILL, handle_crash);
      signal(SIGFPE, handle_crash);
    }
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
                                                                                     DeviceSpecs const&,
                                                                                     DeviceExecutions const&,
                                                                                     DataProcessorInfos& dataProcessorInfos,
                                                                                     CommandInfo const& commandInfo) {
      if (filename == "-") {
        WorkflowSerializationHelpers::dump(std::cout, workflow, dataProcessorInfos, commandInfo);
        // FIXME: this is to avoid trailing garbage..
        exit(0);
      } else {
        std::ofstream output(filename);
        WorkflowSerializationHelpers::dump(output, workflow, dataProcessorInfos, commandInfo);
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
  for (int i = 0; i < (int)indices.size(); i++) {
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
           std::vector<ResourcePolicy> const& resourcePolicies,
           std::vector<CallbacksPolicy> const& callbacksPolicies,
           std::vector<SendingPolicy> const& sendingPolicies,
           std::vector<ConfigParamSpec> const& currentWorkflowOptions,
           o2::framework::ConfigContext& configContext)
{
  O2_SIGNPOST_INIT();
  std::vector<std::string> currentArgs;
  std::vector<PluginInfo> plugins;

  for (int ai = 1; ai < argc; ++ai) {
    currentArgs.emplace_back(argv[ai]);
  }

  WorkflowInfo currentWorkflow{
    argv[0],
    currentArgs,
    currentWorkflowOptions};

  ProcessingPolicies processingPolicies;
  enum LogParsingHelpers::LogLevel minFailureLevel;
  bpo::options_description executorOptions("Executor options");
  const char* helpDescription = "print help: short, full, executor, or processor name";
  executorOptions.add_options()                                                                                                                                        //
    ("help,h", bpo::value<std::string>()->implicit_value("short"), helpDescription)                                                                                    //                                                                                                       //
    ("quiet,q", bpo::value<bool>()->zero_tokens()->default_value(false), "quiet operation")                                                                            //                                                                                                         //
    ("stop,s", bpo::value<bool>()->zero_tokens()->default_value(false), "stop before device start")                                                                    //                                                                                                           //
    ("single-step", bpo::value<bool>()->zero_tokens()->default_value(false), "start in single step mode")                                                              //                                                                                                             //
    ("batch,b", bpo::value<std::vector<std::string>>()->zero_tokens()->composing(), "batch processing mode")                                                           //                                                                                                               //
    ("no-batch", bpo::value<bool>()->zero_tokens(), "force gui processing mode")                                                                                       //                                                                                                            //
    ("no-cleanup", bpo::value<bool>()->zero_tokens()->default_value(false), "do not cleanup the shm segment")                                                          //                                                                                                               //
    ("hostname", bpo::value<std::string>()->default_value("localhost"), "hostname to deploy")                                                                          //                                                                                                                 //
    ("resources", bpo::value<std::string>()->default_value(""), "resources allocated for the workflow")                                                                //                                                                                                                   //
    ("start-port,p", bpo::value<unsigned short>()->default_value(22000), "start port to allocate")                                                                     //                                                                                                                     //
    ("port-range,pr", bpo::value<unsigned short>()->default_value(1000), "ports in range")                                                                             //                                                                                                                       //
    ("completion-policy,c", bpo::value<TerminationPolicy>(&processingPolicies.termination)->default_value(TerminationPolicy::QUIT),                                    //                                                                                                                       //
     "what to do when processing is finished: quit, wait")                                                                                                             //                                                                                                                      //
    ("error-policy", bpo::value<TerminationPolicy>(&processingPolicies.error)->default_value(TerminationPolicy::QUIT),                                                 //                                                                                                                          //
     "what to do when a device has an error: quit, wait")                                                                                                              //                                                                                                                            //
    ("min-failure-level", bpo::value<LogParsingHelpers::LogLevel>(&minFailureLevel)->default_value(LogParsingHelpers::LogLevel::Fatal),                                //                                                                                                                          //
     "minimum message level which will be considered as fatal and exit with 1")                                                                                        //                                                                                                                            //
    ("graphviz,g", bpo::value<bool>()->zero_tokens()->default_value(false), "produce graphviz output")                                                                 //                                                                                                                              //
    ("mermaid", bpo::value<std::string>()->default_value(""), "produce graph output in mermaid format in file under specified name or on stdout if argument is \"-\"") //                                                                                                                              //
    ("timeout,t", bpo::value<uint64_t>()->default_value(0), "forced exit timeout (in seconds)")                                                                        //                                                                                                                                //
    ("dds,D", bpo::value<std::string>()->default_value(""), "create DDS configuration")                                                                                //                                                                                                                                  //
    ("dds-workflow-suffix,D", bpo::value<std::string>()->default_value(""), "suffix for DDS names")                                                                    //                                                                                                                                  //
    ("dump-workflow,dump", bpo::value<bool>()->zero_tokens()->default_value(false), "dump workflow as JSON")                                                           //                                                                                                                                    //
    ("dump-workflow-file", bpo::value<std::string>()->default_value("-"), "file to which do the dump")                                                                 //                                                                                                                                      //
    ("run", bpo::value<bool>()->zero_tokens()->default_value(false), "run workflow merged so far. It implies --batch. Use --no-batch to see the GUI")                  //                                                                                                                                        //
    ("no-IPC", bpo::value<bool>()->zero_tokens()->default_value(false), "disable IPC topology optimization")                                                           //                                                                                                                                        //
    ("o2-control,o2", bpo::value<std::string>()->default_value(""), "dump O2 Control workflow configuration under the specified name")                                 //
    ("resources-monitoring", bpo::value<unsigned short>()->default_value(0), "enable cpu/memory monitoring for provided interval in seconds")                          //
    ("resources-monitoring-dump-interval", bpo::value<unsigned short>()->default_value(0), "dump monitoring information to disk every provided seconds");              //
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
  CommandInfo commandInfo{};

  if (isatty(STDIN_FILENO) == false && isInputConfig()) {
    std::vector<DataProcessorSpec> importedWorkflow;
    bool previousWorked = WorkflowSerializationHelpers::import(std::cin, importedWorkflow, dataProcessorInfos, commandInfo);
    if (previousWorked == false) {
      exit(1);
    }

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
                                [&name = dp.name](DataProcessorSpec const& spec) { return spec.name == name; });
      if (found == physicalWorkflow.end()) {
        physicalWorkflow.push_back(dp);
        rankIndex.insert(std::make_pair(dp.name, workflowHashB));
      }
    }
  }

  /// This is the earlies the services are actually needed
  OverrideServiceSpecs driverServicesOverride = ServiceSpecHelpers::parseOverrides(getenv("DPL_DRIVER_OVERRIDE_SERVICES"));
  ServiceSpecs driverServices = ServiceSpecHelpers::filterDisabled(CommonDriverServices::defaultServices(), driverServicesOverride);
  // We insert the hash for the internal devices.
  WorkflowHelpers::injectServiceDevices(physicalWorkflow, configContext);
  auto reader = std::find_if(physicalWorkflow.begin(), physicalWorkflow.end(), [](DataProcessorSpec& spec) { return spec.name == "internal-dpl-aod-reader"; });
  if (reader != physicalWorkflow.end()) {
    driverServices.push_back(ArrowSupport::arrowBackendSpec());
  }
  for (auto& service : driverServices) {
    if (service.injectTopology == nullptr) {
      continue;
    }
    WorkflowSpecNode node{physicalWorkflow};
    service.injectTopology(node, configContext);
  }
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

  std::vector<TopologyPolicy> topologyPolicies = TopologyPolicy::createDefaultPolicies();
  std::vector<TopologyPolicy::DependencyChecker> dependencyCheckers;
  dependencyCheckers.reserve(physicalWorkflow.size());

  for (auto& spec : physicalWorkflow) {
    for (auto& policy : topologyPolicies) {
      if (policy.matcher(spec)) {
        dependencyCheckers.push_back(policy.checkDependency);
        break;
      }
    }
  }
  assert(dependencyCheckers.size() == physicalWorkflow.size());
  // check if DataProcessorSpec at i depends on j
  auto checkDependencies = [&workflow = physicalWorkflow,
                            &dependencyCheckers](int i, int j) {
    TopologyPolicy::DependencyChecker& checker = dependencyCheckers[i];
    return checker(workflow[i], workflow[j]);
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
            std::ostringstream str;
            for (auto x : {i, j}) {
              str << physicalWorkflow[x].name << ":\n";
              str << "inputs:\n";
              for (auto& input : physicalWorkflow[x].inputs) {
                str << "- " << input << "\n";
              }
              str << "outputs:\n";
              for (auto& output : physicalWorkflow[x].outputs) {
                str << "- " << output << "\n";
              }
            }
            throw std::runtime_error(physicalWorkflow[i].name + " has circular dependency with " + physicalWorkflow[j].name + ":\n" + str.str());
          }
        }
      }
    }

    auto topoInfos = WorkflowHelpers::topologicalSort(physicalWorkflow.size(), &edges[0].first, &edges[0].second, sizeof(std::pair<int, int>), edges.size());
    if (topoInfos.size() != physicalWorkflow.size()) {
      throw std::runtime_error("Unable to do topological sort of the resulting workflow. Do you have loops?\n" + debugTopoInfo(physicalWorkflow, topoInfos, edges));
    }
    // Sort by layer and then by name, to ensure stability.
    std::stable_sort(topoInfos.begin(), topoInfos.end(), [&workflow = physicalWorkflow](TopoIndexInfo const& a, TopoIndexInfo const& b) {
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
    LOGP(error, "error parsing options of {}: {}", argv[0], e.what());
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
  conflicting_options(varmap, "run", "mermaid");
  conflicting_options(varmap, "dump-workflow", "graphviz");
  conflicting_options(varmap, "no-batch", "batch");

  if (varmap.count("help")) {
    printHelp(varmap, executorOptions, physicalWorkflow, currentWorkflowOptions);
    exit(0);
  }
  /// Set the fair::Logger severity to the one specified in the command line
  /// We do it by hand here, because FairMQ device is not initialsed until
  /// much later and we need the logger before that.
  if (varmap.count("severity")) {
    auto logLevel = varmap["severity"].as<std::string>();
    if (logLevel == "debug") {
      fair::Logger::SetConsoleSeverity(fair::Severity::debug);
    } else if (logLevel == "detail") {
      fair::Logger::SetConsoleSeverity(fair::Severity::detail);
    } else if (logLevel == "info") {
      fair::Logger::SetConsoleSeverity(fair::Severity::info);
    } else if (logLevel == "warning") {
      fair::Logger::SetConsoleSeverity(fair::Severity::warning);
    } else if (logLevel == "error") {
      fair::Logger::SetConsoleSeverity(fair::Severity::error);
    } else if (logLevel == "important") {
      fair::Logger::SetConsoleSeverity(fair::Severity::important);
    } else if (logLevel == "alarm") {
      fair::Logger::SetConsoleSeverity(fair::Severity::alarm);
    } else if (logLevel == "fatal") {
      fair::Logger::SetConsoleSeverity(fair::Severity::fatal);
    } else {
      LOGP(error, "Invalid log level '{}'", logLevel);
      exit(1);
    }
  }
  DriverControl driverControl;
  initialiseDriverControl(varmap, driverControl);

  auto evaluateBatchOption = [&varmap]() -> bool {
    if (varmap.count("no-batch") > 0) {
      return false;
    }
    if (varmap.count("batch") == 0) {
      // default value
      return isatty(fileno(stdout)) == 0;
    }
    // FIXME: should actually use the last value, but for some reason the
    // values are not filled into the vector, even if specifying `-b true`
    // need to find out why the boost program options example is not working
    // in our case. Might depend on the parser options
    // auto value = varmap["batch"].as<std::vector<std::string>>();
    return true;
  };
  DriverInfo driverInfo{
    .sendingPolicies = sendingPolicies,
    .callbacksPolicies = callbacksPolicies};
  driverInfo.states.reserve(10);
  driverInfo.sigintRequested = false;
  driverInfo.sigchldRequested = false;
  driverInfo.channelPolicies = channelPolicies;
  driverInfo.completionPolicies = completionPolicies;
  driverInfo.dispatchPolicies = dispatchPolicies;
  driverInfo.resourcePolicies = resourcePolicies;
  driverInfo.argc = argc;
  driverInfo.argv = argv;
  driverInfo.batch = evaluateBatchOption();
  driverInfo.noSHMCleanup = varmap["no-cleanup"].as<bool>();
  driverInfo.processingPolicies.termination = varmap["completion-policy"].as<TerminationPolicy>();
  driverInfo.processingPolicies.earlyForward = varmap["early-forward-policy"].as<EarlyForwardPolicy>();
  if (varmap["error-policy"].defaulted() && driverInfo.batch == false) {
    driverInfo.processingPolicies.error = TerminationPolicy::WAIT;
  } else {
    driverInfo.processingPolicies.error = varmap["error-policy"].as<TerminationPolicy>();
  }
  driverInfo.minFailureLevel = varmap["min-failure-level"].as<LogParsingHelpers::LogLevel>();
  driverInfo.startTime = uv_hrtime();
  driverInfo.timeout = varmap["timeout"].as<uint64_t>();
  driverInfo.deployHostname = varmap["hostname"].as<std::string>();
  driverInfo.resources = varmap["resources"].as<std::string>();
  driverInfo.resourcesMonitoringInterval = varmap["resources-monitoring"].as<unsigned short>();
  driverInfo.resourcesMonitoringDumpInterval = varmap["resources-monitoring-dump-interval"].as<unsigned short>();

  // FIXME: should use the whole dataProcessorInfos, actually...
  driverInfo.processorInfo = dataProcessorInfos;
  driverInfo.configContext = &configContext;

  commandInfo.merge(CommandInfo(argc, argv));

  std::string frameworkId;
  // If the id is set, this means this is a device,
  // otherwise this is the driver.
  if (varmap.count("id")) {
    // The framework id does not want to know anything about DDS template expansion
    // so we simply drop it. Notice that the "id" Property is still the same as the
    // original --id option.
    frameworkId = std::regex_replace(varmap["id"].as<std::string>(), std::regex{"_dds.*"}, "");
    driverInfo.uniqueWorkflowId = fmt::format("{}", getppid());
    driverInfo.defaultDriverClient = "stdout://";
  } else {
    driverInfo.uniqueWorkflowId = fmt::format("{}", getpid());
    driverInfo.defaultDriverClient = "ws://";
  }
  return runStateMachine(physicalWorkflow,
                         currentWorkflow,
                         dataProcessorInfos,
                         commandInfo,
                         driverControl,
                         driverInfo,
                         gDeviceMetricsInfos,
                         varmap,
                         driverServices,
                         frameworkId);
}

void doBoostException(boost::exception&, char const* processName)
{
  LOGP(error, "error while setting up workflow in {}: {}",
       processName, boost::current_exception_diagnostic_information(true));
}
#pragma GCC diagnostic push
