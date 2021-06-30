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

#ifndef O2_FRAMEWORK_DRIVERINFO_H_
#define O2_FRAMEWORK_DRIVERINFO_H_

#include <cstddef>
#include <vector>

#include <csignal>
#include <sys/select.h>

#include "Framework/ChannelConfigurationPolicy.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/TerminationPolicy.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/LogParsingHelpers.h"
#include "DataProcessorInfo.h"
#include "ResourcePolicy.h"

namespace o2::framework
{

class ConfigContext;

/// Possible states for the DPL Driver application
///
/// INIT => Initial state where global initialization should happen
/// MERGE_CONFIGS => Invoked to rework the configuration so that common
///                  options are homogeneous between different invokations.
/// SCHEDULE => Invoked whenever the topology or the resources associated
///             to it change.
/// RUNNING => At least one device is running and processing data.
/// GUI => Event loop for the GUI
/// EXIT => All devices are not running and exit was requested.
/// UNKNOWN
/// LAST => used to indicate how many states are defined
///
/// The state machine is the following:
///
/// @dot << "digraph [
///   INIT -> SCHEDULE
///   SCHEDULE -> RUNNING
///   RUNNING -> RUNNING
///   RUNNING -> GUI
///   GUI -> RUNNING
///   NONE -> QUIT_REQUESTED
///   QUIT_REQUESTED -> HANDLE_CHILDREN
///   RUNNING -> HANDLE_CHILDREN
///   RUNNING -> SCHEDULE
///   RUNNING -> EXIT
/// ]"
///
enum struct DriverState {
  INIT = 0,
  SCHEDULE,
  RUNNING,
  REDEPLOY_GUI,
  QUIT_REQUESTED,
  HANDLE_CHILDREN,
  EXIT,
  UNKNOWN,
  PERFORM_CALLBACKS,
  MATERIALISE_WORKFLOW,
  IMPORT_CURRENT_WORKFLOW,
  DO_CHILD,
  MERGE_CONFIGS,
  LAST
};

/// Information about the driver process (i.e.  / the one which calculates the
/// topology and actually spawns the devices )
struct DriverInfo {
  /// Stack with the states to be processed next.
  std::vector<DriverState> states;

  // Signal handler for children
  struct sigaction sa_handle_child;
  bool sigintRequested;
  bool sigchldRequested;
  /// These are the configuration policies for the channel creation.
  /// Since they are decided by the toplevel configuration, they belong
  /// to the driver process.
  std::vector<ChannelConfigurationPolicy> channelPolicies;
  /// These are the policies which can be applied to decide whether or not
  /// a given record is complete.
  std::vector<CompletionPolicy> completionPolicies;
  /// These are the policies which can be applied to decide when complete
  /// objects/messages are sent out
  std::vector<DispatchPolicy> dispatchPolicies;

  /// These are the policies which can be applied to decide when there
  /// is enough resources to process data.
  std::vector<ResourcePolicy> resourcePolicies;
  /// The argc with which the driver was started.
  int argc;
  /// The argv with which the driver was started.
  char** argv;
  /// Whether the driver was started in batch mode or not.
  bool batch;
  /// What we should do when the workflow is completed.
  enum TerminationPolicy terminationPolicy;
  /// What we should do when one device in the workflow has an error
  enum TerminationPolicy errorPolicy;
  /// The offset at which the process was started.
  uint64_t startTime;
  /// The optional timeout after which the driver will request
  /// all the children to quit.
  double timeout;
  /// The hostname which needs to be deployed by this instance of
  /// the driver. By default it will be localhost
  std::string deployHostname;
  /// resources which are allocated for the whole workflow by
  /// an external resource manager. If the value is an empty string
  /// resources are obtained from the localhost.
  std::string resources;
  /// The current set of metadata associated to each DataProcessor being
  /// executed.
  std::vector<DataProcessorInfo> processorInfo;
  /// The config context. We use a bare pointer because std::observer_ptr is not a thing, yet.
  ConfigContext const* configContext;
  /// The names for all the metrics which have been collected by this driver.
  /// Should always be sorted alphabetically to ease insertion.
  std::vector<std::string> availableMetrics;
  /// The amount of time to process inputs coming from all the processes
  float inputProcessingCost;
  /// The time between one input processing and the other.
  float inputProcessingLatency;
  /// The amount of time to draw last frame in the GUI
  float frameCost;
  /// The time between one frame and the other.
  float frameLatency;
  /// The unique id used for ipc communications
  std::string uniqueWorkflowId = "";
  /// Metrics gathering interval
  unsigned short resourcesMonitoringInterval;
  /// Port used by the websocket control. 0 means not initialised.
  unsigned short port = 0;
  /// Last port used for tracy
  short tracyPort = 8086;
  /// The minimum level after which the device will exit with 1
  LogParsingHelpers::LogLevel minFailureLevel = LogParsingHelpers::LogLevel::Fatal;

  /// Aggregate metrics calculated in the driver itself
  DeviceMetricsInfo metrics;
  /// Skip shared memory cleanup if set
  bool noSHMCleanup;
  /// Default value for the --driver-client-backend. Notice that if we start from
  /// the driver, the default backend will be the websocket one.  On the other hand,
  /// if the device is started standalone, the default becomes the old stdout:// so
  /// that it works as it used to in AliECS.
  std::string defaultDriverClient = "invalid";
};

struct DriverInfoHelper {
  static char const* stateToString(enum DriverState state);
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DRIVERINFO_H_
