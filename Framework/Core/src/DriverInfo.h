// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_DRIVER_INFO_H
#define FRAMEWORK_DRIVER_INFO_H

#include <chrono>
#include <cstddef>
#include <map>
#include <vector>

#include <csignal>
#include <sys/select.h>

#include "Framework/ChannelConfigurationPolicy.h"
#include "Framework/ConfigParamSpec.h"
#include "DataProcessorInfo.h"

namespace o2
{
namespace framework
{

class ConfigContext;

/// Possible states for the DPL Driver application
///
/// INIT => Initial state where global initialization should happen
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
  GUI,
  REDEPLOY_GUI,
  QUIT_REQUESTED,
  HANDLE_CHILDREN,
  EXIT,
  UNKNOWN,
  PERFORM_CALLBACKS,
  MATERIALISE_WORKFLOW,
  IMPORT_CURRENT_WORKFLOW,
  DO_CHILD,
  LAST
};

/// These are the possible actions we can do
/// when a workflow is deemed complete (e.g. when we are done
/// reading from file).
enum struct TerminationPolicy { QUIT,
                                WAIT,
                                RESTART };

/// Information about the driver process (i.e.  / the one which calculates the
/// topology and actually spawns the devices )
struct DriverInfo {
  /// Stack with the states to be processed next.
  std::vector<DriverState> states;
  // Mapping between various pipes and the actual device information.
  // Key is the file description, value is index in the previous vector.
  std::map<int, size_t> socket2DeviceInfo;
  /// The first unused file descriptor
  int maxFd;
  fd_set childFdset;

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
  /// The argc with which the driver was started.
  int argc;
  /// The argv with which the driver was started.
  char** argv;
  /// Whether the driver was started in batch mode or not.
  bool batch;
  /// What we should do when the workflow is completed.
  enum TerminationPolicy terminationPolicy;
  /// The offset at which the process was started.
  std::chrono::time_point<std::chrono::steady_clock> startTime;
  /// The optional timeout after which the driver will request
  /// all the children to quit.
  double timeout;
  /// The start port to use when looking for a free range
  unsigned short startPort;
  /// The size of the port range to consider allocated
  unsigned short portRange;
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
};

} // namespace framework
} // namespace o2

#endif
