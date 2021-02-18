// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DEVICEINFO_H_
#define O2_FRAMEWORK_DEVICEINFO_H_

#include "Framework/LogParsingHelpers.h"
#include "Framework/Metric2DViewIndex.h"
#include "Framework/Variant.h"
#include "Framework/DeviceState.h"

#include <cstddef>
#include <string>
#include <vector>
// For pid_t
#include <unistd.h>
#include <array>
#include <boost/property_tree/ptree.hpp>

namespace o2::framework
{

struct DeviceInfo {
  /// The pid of the device associated to this device
  pid_t pid;
  /// The exit status of the device, if not running.
  /// Notice that -1 means that no exit status was set,
  /// since the actual values which will be seen by the parent
  /// are guaranteed to be between 0 and 255.
  int exitStatus = -1;
  /// The position inside the history circular buffer of this device
  size_t historyPos;
  /// The size of the history circular buffer
  size_t historySize;
  /// The maximum log level ever seen by this device
  LogParsingHelpers::LogLevel maxLogLevel;
  /// A circular buffer for the history of logs entries received
  /// by this device
  std::vector<std::string> history;
  /// A circular buffer for the severity of each of the entries
  /// in the circular buffer associated to the device.
  std::vector<LogParsingHelpers::LogLevel> historyLevel;
  std::string lastError;
  /// An unterminated string which is not ready to be printed yet
  std::string unprinted;
  /// Whether the device is active (running) or not.
  bool active;
  /// Whether the device is ready to quit.
  bool readyToQuit = false;
  /// The current state of the device, as reported by it
  StreamingState streamingState = StreamingState::Streaming;
  /// Index for a particular relayer.
  Metric2DViewIndex dataRelayerViewIndex;
  /// Index for the variables of a given relayer.
  Metric2DViewIndex variablesViewIndex;
  /// Index for the queries of each input route.
  Metric2DViewIndex queriesViewIndex;
  /// Current configuration for the device
  boost::property_tree::ptree currentConfig;
  /// Current provenance for the configuration keys
  boost::property_tree::ptree currentProvenance;
  /// Port to use to connect to tracy profiler
  short tracyPort;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_DEVICEINFO_H_
