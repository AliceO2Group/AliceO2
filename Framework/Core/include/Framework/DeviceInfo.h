// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DEVICEINFO_H
#define FRAMEWORK_DEVICEINFO_H

#include "Framework/LogParsingHelpers.h"
#include "Framework/Variant.h"

#include <cstddef>
#include <string>
#include <vector>
// For pid_t
#include <unistd.h>
#include <array>

namespace o2
{
namespace framework
{

/// Index of what metrics have to be picked up
/// for the DataRelayer view of a given device.
struct DataRelayerViewIndex {
  int w;
  int h;
  std::vector<size_t> indexes;
  /// Whether or not the view is ready to be used.
  bool isComplete() const { return (w * h) != 0; }
};

struct DeviceInfo {
  /// The pid of the device associated to this device
  pid_t pid;
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
  bool readyToQuit;
  /// Index for a particular release.
  DataRelayerViewIndex dataRelayerViewIndex;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_DEVICEINFO_H
