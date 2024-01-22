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
#ifndef O2_FRAMEWORK_DEVICECONTROL_H_
#define O2_FRAMEWORK_DEVICECONTROL_H_

#include "Framework/LogParsingHelpers.h"

#include <map>
#include <string>

namespace o2::framework
{

struct DeviceController;

constexpr int MAX_USER_FILTER_SIZE = 256;

/// Controller state for the Device. This is useful for both GUI and batch
/// operations of the system. Whenever something external to the device wants
/// to modify it, it should be registered here and it will be acted on in the
/// subsequent state update.
struct DeviceControl {
  // whether the device should start in STOP
  bool stopped = false;
  /// wether we should be capturing device output.
  bool quiet = false;
  /// wether the log window should be opened.
  bool logVisible = false;
  /// Minimum log level for messages to appear
  LogParsingHelpers::LogLevel logLevel = LogParsingHelpers::LogLevel::Info;
  /// Lines in the log should match this to be displayed
  char logFilter[MAX_USER_FILTER_SIZE] = {0};
  /// Start printing log with the last occurence of this
  char logStartTrigger[MAX_USER_FILTER_SIZE] = {0};
  /// Stop producing log with the first occurrence of this after the start
  char logStopTrigger[MAX_USER_FILTER_SIZE] = {0};
  /// Where the GUI should store the options it wants.
  std::map<std::string, std::string> options;
  /// Handler used to communicate with the device (if available)
  DeviceController* controller = nullptr;
  /// What kind of events should run with the TRACE level
  int tracingFlags = 0;
  /// What kind of log streams should be enabled
  int logStreams = 0;
  /// An incremental number to identify the device state
  int requestedState = 0;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DEVICECONTROL_H_
