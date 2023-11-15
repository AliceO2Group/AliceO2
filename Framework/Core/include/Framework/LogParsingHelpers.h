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
#ifndef O2_FRAMEWORK_LOGPARSINGHELPERS_H_
#define O2_FRAMEWORK_LOGPARSINGHELPERS_H_

#include <string>
#include <string_view>

namespace o2::framework
{

/// A set of helpers to parse device logs.
struct LogParsingHelpers {
  /// Possible log levels for device log entries.
  enum struct LogLevel {
    Debug,
    Info,
    Warning,
    Alarm,
    Error,
    Fatal,
    Unknown,
    Size
  };

  /// Available log levels
  static char const* const LOG_LEVELS[(int)LogLevel::Size];

  /// Extract the log style from a log string @param s
  /// Token style can then be used for colouring the logs
  /// in the GUI or to exit with error if a sufficient
  /// number of LogLevel::Error is found.
  static LogLevel parseTokenLevel(std::string_view const s);
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_LOGPARSINGHELPERS_H_
