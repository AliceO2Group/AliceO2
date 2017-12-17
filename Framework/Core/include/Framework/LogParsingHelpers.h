// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_LOGPARSINGHELPERS
#define FRAMEWORK_LOGPARSINGHELPERS

#include <string>

namespace o2 {
namespace framework {

/// A set of helpers to parse device logs.
struct LogParsingHelpers {
  /// Possible log levels for device log entries.
  enum struct LogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Unknown,
    Size
  };

  /// Available log levels
  static char const* const LOG_LEVELS[(int)LogLevel::Size];

  /// Extract the log style from a log string @param s
  /// Token style can then be used for colouring the logs
  /// in the GUI or to exit with error if a sufficient
  /// number of LogLevel::Error is found.
  static LogLevel parseTokenLevel(const std::string &s);
};

}
}
#endif // FRAMEWORK_LOGPARSINGHELPERS
