// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/LogParsingHelpers.h"
#include <regex>

namespace o2 {
namespace framework {

char const* const LogParsingHelpers::LOG_LEVELS[(int)LogParsingHelpers::LogLevel::Size] = {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "UNKNOWN"
  };
using LogLevel = o2::framework::LogParsingHelpers::LogLevel;

LogLevel LogParsingHelpers::parseTokenLevel(const std::string &s) {
  std::smatch match;
  const static std::regex metricsRE(R"regex(^\[[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\]\[(DEBUG|INFO|STATE|WARN|ERROR)\] .*)regex");
  std::regex_match(s, match, metricsRE);

  if (match.empty()) {
    return LogLevel::Unknown;
  }
  if (match[1] == "DEBUG") {
    return LogLevel::Debug;
  } else if (match[1] == "INFO" || match[1] == "STATE") {
    return LogLevel::Info;
  } else if (match[1] == "WARN") {
    return LogLevel::Warning;
  } else if (match[1] == "ERROR") {
    return LogLevel::Error;
  }
  return LogLevel::Unknown;
}

}
}
