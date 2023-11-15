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
#include "Framework/LogParsingHelpers.h"
#include <regex>

namespace o2::framework
{

char const* const LogParsingHelpers::LOG_LEVELS[(int)LogParsingHelpers::LogLevel::Size] = {
  "DEBUG",
  "INFO",
  "WARNING",
  "ALARM",
  "ERROR",
  "FATAL",
  "UNKNOWN"};
using LogLevel = o2::framework::LogParsingHelpers::LogLevel;

LogLevel LogParsingHelpers::parseTokenLevel(std::string_view const s)
{

  // Example format: [99:99:99][ERROR] (string begins with that, longest is 17 chars)
  constexpr size_t MAXPREFLEN = 17;
  constexpr size_t LABELPOS = 10;
  if (s.size() < MAXPREFLEN) {
    return LogLevel::Unknown;
  }

  // Check if first chars match [NN:NN:NN]
  //                            0123456789
  if ((unsigned char)s[0] != '[' || (unsigned char)s[9] != ']' ||
      (unsigned char)s[3] != ':' || (unsigned char)s[6] != ':' ||
      (unsigned char)s[1] - '0' > 9 || (unsigned char)s[2] - '0' > 9 ||
      (unsigned char)s[4] - '0' > 9 || (unsigned char)s[5] - '0' > 9 ||
      (unsigned char)s[7] - '0' > 9 || (unsigned char)s[8] - '0' > 9) {
    return LogLevel::Unknown;
  }

  if (s.compare(LABELPOS, 8, "[DEBUG] ") == 0) {
    return LogLevel::Debug;
  } else if (s.compare(LABELPOS, 7, "[INFO] ") == 0 ||
             s.compare(LABELPOS, 8, "[STATE] ") == 0) {
    return LogLevel::Info;
  } else if (s.compare(LABELPOS, 7, "[WARN] ") == 0) {
    return LogLevel::Warning;
  } else if (s.compare(LABELPOS, 8, "[ALARM] ") == 0) {
    return LogLevel::Alarm;
  } else if (s.compare(LABELPOS, 8, "[ERROR] ") == 0) {
    return LogLevel::Error;
  } else if (s.compare(LABELPOS, 8, "[FATAL] ") == 0) {
    return LogLevel::Fatal;
  }
  return LogLevel::Unknown;
}
} // namespace o2
