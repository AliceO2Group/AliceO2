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
#include "ProcessingPoliciesHelpers.h"
#include <ostream>
#include <istream>

namespace o2::framework
{

std::istream& operator>>(std::istream& in, enum TerminationPolicy& policy)
{
  std::string token;
  in >> token;
  if (token == "quit") {
    policy = TerminationPolicy::QUIT;
  } else if (token == "wait") {
    policy = TerminationPolicy::WAIT;
  } else {
    in.setstate(std::ios_base::failbit);
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, const enum DriverMode& mode)
{
  if (mode == DriverMode::STANDALONE) {
    out << "standalone";
  } else if (mode == DriverMode::EMBEDDED) {
    out << "embedded";
  } else {
    out.setstate(std::ios_base::failbit);
  }
  return out;
}

std::istream& operator>>(std::istream& in, enum DriverMode& mode)
{
  std::string token;
  in >> token;
  if (token == "standalone") {
    mode = DriverMode::STANDALONE;
  } else if (token == "embedded") {
    mode = DriverMode::EMBEDDED;
  } else {
    in.setstate(std::ios_base::failbit);
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, const enum TerminationPolicy& policy)
{
  if (policy == TerminationPolicy::QUIT) {
    out << "quit";
  } else if (policy == TerminationPolicy::WAIT) {
    out << "wait";
  } else {
    out.setstate(std::ios_base::failbit);
  }
  return out;
}

std::istream& operator>>(std::istream& in, enum EarlyForwardPolicy& policy)
{
  std::string token;
  in >> token;
  if (token == "never") {
    policy = EarlyForwardPolicy::NEVER;
  } else if (token == "noraw") {
    policy = EarlyForwardPolicy::NORAW;
  } else if (token == "always") {
    policy = EarlyForwardPolicy::ALWAYS;
  } else {
    in.setstate(std::ios_base::failbit);
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, const enum EarlyForwardPolicy& policy)
{
  if (policy == EarlyForwardPolicy::NEVER) {
    out << "never";
  } else if (policy == EarlyForwardPolicy::NORAW) {
    out << "noraw";
  } else if (policy == EarlyForwardPolicy::ALWAYS) {
    out << "always";
  } else {
    out.setstate(std::ios_base::failbit);
  }
  return out;
}

std::istream& operator>>(std::istream& in, enum LogParsingHelpers::LogLevel& level)
{
  std::string token;
  in >> token;
  if (token == "debug") {
    level = LogParsingHelpers::LogLevel::Debug;
  } else if (token == "info") {
    level = LogParsingHelpers::LogLevel::Info;
  } else if (token == "warning") {
    level = LogParsingHelpers::LogLevel::Warning;
  } else if (token == "error") {
    level = LogParsingHelpers::LogLevel::Error;
  } else if (token == "fatal") {
    level = LogParsingHelpers::LogLevel::Fatal;
  } else {
    in.setstate(std::ios_base::failbit);
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, const enum LogParsingHelpers::LogLevel& level)
{
  if (level == LogParsingHelpers::LogLevel::Debug) {
    out << "debug";
  } else if (level == LogParsingHelpers::LogLevel::Info) {
    out << "info";
  } else if (level == LogParsingHelpers::LogLevel::Warning) {
    out << "warning";
  } else if (level == LogParsingHelpers::LogLevel::Error) {
    out << "error";
  } else if (level == LogParsingHelpers::LogLevel::Fatal) {
    out << "fatal";
  } else {
    out.setstate(std::ios_base::failbit);
  }
  return out;
}
} // namespace o2::framework
