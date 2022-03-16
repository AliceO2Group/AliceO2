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

#include "DeviceStateHelpers.h"
#include "Framework/DeviceState.h"
#include <string_view>
#include <cstring>

namespace o2::framework
{

DeviceState::LoopReason loopReasonFromString(std::string_view reason)
{
  if (reason == "NO_REASON") {
    return DeviceState::LoopReason::NO_REASON;
  } else if (reason == "METRICS_MUST_FLUSH") {
    return DeviceState::LoopReason::METRICS_MUST_FLUSH;
  } else if (reason == "SIGNAL_ARRIVED") {
    return DeviceState::LoopReason::SIGNAL_ARRIVED;
  } else if (reason == "DATA_SOCKET_POLLED") {
    return DeviceState::LoopReason::DATA_SOCKET_POLLED;
  } else if (reason == "DATA_INCOMING") {
    return DeviceState::LoopReason::DATA_INCOMING;
  } else if (reason == "DATA_OUTGOING") {
    return DeviceState::LoopReason::DATA_OUTGOING;
  } else if (reason == "WS_COMMUNICATION") {
    return DeviceState::LoopReason::WS_COMMUNICATION;
  } else if (reason == "TIMER_EXPIRED") {
    return DeviceState::LoopReason::TIMER_EXPIRED;
  } else if (reason == "WS_CONNECTED") {
    return DeviceState::LoopReason::WS_CONNECTED;
  } else if (reason == "WS_CLOSING") {
    return DeviceState::LoopReason::WS_CLOSING;
  } else if (reason == "WS_READING") {
    return DeviceState::LoopReason::WS_READING;
  } else if (reason == "WS_WRITING") {
    return DeviceState::LoopReason::WS_WRITING;
  } else if (reason == "ASYNC_NOTIFICATION") {
    return DeviceState::LoopReason::ASYNC_NOTIFICATION;
  } else if (reason == "OOB_ACTIVITY") {
    return DeviceState::LoopReason::OOB_ACTIVITY;
  } else if (reason == "UNKNOWN") {
    return DeviceState::LoopReason::UNKNOWN;
  } else if (reason == "FIRST_LOOP") {
    return DeviceState::LoopReason::FIRST_LOOP;
  } else if (reason == "NEW_STATE_PENDING") {
    return DeviceState::LoopReason::NEW_STATE_PENDING;
  } else if (reason == "PREVIOUSLY_ACTIVE") {
    return DeviceState::LoopReason::PREVIOUSLY_ACTIVE;
  } else if (reason == "TRACE_CALLBACKS") {
    return DeviceState::LoopReason::TRACE_CALLBACKS;
  } else if (reason == "TRACE_USERCODE") {
    return DeviceState::LoopReason::TRACE_USERCODE;
  } else {
    return DeviceState::LoopReason::UNKNOWN;
  }
}

int DeviceStateHelpers::parseTracingFlags(std::string const& s)
{
  // split string by pipe and convert to int based
  // on the LoopReason enum
  std::vector<std::string_view> tokens;
  char const* first = s.c_str();
  char const* last = s.c_str();

  while (true) {
    char const* next = std::strchr(last, '|');
    if (next) {
      tokens.emplace_back(first, next - first);
      first = next + 1;
      last = first;
    } else {
      tokens.emplace_back(first, s.size() - (first - s.c_str()));
      break;
    }
  }
  int ret = 0;
  for (auto const& token : tokens) {
    ret |= static_cast<int>(loopReasonFromString(token));
  }
  return ret;
}
} // namespace o2::framework
