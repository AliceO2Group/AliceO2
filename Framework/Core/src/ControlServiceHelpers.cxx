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
#include "ControlServiceHelpers.h"
#include "Framework/RawDeviceService.h"
#include "Framework/Logger.h"
#include "Framework/DeviceInfo.h"
#include <string>
#include <string_view>
#include <regex>
#include <iostream>

namespace o2::framework
{

bool ControlServiceHelpers::parseControl(std::string const& s, std::smatch& match)
{
  char const* action = strstr(s.data(), "CONTROL_ACTION:");
  if (action == nullptr) {
    return false;
  }
  const static std::regex controlRE1(".*CONTROL_ACTION: READY_TO_(QUIT)_(ME|ALL)", std::regex::optimize);
  const static std::regex controlRE2(".*CONTROL_ACTION: (NOTIFY_STREAMING_STATE) (IDLE|STREAMING|EOS)", std::regex::optimize);
  const static std::regex controlRE3(".*CONTROL_ACTION: (NOTIFY_DEVICE_STATE) ([A-Z ]*)", std::regex::optimize);
  return std::regex_search(s, match, controlRE1) || std::regex_search(s, match, controlRE2) || std::regex_search(s, match, controlRE3);
}

void ControlServiceHelpers::processCommand(std::vector<DeviceInfo>& infos,
                                           pid_t pid,
                                           std::string const& command,
                                           std::string const& arg)
{
  auto doToMatchingPid = [](std::vector<DeviceInfo>& infos, pid_t pid, auto lambda) {
    for (auto& deviceInfo : infos) {
      if (deviceInfo.pid == pid) {
        return lambda(deviceInfo);
      }
    }
    LOGP(error, "Command received for pid {} which does not exists.", pid);
  };
  LOGP(debug2, "Found control command {} from pid {} with argument {}.", command, pid, arg);
  if (command == "QUIT" && arg == "ALL") {
    for (auto& deviceInfo : infos) {
      deviceInfo.readyToQuit = true;
    }
  } else if (command == "QUIT" && arg == "ME") {
    doToMatchingPid(infos, pid, [](DeviceInfo& info) { info.readyToQuit = true; });
  } else if (command == "NOTIFY_STREAMING_STATE" && arg == "IDLE") {
    // FIXME: this should really be a policy...
    doToMatchingPid(infos, pid, [](DeviceInfo& info) { info.readyToQuit = true; info.streamingState = StreamingState::Idle; });
  } else if (command == "NOTIFY_STREAMING_STATE" && arg == "STREAMING") {
    // FIXME: this should really be a policy...
    doToMatchingPid(infos, pid, [](DeviceInfo& info) { info.streamingState = StreamingState::Streaming; });
  } else if (command == "NOTIFY_STREAMING_STATE" && arg == "EOS") {
    // FIXME: this should really be a policy...
    doToMatchingPid(infos, pid, [](DeviceInfo& info) { info.streamingState = StreamingState::EndOfStreaming; });
  } else if (command == "NOTIFY_DEVICE_STATE") {
    doToMatchingPid(infos, pid, [arg](DeviceInfo& info) { info.deviceState = arg; info.providedState++; });
  } else {
    LOGP(error, "Unknown command {} with argument {}", command, arg);
  }
};

} // namespace o2::framework
