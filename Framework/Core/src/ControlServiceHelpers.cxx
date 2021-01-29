// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  return std::regex_search(s, match, controlRE1) || std::regex_search(s, match, controlRE2);
}

void ControlServiceHelpers::processCommand(std::vector<DeviceInfo>& infos,
                                           pid_t pid,
                                           std::string const& command,
                                           std::string const& arg)
{
  auto doToMatchingPid = [](std::vector<DeviceInfo>& infos, int pid, auto lambda) {
    for (auto& deviceInfo : infos) {
      if (deviceInfo.pid == pid) {
        lambda(deviceInfo);
        break;
      }
    }
  };
  LOGP(info, "Found control command {} from pid {} with argument {}.", command, pid, arg);
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
  }
};

} // namespace o2::framework
