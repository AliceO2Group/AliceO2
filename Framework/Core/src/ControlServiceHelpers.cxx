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
#include "Framework/RuntimeError.h"
#include "Framework/DataProcessingStates.h"
#include <string>
#include <string_view>
#include <regex>

namespace o2::framework
{

bool ControlServiceHelpers::parseControl(std::string_view const& s, std::match_results<std::string_view::const_iterator>& match)
{
  size_t pos = s.find("CONTROL_ACTION: ");
  if (pos == std::string::npos) {
    return false;
  }
  const static std::regex controlRE1("^READY_TO_(QUIT)_(ME|ALL)", std::regex::optimize);
  const static std::regex controlRE2("^(NOTIFY_STREAMING_STATE) (IDLE|STREAMING|EOS)", std::regex::optimize);
  const static std::regex controlRE3("^(NOTIFY_DEVICE_STATE) ([A-Z ]*)", std::regex::optimize);
  const static std::regex controlRE4("^(PUT) (.*)", std::regex::optimize);
  std::string_view sv = s.substr(pos + strlen("CONTROL_ACTION: "));
  return std::regex_search(sv.begin(), sv.end(), match, controlRE1) ||
         std::regex_search(sv.begin(), sv.end(), match, controlRE2) ||
         std::regex_search(sv.begin(), sv.end(), match, controlRE3) ||
         std::regex_search(sv.begin(), sv.end(), match, controlRE4);
}

void ControlServiceHelpers::processCommand(std::vector<DeviceInfo>& infos,
                                           std::vector<DataProcessingStates>& allStates,
                                           pid_t pid,
                                           std::string const& command,
                                           std::string const& arg)
{
  auto doToMatchingPid = [&](std::vector<DeviceInfo>& infos, pid_t pid, auto lambda) {
    assert(infos.size() == allStates.size());
    for (size_t i = 0; i < infos.size(); ++i) {
      auto& deviceInfo = infos[i];
      if (deviceInfo.pid == pid) {
        return lambda(deviceInfo);
      }
    }
    LOGP(error, "Command received for pid {} which does not exists.", pid);
  };
  auto doToMatchingStatePid = [&](std::vector<DeviceInfo>& infos, std::vector<DataProcessingStates>& allStates, pid_t pid, auto lambda) {
    assert(infos.size() == allStates.size());
    for (size_t i = 0; i < infos.size(); ++i) {
      auto& deviceInfo = infos[i];
      auto& states = allStates[i];
      if (deviceInfo.pid == pid) {
        return lambda(deviceInfo, states);
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
  } else if (command == "PUT") {
    doToMatchingStatePid(infos, allStates, pid, [&arg](DeviceInfo& info, DataProcessingStates& states) {
      /// Use scanf to parse PUT <key> <timestamp>
      // find the first space, that is the beginning of the key.
      // Find the position of the fist space in beginKey.
      auto beginKey = 0;
      // If we did not find it complain and return.
      if (beginKey == std::string::npos) {
        LOGP(error, "Cannot parse key in PUT command with arg {} for device {}", arg, info.pid);
        return;
      }
      auto endKey = arg.find(' ', beginKey + 1);
      if (endKey == std::string::npos) {
        LOGP(error, "Cannot parse timestamp in PUT command with arg {}", arg);
        return;
      }
      auto beginTimestamp = endKey + 1;
      auto endTimestamp = arg.find(' ', beginTimestamp + 1);
      if (endTimestamp == std::string::npos) {
        LOGP(error, "Cannot parse value in PUT command with arg {}", arg);
        return;
      }
      auto beginValue = endTimestamp + 1;
      auto endValue = arg.size();

      std::string_view key(arg.data() + beginKey, endKey - beginKey);
      std::string_view timestamp(arg.data() + beginTimestamp, endTimestamp - beginTimestamp);
      std::string_view value(arg.data() + beginValue, endValue - beginValue);
      // Find the assocaiated StateSpec and get the id.
      auto spec = std::find_if(states.stateSpecs.begin(), states.stateSpecs.end(), [&key](auto const& spec) {
        return spec.name == key;
      });
      if (spec == states.stateSpecs.end()) {
        LOGP(warn, "Cannot find state {}", key.data());
        return;
      }
      if (value.data() == nullptr) {
        LOGP(debug, "State {} value is null skipping", key.data());
        return;
      }
      /// Notice this will remap the actual time to the time we received the command.
      /// This should not be a problem, because we have separate states per device.
      states.updateState(DataProcessingStates::CommandSpec{.id = spec->stateId, .size = (int)value.size(), .data = value.data()});
      states.processCommandQueue();
    });
  } else {
    LOGP(error, "Unknown command {} with argument {}", command, arg);
  }
};

} // namespace o2::framework
