// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/ControlService.h"
#include "Framework/DriverClient.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceState.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/Logger.h"
#include "DataProcessingHelpers.h"
#include <string>
#include <string_view>
#include <regex>
#include <iostream>

namespace o2::framework
{

ControlService::ControlService(ServiceRegistry& registry, DeviceState& deviceState)
  : mRegistry{registry},
    mDeviceState{deviceState},
    mDriverClient{registry.get<DriverClient>()}
{
}

// This will send an end of stream to all the devices downstream.
void ControlService::endOfStream()
{
  std::scoped_lock lock(mMutex);
  mDeviceState.streaming = StreamingState::EndOfStreaming;
}

// All we do is to printout
void ControlService::readyToQuit(QuitRequest what)
{
  std::scoped_lock lock(mMutex);
  if (mOnce == true) {
    return;
  }
  mOnce = true;
  switch (what) {
    case QuitRequest::All:
      mDeviceState.quitRequested = true;
      mDriverClient.tell("CONTROL_ACTION: READY_TO_QUIT_ALL");
      break;
    case QuitRequest::Me:
      mDeviceState.quitRequested = true;
      mDriverClient.tell("CONTROL_ACTION: READY_TO_QUIT_ME");
      break;
  }
}

void ControlService::notifyStreamingState(StreamingState state)
{
  std::scoped_lock lock(mMutex);
  switch (state) {
    case StreamingState::Idle:
      mDriverClient.tell("CONTROL_ACTION: NOTIFY_STREAMING_STATE IDLE");
      break;
    case StreamingState::Streaming:
      mDriverClient.tell("CONTROL_ACTION: NOTIFY_STREAMING_STATE STREAMING");
      break;
    case StreamingState::EndOfStreaming:
      mDriverClient.tell("CONTROL_ACTION: NOTIFY_STREAMING_STATE EOS");
      break;
    default:
      throw std::runtime_error("Unknown streaming state");
  }
}

bool parseControl(std::string const& s, std::smatch& match)
{
  char const* action = strstr(s.data(), "CONTROL_ACTION:");
  if (action == nullptr) {
    return false;
  }
  const static std::regex controlRE1(".*CONTROL_ACTION: READY_TO_(QUIT)_(ME|ALL)", std::regex::optimize);
  const static std::regex controlRE2(".*CONTROL_ACTION: (NOTIFY_STREAMING_STATE) (IDLE|STREAMING|EOS)", std::regex::optimize);
  return std::regex_search(s, match, controlRE1) || std::regex_search(s, match, controlRE2);
}

} // namespace o2::framework
