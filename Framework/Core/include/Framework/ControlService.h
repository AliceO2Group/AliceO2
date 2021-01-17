// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_CONTROLSERVICE_H_
#define O2_FRAMEWORK_CONTROLSERVICE_H_

#include "Framework/ServiceHandle.h"
#include <regex>
#include <mutex>

namespace o2::framework
{

struct ServiceRegistry;
struct DeviceState;
struct DriverClient;

enum struct StreamingState : int;

/// Kind of request we want to issue to control
enum struct QuitRequest {
  /// Only quit this data processor
  Me = 0,
  /// Quit all data processor, regardless of their state
  All = 1,
};

/// A service that data processors can use to talk to control and ask for their
/// own state change or others.
/// A ControlService is requried to be a ServiceKind::Global kind of service.
class ControlService
{
 public:
  constexpr static ServiceKind service_kind = ServiceKind::Global;

  ControlService(ServiceRegistry& registry, DeviceState& deviceState);
  /// Compatibility with old API.
  void readyToQuit(bool all) { this->readyToQuit(all ? QuitRequest::All : QuitRequest::Me); }
  /// Signal control that we are potentially ready to quit some / all
  /// dataprocessor.
  void readyToQuit(QuitRequest kind);
  /// Signal that we are done with the current stream
  void endOfStream();
  /// Report the current streaming state of a given device
  void notifyStreamingState(StreamingState state);

 private:
  bool mOnce = false;
  ServiceRegistry& mRegistry;
  DeviceState& mDeviceState;
  DriverClient& mDriverClient;
  std::mutex mMutex;
};

bool parseControl(std::string const& s, std::smatch& match);

} // namespace o2::framework
#endif // O2_FRAMEWORK_CONTROLSERVICE_H_
