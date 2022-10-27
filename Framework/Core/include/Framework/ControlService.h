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
#ifndef O2_FRAMEWORK_CONTROLSERVICE_H_
#define O2_FRAMEWORK_CONTROLSERVICE_H_

#include "Framework/ThreadSafetyAnalysis.h"
#include "Framework/ServiceRegistryRef.h"
#include "Framework/ServiceHandle.h"
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

  ControlService(ServiceRegistryRef registry, DeviceState& deviceState);
  /// Compatibility with old API.
  void readyToQuit(bool all) { this->readyToQuit(all ? QuitRequest::All : QuitRequest::Me); }
  /// Signal control that we are potentially ready to quit some / all
  /// dataprocessor.
  void readyToQuit(QuitRequest kind);
  /// Signal that we are done with the current stream
  void endOfStream();
  /// Report the current streaming state of a given device
  void notifyStreamingState(StreamingState state);
  /// Report the current FairMQ state of a given device
  void notifyDeviceState(std::string state);

 private:
  bool mOnce = false;
  ServiceRegistryRef mRegistry;
  DeviceState& mDeviceState GUARDED_BY(mMutex);
  DriverClient& mDriverClient GUARDED_BY(mMutex);
  std::mutex mMutex;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_CONTROLSERVICE_H_
