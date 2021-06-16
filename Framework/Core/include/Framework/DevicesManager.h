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

#ifndef O2_FRAMEWORK_DEVICESMANAGER_H_
#define O2_FRAMEWORK_DEVICESMANAGER_H_

#include "Framework/DeviceControl.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceSpec.h"
#include <string>
#include <vector>

namespace o2::framework
{

struct DeviceIndex {
  int index;
};
struct DeviceControl;

struct DeviceMessageHandle {
  DeviceIndex ref;
  std::string message;
};

struct DevicesManager {
  void queueMessage(char const* receiver, char const* msg);
  void flush();

  std::vector<DeviceControl>& controls;
  std::vector<DeviceInfo>& infos;
  std::vector<DeviceSpec>& specs;
  std::vector<DeviceMessageHandle> messages;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DEVICESMANAGER_H_
