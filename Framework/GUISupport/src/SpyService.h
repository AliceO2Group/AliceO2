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

#ifndef O2_FRAMEWORK_SPYSERVICE_H
#define O2_FRAMEWORK_SPYSERVICE_H

#include "Framework/ServiceHandle.h"
#include "Framework/ServiceSpec.h"
#include <mutex>
#include <vector>
#include <fairmq/FwdDecls.h>

namespace o2::framework
{
struct ServiceRegistry;
struct DeviceState;
struct DriverClient;

struct GuiRenderer;

class SpyService
{
 public:
  SpyService(ServiceRegistry& registry, DeviceState& deviceState);

  GuiRenderer* renderer;
  int selectedFrame = 0;
  int selectedData = 0;
  int debugCurrentItemIndex = 0;

  fair::mq::Parts* parts;

  bool partsAlive = false;
  size_t enableAfter = 0;

 private:
  bool mOnce = false;
  ServiceRegistry& mRegistry;
  DeviceState& mDeviceState;
  std::mutex mMutex;
};

struct SpyGUIPlugin : public ServicePlugin {
  auto create() -> ServiceSpec* final;
};

} // namespace o2::framework

#endif // O2_SPYSERVICE_H
