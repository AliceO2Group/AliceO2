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
#include "SpyService.h"
#include <uv.h>
#include "Framework/DriverClient.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceState.h"
#include "Framework/ServiceRegistry.h"
#include "GuiCallbackContext.h"

#include <string>
#include <string_view>

namespace o2::framework
{
SpyService::SpyService(ServiceRegistry& registry, DeviceState& deviceState)
  : mRegistry{registry},
    mDeviceState{deviceState},
    mDriverClient{registry.get<DriverClient>()}
{
  renderer = new GuiRenderer;
}

void SpyService::sendHeader(std::string header)
{
  std::scoped_lock lock(mMutex);
  mDriverClient.tell(fmt::format("HEADER: {}", header));
  mDriverClient.flushPending();
}

void SpyService::sendData(std::string data, int num)
{
  std::scoped_lock lock(mMutex);
  mDriverClient.tell(fmt::format("DATA: {}, {}", num, data));
  mDriverClient.flushPending();
}

} // namespace o2::framework