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
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceState.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/GuiCallbackContext.h"
#include "SpyServiceHelpers.h"

#include <string>
#include <string_view>

namespace o2::framework
{
SpyService::SpyService(ServiceRegistry& registry, DeviceState& deviceState)
  : mRegistry{registry},
    mDeviceState{deviceState}
{
  renderer = new GuiRenderer;
}

ServiceSpec* SpyGUIPlugin::create()
{
  return new ServiceSpec{
    .name = "spy",
    .init = [](ServiceRegistry& services, DeviceState& state, fair::mq::ProgOptions& options) -> ServiceHandle {
      return ServiceHandle{TypeIdHelpers::uniqueId<SpyService>(), new SpyService(services, state)};
    },
    .configure = CommonServices::noConfiguration(),
    .preSendingMessages = [](ServiceRegistry& registry, fair::mq::Parts& parts, ChannelIndex channelIndex) {
      auto &spy = registry.get<SpyService>();
              spy.parts = &parts;
              spy.partsAlive = false;

              auto loop = registry.get<DeviceState>().loop;
              GuiRenderer* renderer = registry.get<SpyService>().renderer;

              if (renderer->guiConnected && uv_now(loop) > spy.enableAfter) {
                LOG(info) << "Sending Policy uv_run";
                registry.get<SpyService>().partsAlive = true;
                uv_run(loop, UV_RUN_DEFAULT);
              } },
    .postRenderGUI = [](ServiceRegistry& registry) { SpyServiceHelpers::webGUI(registry); },
    .kind = ServiceKind::Serial};
};

} // namespace o2::framework
