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
#include "LibUVInspector.h"
#include "Framework/ServiceHandle.h"
#include "Framework/ServiceSpec.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/CommonServices.h"
#include "DebugGUI/imgui.h"
#include "Framework/Logger.h"
#include <uv.h>

namespace o2::framework
{
ServiceSpec* LibUVInspectorGUIPlugin::create(void)
{
  return new ServiceSpec{
    .name = "libuv-inspector",
    .init = [](ServiceRegistry& services, DeviceState& state, fair::mq::ProgOptions& options) -> ServiceHandle {
      return ServiceHandle{TypeIdHelpers::uniqueId<LibUVInspector>(), new LibUVInspector()};
    },
    .configure = CommonServices::noConfiguration(),
    .postRenderGUI = [](ServiceRegistry& registry) { 
      LOG(info) << "LibUVInspectorGUIPlugin::postRenderGUI";
      ImGui::Begin("LibUV Inspector");
      ImGui::End(); },
    .kind = ServiceKind::Serial};
};

} // namespace o2::framework
