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
#if __has_include(<DebugGUI/DebugGUI.h>)
#include <DebugGUI/DebugGUI.h>
#else
#pragma message "Building DPL without Debug GUI"
#include "NoDebugGUI.h"
#endif

#include "Framework/Plugins.h"
#include "Framework/DebugGUI.h"
#include "FrameworkGUIDebugger.h"
#include "Framework/ServiceSpec.h"
#include "Framework/CommonServices.h"
#include "Framework/GuiCallbackContext.h"
#include "SpyService.h"
#include "SpyServiceHelpers.h"
#include <fairmq/Channel.h>

using namespace o2::framework;

struct ImGUIDebugGUI : o2::framework::DebugGUI {
  std::function<void(void)> getGUIDebugger(std::vector<DeviceInfo> const& infos,
                                           std::vector<DeviceSpec> const& devices,
                                           std::vector<DataProcessorInfo> const& metadata,
                                           std::vector<DeviceMetricsInfo> const& metricsInfos,
                                           DriverInfo const& driverInfo,
                                           std::vector<DeviceControl>& controls,
                                           DriverControl& driverControl) override
  {
    return o2::framework::gui::getGUIDebugger(infos, devices, metadata, metricsInfos, driverInfo, controls, driverControl);
  }

  void updateMousePos(float x, float y) override
  {
    o2::framework::gui::updateMousePos(x, y);
  }
  void updateMouseButton(bool clicked) override
  {
    o2::framework::gui::updateMouseButton(clicked);
  }
  void updateMouseWheel(int direction) override
  {
    o2::framework::gui::updateMouseWheel(direction);
  }
  void updateWindowSize(int x, int y) override
  {
    o2::framework::gui::updateWindowSize(x, y);
  }
  void keyDown(char key) override
  {
    o2::framework::gui::keyDown(key);
  }
  void keyUp(char key) override
  {
    o2::framework::gui::keyUp(key);
  }
  void charIn(char key) override
  {
    o2::framework::gui::charIn(key);
  }

  void* initGUI(char const* windowTitle, ServiceRegistryRef registry_) override
  {
    registry = &registry_;
    return o2::framework::initGUI(windowTitle);
  }
  void disposeGUI() override
  {
    o2::framework::disposeGUI();
  }
  void getFrameJSON(void* data, std::ostream& json_data) override
  {
    o2::framework::getFrameJSON(data, json_data);
  }
  void getFrameRaw(void* data, void** raw_data, int* size) override
  {
    o2::framework::getFrameRaw(data, raw_data, size);
  }
  bool pollGUIPreRender(void* context, float delta) override
  {
    return o2::framework::pollGUIPreRender(context, delta);
  }
  void* pollGUIRender(std::function<void(void)> guiCallback) override
  {
    auto* result = o2::framework::pollGUIRender(guiCallback);
    registry->postRenderGUICallbacks();
    return result;
  }
  void pollGUIPostRender(void* context, void* draw_data) override
  {
    o2::framework::pollGUIPostRender(context, draw_data);
  }
  ServiceRegistry* registry;
};

DEFINE_DPL_PLUGINS_BEGIN
DEFINE_DPL_PLUGIN_INSTANCE(ImGUIDebugGUI, DebugGUIImpl);
DEFINE_DPL_PLUGIN_INSTANCE(SpyGUIPlugin, CustomService);
DEFINE_DPL_PLUGINS_END
