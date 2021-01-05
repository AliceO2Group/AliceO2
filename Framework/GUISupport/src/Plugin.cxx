// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  void* initGUI(char const* windowTitle) override
  {
    return o2::framework::initGUI(windowTitle);
  }
  bool pollGUI(void* context, std::function<void(void)> guiCallback) override
  {
    return o2::framework::pollGUI(context, guiCallback);
  }
  void disposeGUI() override
  {
    o2::framework::disposeGUI();
  }
};

DEFINE_DPL_PLUGIN(ImGUIDebugGUI, DebugGUIImpl);
