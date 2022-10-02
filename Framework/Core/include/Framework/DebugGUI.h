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
#ifndef O2_FRAMEWORK_DEBUGUIINTERFACE_H_
#define O2_FRAMEWORK_DEBUGUIINTERFACE_H_

#include "Framework/DeviceInfo.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceControl.h"
#include "Framework/DataProcessorInfo.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DriverInfo.h"
#include "Framework/DriverControl.h"

#include <functional>
#include <vector>

namespace o2::framework
{
/// Plugin interface for DPL GUIs.
struct DebugGUI {
  virtual std::function<void(void)> getGUIDebugger(std::vector<o2::framework::DeviceInfo> const& infos,
                                                   std::vector<o2::framework::DeviceSpec> const& devices,
                                                   std::vector<o2::framework::DataProcessorInfo> const& metadata,
                                                   std::vector<o2::framework::DeviceMetricsInfo> const& metricsInfos,
                                                   o2::framework::DriverInfo const& driverInfo,
                                                   std::vector<o2::framework::DeviceControl>& controls,
                                                   o2::framework::DriverControl& driverControl) = 0;
  virtual void updateMousePos(float x, float y) = 0;
  virtual void updateMouseButton(bool isClicked) = 0;
  virtual void updateMouseWheel(int direction) = 0;
  virtual void updateWindowSize(int x, int y) = 0;
  virtual void keyDown(char key) = 0;
  virtual void keyUp(char key) = 0;
  virtual void charIn(char key) = 0;

  virtual void* initGUI(char const* windowTitle, ServiceRegistry& registry) = 0;
  virtual void getFrameJSON(void* data, std::ostream& json_data) = 0;
  virtual void getFrameRaw(void* data, void** raw_data, int* size) = 0;
  virtual bool pollGUIPreRender(void* context, float delta) = 0;
  virtual void* pollGUIRender(std::function<void(void)> guiCallback) = 0;
  virtual void pollGUIPostRender(void* context, void* draw_data) = 0;
  virtual void disposeGUI() = 0;
};
} // namespace o2::framework
#endif // O2_FRAMEWORK_DEBUGUIINTERFACE_H_
