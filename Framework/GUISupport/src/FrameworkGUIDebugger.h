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
#ifndef O2_FRAMEWORK_FRAMEWORKGUIDEBUGGER_H_
#define O2_FRAMEWORK_FRAMEWORKGUIDEBUGGER_H_

#include "Framework/DataProcessorInfo.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataProcessingStates.h"

#include <functional>
#include <vector>

namespace o2::framework
{

class DriverInfo;
class DriverControl;

namespace gui
{
/// Helper to get the callback to draw the debug GUI
std::function<void(void)> getGUIDebugger(std::vector<DeviceInfo> const& infos,
                                         std::vector<DeviceSpec> const& devices,
                                         std::vector<DataProcessingStates> const& allStates,
                                         std::vector<DataProcessorInfo> const& metadata,
                                         std::vector<DeviceMetricsInfo> const& metricsInfos,
                                         DriverInfo const& driverInfo,
                                         std::vector<DeviceControl>& controls,
                                         DriverControl& driverControl);

void updateMousePos(float x, float y);
void updateMouseButton(bool clicked);
void updateMouseWheel(int direction);
void updateWindowSize(int x, int y);
void keyDown(char key);
void keyUp(char key);
void charIn(char key);

} // namespace gui
} // namespace o2::framework
#endif // O2_FRAMEWORK_FRAMEWORKGUIDEBUGGER_H_
