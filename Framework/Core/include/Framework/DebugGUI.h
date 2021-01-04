// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  virtual void* initGUI(char const* windowTitle) = 0;
  virtual bool pollGUI(void* context, std::function<void(void)> guiCallback) = 0;
  virtual void disposeGUI() = 0;
};
} // namespace o2::framework
#endif // O2_FRAMEWORK_DEBUGUIINTERFACE_H_
