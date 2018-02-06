// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_FRAMEWORKGUIDEBUGGER_H
#define FRAMEWORK_FRAMEWORKGUIDEBUGGER_H

#include "Framework/DeviceControl.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceSpec.h"

#include <functional>
#include <vector>

namespace o2
{
namespace framework
{

class DriverInfo;
class DriverControl;

std::function<void(void)> getGUIDebugger(const std::vector<DeviceInfo>& infos, const std::vector<DeviceSpec>& devices,
                                         const std::vector<DeviceMetricsInfo>& metricsInfos,
                                         const DriverInfo& driverInfo, std::vector<DeviceControl>& controls,
                                         DriverControl& driverControl);
} // namespace framework
} // namespace o2
#endif // FRAMEWORK_FRAMEWORKGUIDEBUGGER_H
