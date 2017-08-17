// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/FrameworkGUIDebugger.h"
#include <vector>
#include <algorithm>

namespace o2 {
namespace framework {

// Dummy function in case we want to build without debugger.
std::function<void(void)>
getGUIDebugger(const std::vector<DeviceInfo> &infos,
               const std::vector<DeviceSpec> &devices,
               const std::vector<DeviceMetricsInfo> &metricsInfos,
               std::vector<DeviceControl> &controls
               )
{
  return [](){};
}

void showNodeGraph(bool* opened) {}

} // namespace framework
} // namespace o2
