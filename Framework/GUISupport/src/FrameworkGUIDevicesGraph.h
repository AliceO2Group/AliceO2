// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_FRAMEWORKGUIDEVICEGRAPH_H_
#define O2_FRAMEWORK_FRAMEWORKGUIDEVICEGRAPH_H_

#include "Framework/DataProcessorInfo.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceMetricsInfo.h"

#include <vector>

namespace o2::framework::gui
{

class WorkspaceGUIState;

void showTopologyNodeGraph(WorkspaceGUIState& state,
                           std::vector<DeviceInfo> const& infos,
                           std::vector<DeviceSpec> const& specs,
                           std::vector<DataProcessorInfo> const& metadata,
                           std::vector<DeviceControl>& controls,
                           std::vector<DeviceMetricsInfo> const& metricsInfos);

} // namespace o2::framework::gui

#endif // O2_FRAMEWORK_FRAMEWORKGUIDEVICEGRAPH_H_
