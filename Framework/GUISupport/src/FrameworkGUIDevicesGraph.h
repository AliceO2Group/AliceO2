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
#ifndef O2_FRAMEWORK_FRAMEWORKGUIDEVICEGRAPH_H_
#define O2_FRAMEWORK_FRAMEWORKGUIDEVICEGRAPH_H_

#include "Framework/DataProcessorInfo.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DataProcessingStates.h"

#include <vector>

namespace o2::framework::gui
{

class WorkspaceGUIState;

void showTopologyNodeGraph(WorkspaceGUIState& state,
                           std::vector<DeviceInfo> const& infos,
                           std::vector<DeviceSpec> const& specs,
                           std::vector<DataProcessingStates> const& allStates,
                           std::vector<DataProcessorInfo> const& metadata,
                           std::vector<DeviceControl>& controls,
                           std::vector<DeviceMetricsInfo> const& metricsInfos);

} // namespace o2::framework::gui

#endif // O2_FRAMEWORK_FRAMEWORKGUIDEVICEGRAPH_H_
