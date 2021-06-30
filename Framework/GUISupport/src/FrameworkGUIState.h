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

/// State for the main GUI window

#include <vector>
#include <string>

namespace o2::framework::gui
{

/// State for the Device specific inspector
struct DeviceGUIState {
  std::string label;
};

/// State for the workspace
struct WorkspaceGUIState {
  int selectedMetric;
  size_t metricMinRange;
  size_t metricMaxRange;
  std::vector<DeviceGUIState> devices;
  float leftPaneSize;
  float rightPaneSize;
  float bottomPaneSize;
  bool leftPaneVisible;
  bool rightPaneVisible;
  bool bottomPaneVisible;
  double startTime;
};

} // namespace o2::framework::gui
