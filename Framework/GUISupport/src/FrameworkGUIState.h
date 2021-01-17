// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// State for the main GUI window

#include <vector>
#include <string>

namespace o2
{
namespace framework
{
namespace gui
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
};

} // namespace gui
} // namespace framework
} // namespace o2
