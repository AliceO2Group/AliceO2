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
#ifndef O2_FRAMEWORK_GRAPHVIZHELPERS_H
#define O2_FRAMEWORK_GRAPHVIZHELPERS_H

#include "Framework/WorkflowSpec.h"
#include "Framework/DeviceSpec.h"
#include <vector>
#include <iosfwd>

namespace o2::framework
{

struct GraphvizHelpers {
  using Devices = std::vector<DeviceSpec>;
  static void dumpDataProcessorSpec2Graphviz(std::ostream&, const WorkflowSpec& specs, std::vector<std::pair<int, int>> const& edges = {});
  static void dumpDeviceSpec2Graphviz(std::ostream&, const Devices& specs);
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_GRAPHVIZHELPERS_H
