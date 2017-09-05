// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_GRAPHVIZHELPERS_H
#define FRAMEWORK_GRAPHVIZHELPERS_H

#include "Framework/WorkflowSpec.h"
#include "Framework/DeviceSpec.h"
#include <vector>
#include <iosfwd>

namespace o2 {
namespace framework {

struct GraphvizHelpers {
  using Devices = std::vector<DeviceSpec>;
  static void dumpDataProcessorSpec2Graphviz(std::ostream &, const WorkflowSpec &specs);
  static void dumpDeviceSpec2Graphviz(std::ostream &, const Devices &specs);
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_GRAPHVIZHELPERS_H
