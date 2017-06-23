// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "DDSConfigHelpers.h"
#include <map>
#include <iostream>
#include <string>

namespace o2 {
namespace framework {

/// Helper to dump a set of devices as a graphviz file
void
dumpDeviceSpec2DDS(const std::vector<DeviceSpec> &specs)
{
  std::cout << R"(<topology id="o2-framework-topology">)" "\n";
  for (auto &spec : specs) {
    auto id = spec.id;
    std::replace(id.begin(), id.end(), '-', '_'); // replace all 'x' to 'y'
    std::cout << "   " << R"(<decltask id=")" << id << R"(">)" "\n";
    std::cout << "       " << R"(<exe reachable="true">)";
    for (auto &arg : spec.args) {
      if (!arg) {
        break;
      }
      std::cout << arg << " ";
    }
    std::cout << "</exe>\n";
    std::cout << "   </decltask>\n";
  }
  std::cout << "</topology>\n";
}

} // namespace framework
} // namespace o2
