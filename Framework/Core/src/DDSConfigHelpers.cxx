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

void
dumpDeviceSpec2DDS(std::ostream &out, const std::vector<DeviceSpec> &specs)
{
  out << R"(<topology id="o2-dataflow">)" "\n";
  for (auto &spec : specs) {
    auto id = spec.id;
    std::replace(id.begin(), id.end(), '-', '_'); // replace all 'x' to 'y'
    out << "   " << R"(<decltask id=")" << id << R"(">)" "\n";
    out << "       " << R"(<exe reachable="true">)";
    for (size_t ai = 0; ai < spec.args.size(); ++ai) {
      const char *arg = spec.args[ai];
      if (!arg) {
        break;
      }
      // Do not print out channel information
      if (strcmp(arg, "--channel-config") == 0) {
        ai++;
        continue;
      }
      out << arg << " ";
    }
    out << "</exe>\n";
    out << "   </decltask>\n";
  }
  out << "</topology>\n";
}

} // namespace framework
} // namespace o2
