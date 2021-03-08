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
#include <cstring>

namespace o2
{
namespace framework
{

void dumpDeviceSpec2DDS(std::ostream& out,
                        const std::vector<DeviceSpec>& specs,
                        const std::vector<DeviceExecution>& executions)
{
  out << R"(<topology name="o2-dataflow">)"
         "\n";
  assert(specs.size() == executions.size());

  for (size_t di = 0; di < specs.size(); ++di) {
    auto& spec = specs[di];
    auto& execution = executions[di];

    auto id = spec.id;
    std::replace(id.begin(), id.end(), '-', '_'); // replace all 'x' to 'y'
    out << "   "
        << R"(<decltask name=")" << id << R"(">)"
                                          "\n";
    out << "       "
        << R"(<exe reachable="true">)";
    for (size_t ai = 0; ai < execution.args.size(); ++ai) {
      const char* arg = execution.args[ai];
      if (!arg) {
        break;
      }
      // Do not print out the driver client explicitly
      if (strcmp(arg, "--driver-client-backend") == 0) {
        ai++;
        continue;
      }
      // Do not print out channel information
      if (strcmp(arg, "--channel-config") == 0) {
        ai++;
        continue;
      }
      // If channel-prefix is empty do not print it out
      if (strcmp(arg, "--channel-prefix") == 0 &&
          ai + 1 < execution.args.size() &&
          *execution.args[ai + 1] == 0) {
        ai++;
        continue;
      }
      out << arg << " ";
    }
    out << "--plugin-search-path $FAIRMQ_ROOT/lib --plugin dds";
    out << "</exe>\n";
    out << "   </decltask>\n";
  }
  out << "   <declcollection name=\"DPL\">\n       <tasks>\n";
  for (size_t di = 0; di < specs.size(); ++di) {
    out << "          <name>" << specs[di].name << "</name>\n";
  }
  out << "       </tasks>\n   </declcollection>\n";
  out << "</topology>\n";
}

} // namespace framework
} // namespace o2
