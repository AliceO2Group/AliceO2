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
#include <fmt/format.h>
#include <libgen.h>

namespace o2::framework
{

std::string replaceFirstOccurrence(
  std::string s,
  const std::string& toReplace,
  const std::string& replaceWith)
{
  std::size_t pos = s.find(toReplace);
  if (pos == std::string::npos) {
    return s;
  }
  return s.replace(pos, toReplace.length(), replaceWith);
}

void dumpDeviceSpec2DDS(std::ostream& out,
                        const std::vector<DeviceSpec>& specs,
                        const std::vector<DeviceExecution>& executions,
                        const CommandInfo& commandInfo)
{
  out << R"(<topology name="o2-dataflow">)"
         "\n";
  assert(specs.size() == executions.size());

  for (size_t di = 0; di < specs.size(); ++di) {
    auto& spec = specs[di];
    auto& execution = executions[di];
    if (execution.args.empty()) {
      continue;
    }
    out << "   "
        << fmt::format("<decltask name=\"{}\">\n", spec.id);
    out << "       "
        << R"(<exe reachable="true">)";
    out << replaceFirstOccurrence(commandInfo.command, "--dds", "--dump") << " | ";
    std::string accumulatedChannelPrefix;
    char* s = strdup(execution.args[0]);
    out << basename(s) << " ";
    free(s);
    for (size_t ai = 1; ai < execution.args.size(); ++ai) {
      const char* arg = execution.args[ai];
      if (!arg) {
        break;
      }
      // Do not print out the driver client explicitly
      if (strcmp(arg, "--driver-client-backend") == 0) {
        ai++;
        continue;
      }
      if (strcmp(arg, "--control") == 0) {
        ai++;
        continue;
      }
      if (strcmp(arg, "--channel-prefix") == 0 &&
          ai + 1 < execution.args.size() &&
          *execution.args[ai + 1] == 0) {
        ai++;
        continue;
      }
      if (strpbrk(arg, "' ;@") != nullptr || arg[0] == 0) {
        out << fmt::format(R"("{}" )", arg);
      } else if (strpbrk(arg, "\"") != nullptr || arg[0] == 0) {
        out << fmt::format(R"('{}' )", arg);
      } else {
        out << fmt::format(R"({} )", arg);
      }
    }
    out << "--plugin dds";
    if (accumulatedChannelPrefix.empty() == false) {
      out << " --channel-config \"" << accumulatedChannelPrefix << "\"";
    }
    out << "</exe>\n";
    out << "   </decltask>\n";
  }
  out << "   <declcollection name=\"DPL\">\n       <tasks>\n";
  for (size_t di = 0; di < specs.size(); ++di) {
    out << fmt::format("          <name>{}</name>\n", specs[di].id);
  }
  out << "       </tasks>\n   </declcollection>\n";
  out << "</topology>\n";
}

} // namespace o2::framework
