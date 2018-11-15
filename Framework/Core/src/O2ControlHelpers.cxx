// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "O2ControlHelpers.h"
#include "ChannelSpecHelpers.h"
#include <map>
#include <iostream>
#include <cstring>

namespace o2
{
namespace framework
{

void dumpDeviceSpec2O2Control(std::ostream& out,
                              const std::vector<DeviceSpec>& specs,
                              const std::vector<DeviceExecution>& executions)
{
  out << R"(- o2:)"
      << "\n";
  out << R"(  tasks:)"
      << "\n";
  assert(specs.size() == executions.size());

  for (size_t di = 0; di < specs.size(); ++di) {
    auto& spec = specs[di];
    auto& execution = executions[di];

    out << R"(    - name: )" << spec.id << "\n";
    out << R"(        control: )"
        << "\n";
    out << R"(          mode: "fairmq")"
        << "\n";
    if (spec.outputChannels.empty()) {
      out << R"(      bind: [])"
          << "\n";
    } else {
      out << R"(      bind: )"
          << "\n";
      for (auto& channel : spec.outputChannels) {
        out << R"(        - name: ")" << channel.name << "\"\n";
        out << R"(          type: ")" << ChannelSpecHelpers::typeAsString(channel.type) << "\"\n";
      }
    }
    out << R"(      command:)"
        << "\n";
    out << R"(        - shell: true)"
        << "\n";
    out << R"(        - value: )" << execution.args[0] << "\n";
    out << R"(        - arguments:)"
        << "\n";
    out << R"(          - -b)"
        << "\n";
    out << R"(          - --monitoring-backend)"
        << "\n";
    out << R"(          - no-op://)"
        << "\n";
    for (size_t ai = 1; ai < execution.args.size(); ++ai) {
      const char* option = execution.args[ai];
      const char* value = nullptr; // no value by default (i.e. a boolean)
      // If the subsequent option exists and does not start with -, we assume
      // it is an argument to the previous one.
      if (ai + 1 < execution.args.size() && execution.args[ai + 1][0] != '-') {
        value = execution.args[ai + 1];
        ai++;
      }
      if (!option) {
        break;
      }
      // Do not print out channel information
      if (strcmp(option, "--channel-config") == 0) {
        ai += 2;
        continue;
      } else if (strcmp(option, "--control") == 0) {
        continue;
      }
      out << R"(          - )" << option << "\n";
      if (value) {
        out << R"(          - )" << value << "\n";
      }
    }
    out << "\n";
  }
  out << "  workflows:\n";
  out << "    o2-workflow:\n";
  out << "      name: \"o2-workflow-roles\"\n";
  out << "      roles: \"\n";
  for (size_t di = 0; di < specs.size(); ++di) {
    auto& spec = specs[di];
    out << "        - name: \"" << spec.name << "\"\n";
    out << "          connect:\n";
    for (auto& channel : spec.inputChannels) {
      out << R"(          - name: ")" << channel.name << "\"\n";
      // FIXME: Until we get a {{workflow}} placeholder.
      std::string sourceDevice = channel.name;
      sourceDevice.erase(0, 5);
      auto startSuffix = sourceDevice.find_last_of("_to_");
      sourceDevice = sourceDevice.substr(0, startSuffix - 3);
      out << R"(            target: "{{parent}}.)" << sourceDevice << ":" << channel.name << "\"\n";
      out << R"(            type: ")" << ChannelSpecHelpers::typeAsString(channel.type) << "\"\n";
    }
    out << "          task:\n";
    out << "            load: " << spec.name << "\n";
  }
}

} // namespace framework
} // namespace o2
