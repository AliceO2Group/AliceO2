// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "GraphvizHelpers.h"
#include <map>
#include <iostream>
#include <string>

namespace o2 {
namespace framework {

/// Helper to dump a workflow as a graphviz file
void
dumpDataProcessorSpec2Graphviz(std::ostream &out, const std::vector<DataProcessorSpec> &specs)
{
  out << "digraph structs {\n";
  out << "  node[shape=record]\n";
  for (auto &spec : specs) {
    out << R"(  struct [label=")" << spec.name << R"("];)" << "\n";
  }
  out << "}\n";
}

/// Helper to dump a set of devices as a graphviz file
void
dumpDeviceSpec2Graphviz(std::ostream &out, const std::vector<DeviceSpec> &specs)
{
  out << R"GRAPHVIZ(digraph structs {
  node[shape=record]
)GRAPHVIZ";
  std::map<std::string, std::string> outputChannel2Device;
  std::map<std::string, unsigned int> outputChannel2Port;

  for (auto &spec : specs) {
    auto id = spec.id;
    std::replace(id.begin(), id.end(), '-', '_'); // replace all 'x' to 'y'
    out << "  " << id << R"( [label="{{)";
    bool firstInput = true;
    for (auto && input : spec.channels) {
      if (input.type != Sub) {
        continue;
      }
      if (firstInput == false) {
        out << "|";
      }
      firstInput = false;
      out << "<" << input.name << ">" << input.name;
    }
    out << "}|";
    out << id << "(" << spec.channels.size() << ")";
    out << "|{";
    bool firstOutput = true;
    for (auto && output : spec.channels) {
      outputChannel2Device.insert(std::make_pair(output.name, id));
      outputChannel2Port.insert(std::make_pair(output.name, output.port));
      if (output.type != Pub) {
        continue;
      }
      if (firstOutput == false) {
        out << "|";
      }
      firstOutput = false;
      out <<  "<" << output.name << ">" << output.name;
    }
    out << R"(}}"];)" << "\n";
  }
  for (auto &spec : specs) {
    for (auto &channel: spec.channels) {
      auto id = spec.id;
      std::replace(id.begin(), id.end(), '-', '_'); // replace all 'x' to 'y'
      // If this is an output, we do not care for now.
      // FIXME: make sure that all the outputs are sinked by something.
      if (channel.type == Pub) {
        continue;
      }
      auto outputName = channel.name;
      outputName.erase(0, 3);
      out << "  " << outputChannel2Device[outputName] << ":" << outputName
                  << "-> "
                  << id << ":" << channel.name
                  << R"( [label=")" << channel.port << R"(")"
                  << "]\n";
    }
  }
  out << "}\n";
}

} // namespace framework
} // namespace o2
