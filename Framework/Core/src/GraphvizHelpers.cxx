// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
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
dumpDataProcessorSpec2Graphviz(const std::vector<DataProcessorSpec> &specs)
{
  std::cout << "digraph structs {\n";
  std::cout << "   node[shape=record]\n";
  for (auto &spec : specs) {
    std::cout << "  struct [label=\"" << spec.name << "\"];\n";
  }
  std::cout << "}\n";
}

/// Helper to dump a set of devices as a graphviz file
void
dumpDeviceSpec2Graphviz(const std::vector<DeviceSpec> &specs)
{
  std::cout << R"GRAPHVIZ(
    digraph structs {
    node[shape=record]
  )GRAPHVIZ";
  std::map<std::string, std::string> outputChannel2Device;
  std::map<std::string, unsigned int> outputChannel2Port;

  for (auto &spec : specs) {
    auto id = spec.id;
    std::replace(id.begin(), id.end(), '-', '_'); // replace all 'x' to 'y'
    std::cout << "   " << id << " [label=\"{{";
    bool firstInput = true;
    for (auto && input : spec.channels) {
      if (input.type != Sub) {
        continue;
      }
      if (firstInput == false) {
        std::cout << "|";
      }
      firstInput = false;
      std::cout << "<" << input.name << ">" << input.name;
    }
    std::cout << "}|";
    std::cout << id << "(" << spec.channels.size() << ")";
    std::cout << "|{";
    bool firstOutput = true;
    for (auto && output : spec.channels) {
      outputChannel2Device.insert(std::make_pair(output.name, id));
      outputChannel2Port.insert(std::make_pair(output.name, output.port));
      if (output.type != Pub) {
        continue;
      }
      if (firstOutput == false) {
        std::cout << "|";
      }
      firstOutput = false;
      std::cout <<  "<" << output.name << ">" << output.name;
    }
    std::cout << "}}\"];\n";
  }
  for (auto &spec : specs) {
    for (auto &channel: spec.channels) {
      auto id = spec.id;
      std::replace(id.begin(), id.end(), '-', '_'); // replace all 'x' to 'y'
      // If this is an output, we do not care for now.
      // FIXME: make sure that all the outputs are sinked by something.
      if (channel.type == Pub)
        continue;
      auto outputName = channel.name;
      outputName.erase(0, 3);
      std::cout << outputChannel2Device[outputName] << ":" << outputName
                << "-> "
                << id << ":" << channel.name
                << " [label=\"" << channel.port << "\""
              //  << " taillabel=\"" << outputChannel2Port[outputName] << "\""
                << "]\n";
    }
  }
  std::cout << "}\n";
}

} // namespace framework
} // namespace o2
