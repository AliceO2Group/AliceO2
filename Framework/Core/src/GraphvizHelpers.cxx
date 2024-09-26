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

#include "GraphvizHelpers.h"
#include <map>
#include <iostream>
#include <string>

namespace o2::framework
{

namespace
{
std::string quote(std::string const& s) { return R"(")" + s + R"(")"; }
} // namespace

/// Helper to dump a workflow as a graphviz file
void GraphvizHelpers::dumpDataProcessorSpec2Graphviz(std::ostream& out, const std::vector<DataProcessorSpec>& specs,
                                                     std::vector<std::pair<int, int>> const& edges)
{
  out << "digraph structs {\n";
  out << "  node[shape=record]\n";
  for (auto& spec : specs) {
    std::string labels = " xlabel=\"";
    for (auto& label : spec.labels) {
      labels += labels == " xlabel=\"" ? "" : ",";
      labels += label.value;
    }
    labels += "\"";
    out << fmt::format("  \"{}\" [label=\"{}\"{}];\n", spec.name, spec.name, labels != " xlabel=\"\"" ? labels : "");
  }
  for (auto& e : edges) {
    out << fmt::format("  \"{}\" -> \"{}\"\n", specs[e.first].name, specs[e.second].name);
  }
  out << "}\n";
}

/// Helper to dump a set of devices as a graphviz file
void GraphvizHelpers::dumpDeviceSpec2Graphviz(std::ostream& out, const std::vector<DeviceSpec>& specs)
{
  out << R"GRAPHVIZ(digraph structs {
  node[shape=record]
)GRAPHVIZ";
  std::map<std::string, std::string> outputChannel2Device;
  std::map<std::string, unsigned int> outputChannel2Port;

  for (auto& spec : specs) {
    auto id = spec.id;
    out << "  " << quote(id) << R"( [label="{{)";
    bool firstInput = true;
    for (auto&& input : spec.inputChannels) {
      if (firstInput == false) {
        out << "|";
      }
      firstInput = false;
      out << "<" << input.name << ">" << input.name;
    }
    out << "}|";
    auto totalChannels = spec.inputChannels.size() +
                         spec.outputChannels.size();
    out << id << "(" << totalChannels << ")";
    out << "|{";
    bool firstOutput = true;
    for (auto&& output : spec.outputChannels) {
      outputChannel2Device.insert(std::make_pair(output.name, id));
      outputChannel2Port.insert(std::make_pair(output.name, output.port));
      if (firstOutput == false) {
        out << "|";
      }
      firstOutput = false;
      out << "<" << output.name << ">" << output.name;
    }
    out << R"(}}"];)"
        << "\n";
  }
  for (auto& spec : specs) {
    for (auto& input : spec.inputChannels) {
      // input and output name are now the same
      auto outputName = input.name;
      out << "  " << quote(outputChannel2Device[outputName]) << ":" << quote(outputName) << "-> " << quote(spec.id)
          << ":" << quote(input.name) << R"( [label=")" << input.port << R"(")"
          << "]\n";
    }
  }
  out << "}\n";
}

} // namespace o2::framework
