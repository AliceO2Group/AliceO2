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

#include "MermaidHelpers.h"
#include <map>
#include <iostream>
#include <string>

namespace o2
{
namespace framework
{

namespace
{
std::string quote(std::string const& s) { return R"(")" + s + R"(")"; }
} // namespace

/// Helper to dump a set of devices as a mermaid file
void MermaidHelpers::dumpDeviceSpec2Mermaid(std::ostream& out, const std::vector<DeviceSpec>& specs)
{
  out << "flowchart TD\n";
  std::map<std::string, std::string> outputChannel2Device;
  std::map<std::string, unsigned int> outputChannel2Port;

  for (auto& spec : specs) {
    auto id = spec.id;
    out << "    " << id << "\n";
    for (auto&& output : spec.outputChannels) {
      outputChannel2Device.insert(std::make_pair(output.name, id));
      outputChannel2Port.insert(std::make_pair(output.name, output.port));
    }
  }
  for (auto& spec : specs) {
    for (auto& input : spec.inputChannels) {
      auto outputName = input.name;
      out << "    " << outputChannel2Device[outputName] << "-- " << input.port << ":" << outputName << " -->" << spec.id << "\n";
    }
  }
}

} // namespace framework
} // namespace o2
