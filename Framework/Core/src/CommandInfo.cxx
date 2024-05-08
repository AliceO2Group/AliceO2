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

#include "Framework/CommandInfo.h"

#include <sstream>
#include <cassert>
#include <cstring>

namespace o2::framework
{

CommandInfo::CommandInfo(int argc, char* const* argv)
{
  assert(argc > 0);

  std::stringstream commandStream;
  commandStream << argv[0];

  for (size_t ai = 1; ai < argc; ++ai) {
    const char* arg = argv[ai];
    if (strpbrk(arg, "\" ;@&") != nullptr || arg[0] == 0) {
      commandStream << " '" << arg << "'";
    } else if (strpbrk(arg, "'") != nullptr) {
      commandStream << " \"" << arg << "\"";
    } else {
      commandStream << " " << arg;
    }
  }
  command = commandStream.str();
}

void CommandInfo::merge(CommandInfo const& other)
{
  if (!command.empty()) {
    command += " | ";
  }
  command += other.command;
}

} // namespace o2::framework