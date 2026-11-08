// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayFrontendConfigConsole.inc
/// \author David Rohr

#ifndef GPUDISPLAYFRONTENDCONFIGCONSOLE_INC
#define GPUDISPLAYFRONTENDCONFIGCONSOLE_INC

void GPUDisplayFrontend::buildConsoleCommands()
{
  auto& commands = mConsole->mConsoleCommands;
  commands["help"] = [this](std::span<const GPUDisplayConsole::consoleArg> args) {
    if (args.size()) {
      mConsole->consolePrint("help: no arguments expected");
      return;
    }
    std::vector<std::string> names;
    for (const auto& [name, function] : mConsole->mConsoleCommands) {
      names.emplace_back(name);
    }
    std::sort(names.begin(), names.end());
    mConsole->consolePrint("Console commands:");
    for (const auto& name : names) {
      mConsole->consolePrint("  %s", name.c_str());
    }
  };
  commands["clearHistory"] = [this](std::span<const GPUDisplayConsole::consoleArg> args) {
    if (args.size()) {
      mConsole->consolePrint("clearHistory: no arguments expected");
      return;
    }
    mConsole->mConsoleHistory.resize(1); // Keep the title
    mConsole->mConsoleTextCurrent = 0;
    mConsole->mConsoleInfoLastMessage.clear();
  };
}

#endif // GPUDISPLAYFRONTENDCONFIGCONSOLE_INC
