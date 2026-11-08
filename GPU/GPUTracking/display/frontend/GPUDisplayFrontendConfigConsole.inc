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
  using args = std::span<const GPUDisplayConsole::consoleArg>;
  auto& commands = mConsole->mConsoleCommands;
  commands["help"] = {[this](args) { mConsole->printCommands([this](const std::string& line) { mConsole->consolePrint("%s", line.c_str()); }); }, "", "List available console commands"};
  commands["clearHistory"] = {[this](args) {
                                mConsole->mConsoleHistory.resize(1); // Keep the title
                                mConsole->mConsoleTextCurrent = 0;
                                mConsole->mConsoleInfoLastMessage.clear();
                              },
                              "", "Clear the console history"};
  commands["runScript"] = {[this](args a) { mConsole->runScriptFile(std::get<std::string>(a[0])); }, "s", "Execute script file"};
}

#endif // GPUDISPLAYFRONTENDCONFIGCONSOLE_INC
