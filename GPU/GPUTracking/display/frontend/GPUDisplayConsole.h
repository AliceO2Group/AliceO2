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

/// \file GPUDisplayConsole.h
/// \author David Rohr

#ifndef GPUDISPLAYCONSOLE_H
#define GPUDISPLAYCONSOLE_H

#include "GPUDisplayFrontendKeyVars.h"
#include <chrono>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace o2::gpu
{
class GPUDisplayConsole : public GPUDisplayFrontendKeyVars // Text console of the event display (^ key): typed text, history with the output, commands (filled by GPUDisplayFrontend::buildConsoleCommands)
{
 public:
  using consoleArg = std::variant<int64_t, double, std::string>; // Argument of a console command
  GPUDisplayConsole(const std::string& title);                   // First history entry, not selectable

  void handleConsoleKey(uint8_t key);                                         // Key typed while the console is shown
  void handleConsoleCommand(const std::string& cmd);                          // Runs the command named by the first token, error if unknown
  static std::vector<consoleArg> parseConsoleCommand(const std::string& cmd); // Split into arguments, throws std::runtime_error for invalid input
  void consolePrint(const char* format, ...);                                 // printf-like output to the console, indented by 2 spaces in mConsoleHistory
  void consoleInfo(const char* format, const char* text);                     // Output of SetInfo, updates the last line instead for the same message within 2 s
  void consoleAppend(std::string line);                                       // Every other new line of the console
  void consoleLimitHistory(uint32_t historySize);                             // Once per frame: keeps historySize lines when the history reaches twice that
  const std::string& shownText() const { return mConsoleTextCurrent ? mConsoleHistory[mConsoleTextCurrent] : mConsoleText; }

  std::unordered_map<std::string, std::function<void(std::span<const consoleArg>)>> mConsoleCommands; // Console commands by name
  std::string mConsoleInfoLastMessage;                                                                // Format of the SetInfo that wrote the last console line, empty if something else came after it
  std::chrono::steady_clock::time_point mConsoleInfoLastTime;                                         //
  std::vector<std::string> mConsoleHistory;                                                           // Title, then commands entered in the console and their output (indented)
  std::string mConsoleText;                                                                           // Text being typed in the console
  uint32_t mConsoleTextCurrent = 0;                                                                   // Entry of mConsoleHistory shown in the console (selected with up / down, the title entry 0 is not selectable), 0 for mConsoleText
  uint32_t mConsoleTextPos = 0;                                                                       // Cursor position in the shown text (left / right), the character there is the current one
};
} // namespace o2::gpu

#endif
