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
#include <deque>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

namespace o2::gpu
{
class GPUDisplayConsole : public GPUDisplayFrontendKeyVars // Text console of the event display (^ key): typed text, history with the output, commands (filled by GPUDisplayFrontend::buildConsoleCommands)
{
 public:
  using consoleArg = std::variant<int64_t, double, std::string>; // Argument of a console command
  struct GPUDisplayConsoleCommand {
    std::function<void(std::span<const consoleArg>)> func;
    std::string args;        // One character per argument: s string, i int64_t, d double (also from an integer), a any (type from the text), + at the end: any further arguments (type from the text)
    std::string description; // For help
  };
  GPUDisplayConsole(const std::string& title); // First history entry, not selectable

  void handleConsoleKey(uint8_t key, bool shift);  // Key typed while the console is shown
  void scroll(int32_t lines);                      // Scrolls the history up (> 0) or down, limited to its lines
  void handleConsoleCommand(std::string_view cmd); // Runs the command named by the first token, error if unknown
  void runScript(std::string_view script);         // Queues the lines (separated by \n / \r) as console commands before the remaining script lines, runs them up to a wait
  void runScriptFile(const std::string& filename); // runScript with the content of the file
  void processScripts(bool newFrame);              // Runs queued script lines until a wait; newFrame: called once per frame, counts the frames of wait / benchmark
  void waitFrames(uint32_t n) { mWaitFrames = n; } // The following script lines run after n frames
  void waitSeconds(double t) { mWaitUntil = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(t)); }
  void waitFor(std::function<bool()> condition) { mWaitCondition = std::move(condition); }                              // The following script lines run once the condition is true
  void startBenchmark(uint32_t n);                                                                                      // Measures the next n frames, the following script lines run afterwards
  void printCommands(const std::function<void(const std::string&)>& print) const;                                       // List of the commands with argument types and description, one line per call
  std::tuple<const GPUDisplayConsoleCommand&, std::vector<consoleArg>> parseConsoleCommand(std::string_view cmd) const; // Command and its arguments with the types it requests, throws std::runtime_error for syntax errors, unknown commands, wrong arguments
  void consolePrint(const char* format, ...);                                                                           // printf-like output to the console, indented by 2 spaces in mConsoleHistory
  void consoleInfo(const char* format, const char* text);                                                               // Output of SetInfo, updates the last line instead for the same message within 2 s
  void consoleAppend(std::string line);                                                                                 // Every other new line of the console
  void consoleLimitHistory(uint32_t historySize);                                                                       // Once per frame: keeps historySize lines when the history reaches twice that
  const std::string& shownText() const { return mConsoleTextCurrent ? mConsoleHistory[mConsoleTextCurrent] : mConsoleText; }

  std::unordered_map<std::string, GPUDisplayConsoleCommand> mConsoleCommands; // Console commands by name, filled by GPUDisplayFrontend::buildConsoleCommands
  struct ScriptLine {
    std::string text;
    uint32_t depth; // Nesting of runScript
  };
  std::deque<ScriptLine> mScriptQueue; // Script lines not yet run
  uint32_t mScriptDepth = 0;           // Nesting of the script line running now
  uint32_t mWaitFrames = 0;
  std::chrono::steady_clock::time_point mWaitUntil;
  std::function<bool()> mWaitCondition;
  uint32_t mBenchmarkFrames = 0, mBenchmarkLeft = 0;
  std::chrono::steady_clock::time_point mBenchmarkStart;
  bool mPrintToTerminal = false;                              // consolePrint also with GPUInfo (consolePrintToTerminal)
  std::string mConsoleInfoLastMessage;                        // Format of the SetInfo that wrote the last console line, empty if something else came after it
  std::chrono::steady_clock::time_point mConsoleInfoLastTime; //
  std::vector<std::string> mConsoleHistory;                   // Title, then commands entered in the console and their output (indented)
  std::string mConsoleText;                                   // Text being typed in the console
  uint32_t mConsoleTextCurrent = 0;                           // Entry of mConsoleHistory shown in the console (selected with up / down, the title entry 0 is not selectable), 0 for mConsoleText
  uint32_t mScroll = 0;                                       // History lines scrolled up (the lowest ones not shown), reset by enter
  uint32_t mVisibleLines = 1;                                 // History lines fitting into the console, set when it is drawn
  uint32_t mConsoleTextPos = 0;                               // Cursor position in the shown text (left / right), the character there is the current one
};
} // namespace o2::gpu

#endif
