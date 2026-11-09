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

/// \file GPUDisplayConsole.cxx
/// \author David Rohr

#include "GPUDisplayConsole.h"
#include "GPULogging.h"
#include <algorithm>
#include <charconv>
#include <stdexcept>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace o2::gpu;

namespace
{
struct consoleArgumentError : public std::runtime_error { // Wrong arguments for an existing command, what(): the error, usage: the argument types
  consoleArgumentError(const std::string& cmd, const std::string& error, const std::string& types) : std::runtime_error(error), command(cmd), usage("[")
  {
    static constexpr std::pair<char, const char*> names[] = {{'s', "string"}, {'i', "int"}, {'d', "double"}, {'a', "any"}, {'+', "[optional]"}};
    for (char t : types) {
      if (const auto* n = std::find_if(std::begin(names), std::end(names), [t](const auto& entry) { return entry.first == t; }); n != std::end(names)) {
        usage += (usage.size() > 1 ? " " : "") + std::string(n->second);
      }
    }
    usage += "]";
  }
  std::string command, usage;
};
} // namespace

GPUDisplayConsole::GPUDisplayConsole(const std::string& title) : mConsoleHistory{title} {}

void GPUDisplayConsole::scroll(int32_t lines)
{
  const int64_t maxScroll = std::max<int64_t>(0, (int64_t)mConsoleHistory.size() - mVisibleLines);
  mScroll = std::clamp<int64_t>((int64_t)mScroll + lines, 0, maxScroll);
}

void GPUDisplayConsole::handleConsoleKey(uint8_t key, bool shift)
{
  if (key == KEY_PAGEUP || key == KEY_PAGEDOWN || (shift && (key == KEY_UP || key == KEY_DOWN))) { // Half a page / one line
    const int32_t lines = key == KEY_PAGEUP || key == KEY_PAGEDOWN ? std::max<int32_t>(1, mVisibleLines / 2) : 1;
    scroll(key == KEY_PAGEUP || key == KEY_UP ? lines : -lines);
    return;
  }
  const uint32_t nHistory = mConsoleHistory.size();
  auto isOutput = [this](uint32_t i) { return mConsoleHistory[i].size() && mConsoleHistory[i][0] == ' '; }; // Indented output of commands, not selectable
  if (key == KEY_UP || key == KEY_DOWN) {                                                                   // Entry 0 is the title, not selectable
    if (key == KEY_UP) {
      uint32_t i = mConsoleTextCurrent ? mConsoleTextCurrent : nHistory;
      while (--i >= 1 && isOutput(i)) {
      }
      mConsoleTextCurrent = i >= 1 ? i : mConsoleTextCurrent;
    } else if (mConsoleTextCurrent) {
      uint32_t i = mConsoleTextCurrent;
      while (++i < nHistory && isOutput(i)) {
      }
      mConsoleTextCurrent = i < nHistory ? i : 0;
    }
    mConsoleTextPos = shownText().size(); // At the end, as when editing the entry starts
    return;
  } else if (key == KEY_ENTER) {
    std::string command = mConsoleTextCurrent ? mConsoleHistory[mConsoleTextCurrent] : mConsoleText;
    command.erase(0, std::min(command.size(), command.find_first_not_of(' '))); // Leading spaces would mark it as output in the history
    mConsoleText.clear();
    mConsoleTextCurrent = mConsoleTextPos = mScroll = 0;
    if (!command.empty()) {
      consoleAppend(command); // Before its output
      handleConsoleCommand(command);
    }
    return;
  } else if (key != KEY_LEFT && key != KEY_RIGHT && key != KEY_BACKSPACE && key != KEY_DELETE && (key < 32 || key >= 127)) { // Other special keys are ignored
    return;
  }
  if (mConsoleTextCurrent) { // Edit a copy of the history entry, with the cursor at its end (left: on its last character), delete then removes nothing
    mConsoleText = mConsoleHistory[mConsoleTextCurrent];
    mConsoleTextCurrent = 0;
    mConsoleTextPos = mConsoleText.size() - (key == KEY_LEFT && mConsoleText.size());
    if (key == KEY_LEFT || key == KEY_RIGHT) {
      return;
    }
  }
  mConsoleTextPos = std::min<uint32_t>(mConsoleTextPos, mConsoleText.size());
  if (key == KEY_LEFT) {
    mConsoleTextPos -= mConsoleTextPos > 0;
  } else if (key == KEY_RIGHT) {
    mConsoleTextPos += mConsoleTextPos < mConsoleText.size();
  } else if (key == KEY_BACKSPACE) {
    if (mConsoleTextPos > 0) {
      mConsoleText.erase(--mConsoleTextPos, 1);
    }
  } else if (key == KEY_DELETE) {
    if (mConsoleTextPos < mConsoleText.size()) {
      mConsoleText.erase(mConsoleTextPos, 1);
    }
  } else {
    mConsoleText.insert(mConsoleTextPos++, 1, (char)key);
  }
}

void GPUDisplayConsole::consolePrint(const char* format, ...)
{
  va_list args, argsCopy;
  va_start(args, format);
  va_copy(argsCopy, args);
  std::string buffer(std::max(0, vsnprintf(nullptr, 0, format, args)), '\0');
  va_end(args);
  vsnprintf(buffer.data(), buffer.size() + 1, format, argsCopy);
  va_end(argsCopy);
  consoleAppend("  " + buffer); // Indented: output, not a command, skipped by up / down
  if (mPrintToTerminal) {
    GPUInfo("%s", buffer.c_str());
  }
}

void GPUDisplayConsole::consoleInfo(const char* format, const char* text)
{
  const auto now = std::chrono::steady_clock::now();
  if (!mConsoleInfoLastMessage.empty() && mConsoleInfoLastMessage == format && now - mConsoleInfoLastTime < std::chrono::seconds(2)) {
    mConsoleHistory.back() = std::string("  ") + text; // Same message again shortly after (animation progress, held keys): update its line
  } else {
    consoleAppend(std::string("  ") + text);
  }
  mConsoleInfoLastMessage = strcmp(format, "%s") ? format : ""; // A plain "%s" can be any message, never merged
  mConsoleInfoLastTime = now;
}

void GPUDisplayConsole::consoleAppend(std::string line)
{
  mConsoleHistory.emplace_back(std::move(line));
  mConsoleInfoLastMessage.clear();
}

void GPUDisplayConsole::consoleLimitHistory(uint32_t historySize)
{
  const size_t size = std::max<uint32_t>(historySize, 2);
  if (mConsoleHistory.size() < 2 * size) {
    return;
  }
  const size_t removed = mConsoleHistory.size() - size;
  std::move(mConsoleHistory.begin() + 1 + removed, mConsoleHistory.end(), mConsoleHistory.begin() + 1); // Keep the title (entry 0)
  mConsoleHistory.resize(size);
  mConsoleTextCurrent = mConsoleTextCurrent > removed ? mConsoleTextCurrent - removed : 0; // Same entry, the end if it was removed
  mConsoleTextPos = std::min<uint32_t>(mConsoleTextPos, shownText().size());
  mScroll = mScroll + 1 >= size ? 0 : mScroll; // Counted from the end: same lines, the end if the lowest shown one was removed
}

// Tokens separated by spaces: text in single / double quotes and characters after a backslash are plain text (the token is a string),
// otherwise a token is int64_t for [+-]digits, double for other decimal numbers (also with exponent), a string otherwise. Throws std::runtime_error for an unterminated quote / trailing backslash.

std::tuple<const GPUDisplayConsole::GPUDisplayConsoleCommand&, std::vector<GPUDisplayConsole::consoleArg>> GPUDisplayConsole::parseConsoleCommand(std::string_view cmd) const
{
  std::vector<std::pair<std::string, bool>> tokens; // Text, plain text (quoted / escaped)
  std::string token;
  bool inToken = false, plainText = false, escape = false;
  char quote = 0;
  auto finishToken = [&]() {
    if (inToken) {
      tokens.emplace_back(std::move(token), plainText);
      token.clear();
      inToken = plainText = false;
    }
  };
  for (char c : cmd) {
    if (escape) {
      token += c;
      escape = false;
    } else if (quote) {
      if (c == quote) {
        quote = 0;
      } else {
        token += c;
      }
    } else if (c == '\\') {
      escape = inToken = plainText = true;
    } else if (c == '"' || c == '\'') {
      quote = c;
      inToken = plainText = true;
    } else if (c == ' ') {
      finishToken();
    } else {
      token += c;
      inToken = true;
    }
  }
  if (quote || escape) {
    throw std::runtime_error(quote ? "unterminated quote" : "backslash at the end");
  }
  finishToken();
  if (tokens.empty()) {
    throw std::runtime_error("empty command");
  }

  const std::string& name = tokens[0].first;
  const auto it = mConsoleCommands.find(name);
  if (it == mConsoleCommands.end()) {
    throw std::runtime_error("unknown command " + name);
  }
  const std::string& types = it->second.args;
  const bool more = types.size() && types.back() == '+';
  const size_t nFixed = types.size() - more, nArgs = tokens.size() - 1;
  if (nArgs < nFixed || (!more && nArgs > nFixed)) {
    throw consoleArgumentError(name, std::to_string(nFixed) + (more ? " or more" : "") + " arguments expected, got " + std::to_string(nArgs), types);
  }
  auto autoType = [](const std::string& text, bool plain) -> consoleArg { // int64_t for [+-]digits, double for other decimal numbers (also with exponent, as saveSettings writes them), string otherwise or if plain text
    const char *begin = text.c_str() + (text[0] == '+' || text[0] == '-'), *end = text.c_str() + text.size();
    const bool number = !plain && begin < end && ((*begin >= '0' && *begin <= '9') || *begin == '.') && std::all_of(begin, end, [](char c) { return (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-'; });
    if (!number) {
      return text;
    }
    int64_t intValue = 0;
    double doubleValue = 0.;
    if (const auto rInt = std::from_chars(begin, end, intValue); rInt.ec == std::errc() && rInt.ptr == end) {
      return text[0] == '-' ? -intValue : intValue;
    } else if (const auto rDouble = std::from_chars(begin, end, doubleValue); rDouble.ec == std::errc() && rDouble.ptr == end) {
      return text[0] == '-' ? -doubleValue : doubleValue; // Also integers out of the int64_t range
    }
    return text;
  };
  std::vector<consoleArg> args;
  for (size_t i = 0; i < nArgs; i++) {
    const auto& [text, plain] = tokens[i + 1];
    const char type = i < nFixed ? types[i] : '+';
    consoleArg value = type == 's' ? consoleArg(text) : autoType(text, plain);
    if (type == 'i' && !std::holds_alternative<int64_t>(value)) {
      throw consoleArgumentError(name, "argument " + std::to_string(i + 1) + " (" + text + ") must be an integer", types);
    } else if (type == 'd') {
      if (std::holds_alternative<int64_t>(value)) {
        value = (double)std::get<int64_t>(value);
      } else if (!std::holds_alternative<double>(value)) {
        throw consoleArgumentError(name, "argument " + std::to_string(i + 1) + " (" + text + ") must be a number", types);
      }
    } else if (type != 's' && type != 'i' && type != 'a' && type != '+') {
      throw std::runtime_error(name + ": invalid argument types " + types);
    }
    args.emplace_back(std::move(value));
  }
  return {it->second, std::move(args)};
}

void GPUDisplayConsole::handleConsoleCommand(std::string_view cmd)
{
  if (cmd.find_first_not_of(' ') == std::string_view::npos) { // Empty line
    return;
  }
  const GPUDisplayConsoleCommand* command = nullptr;
  std::vector<consoleArg> args;
  try {
    auto parsed = parseConsoleCommand(cmd);
    command = &std::get<0>(parsed);
    args = std::move(std::get<1>(parsed));
  } catch (const consoleArgumentError& e) {
    consolePrint("Invalid arguments for %s: %s %s", e.command.c_str(), e.what(), e.usage.c_str());
    return;
  } catch (const std::runtime_error& e) {
    consolePrint("Invalid command: %s", e.what());
    return;
  }
  try { // A failing command must not end the display
    command->func(args);
  } catch (const std::exception& e) {
    consolePrint("Error in command %.*s: %s", (int32_t)cmd.size(), cmd.data(), e.what());
  }
}

void GPUDisplayConsole::runScript(std::string_view script)
{
  if (mScriptDepth >= 16) {
    consolePrint("Scripts nested too deeply");
    return;
  }
  std::vector<ScriptLine> lines;
  for (size_t pos = 0; pos < script.size();) {
    const size_t end = std::min(script.find_first_of("\r\n", pos), script.size());
    if (end > pos) { // Empty lines (consecutive \r / \n) are ignored
      lines.push_back({std::string(script.substr(pos, end - pos)), mScriptDepth + 1});
    }
    pos = end + 1;
  }
  mScriptQueue.insert(mScriptQueue.begin(), lines.begin(), lines.end()); // Before the remaining lines of a script running this one
  if (mScriptDepth == 0) {                                               // From a script, its processScripts loop continues with these lines
    processScripts(false);
  }
}

void GPUDisplayConsole::processScripts(bool newFrame)
{
  if (newFrame) {
    mWaitFrames -= mWaitFrames > 0;
    if (mBenchmarkLeft && --mBenchmarkLeft == 0) {
      const double time = std::chrono::duration<double>(std::chrono::steady_clock::now() - mBenchmarkStart).count();
      consolePrint("benchmark: %u frames in %.3f s, %.2f fps, %.3f ms per frame", mBenchmarkFrames, time, mBenchmarkFrames / time, time * 1000. / mBenchmarkFrames);
    }
  }
  if (mWaitCondition && mWaitCondition()) {
    mWaitCondition = nullptr;
  }
  const uint32_t outerDepth = mScriptDepth;
  for (uint32_t n = 0; n < 100000 && mScriptQueue.size() && !mWaitFrames && !mBenchmarkLeft && !mWaitCondition && std::chrono::steady_clock::now() >= mWaitUntil; n++) { // Limit per call: a script must not block the display
    ScriptLine line = std::move(mScriptQueue.front());
    mScriptQueue.pop_front();
    mScriptDepth = line.depth;
    handleConsoleCommand(line.text);
  }
  mScriptDepth = outerDepth;
}

void GPUDisplayConsole::startBenchmark(uint32_t n)
{
  mBenchmarkFrames = mBenchmarkLeft = n;
  mBenchmarkStart = std::chrono::steady_clock::now();
}

void GPUDisplayConsole::runScriptFile(const std::string& filename)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    consolePrint("Cannot open script file %s", filename.c_str());
    return;
  }
  std::stringstream content;
  content << file.rdbuf();
  runScript(content.str());
}

void GPUDisplayConsole::printCommands(const std::function<void(const std::string&)>& print) const
{
  std::vector<const std::pair<const std::string, GPUDisplayConsoleCommand>*> commands;
  for (const auto& entry : mConsoleCommands) {
    commands.emplace_back(&entry);
  }
  std::sort(commands.begin(), commands.end(), [](const auto* a, const auto* b) { return a->first < b->first; });
  print("Console commands (argument types: s string, i integer, d number, a any, + further arguments):");
  for (const auto* c : commands) {
    print("  " + c->first + (c->second.args.empty() ? "" : " ") + c->second.args + ": " + c->second.description);
  }
}
