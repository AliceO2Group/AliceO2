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
#include <algorithm>
#include <charconv>
#include <stdexcept>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace o2::gpu;

GPUDisplayConsole::GPUDisplayConsole(const std::string& title) : mConsoleHistory{title} {}

void GPUDisplayConsole::handleConsoleKey(uint8_t key)
{
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
    mConsoleTextPos = 0;
    return;
  } else if (key == KEY_ENTER) {
    std::string command = mConsoleTextCurrent ? mConsoleHistory[mConsoleTextCurrent] : mConsoleText;
    command.erase(0, std::min(command.size(), command.find_first_not_of(' '))); // Leading spaces would mark it as output in the history
    mConsoleText.clear();
    mConsoleTextCurrent = mConsoleTextPos = 0;
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
  char buffer[1024];
  va_list args;
  va_start(args, format);
  vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  consoleAppend(std::string("  ") + buffer); // Indented: output, not a command, skipped by up / down
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
  mConsoleTextCurrent = mConsoleTextCurrent > removed ? mConsoleTextCurrent - removed : 0;
}

// Tokens separated by spaces: text in single / double quotes and characters after a backslash are plain text (the token is a string),
// otherwise a token is int64_t for [+-]digits, double for [+-]digits with one '.', a string otherwise. Throws std::runtime_error for an unterminated quote / trailing backslash.

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
    throw std::runtime_error(name + ": " + std::to_string(nFixed) + (more ? " or more" : "") + " arguments expected, got " + std::to_string(nArgs));
  }
  auto autoType = [](const std::string& text, bool plain) -> consoleArg { // int64_t for [+-]digits, double for [+-]digits with one '.', string otherwise or if plain text
    const char *begin = text.c_str() + (text[0] == '+' || text[0] == '-'), *end = text.c_str() + text.size();
    const size_t nDigits = std::count_if(begin, end, [](char c) { return c >= '0' && c <= '9'; });
    const size_t nDots = std::count(begin, end, '.');
    const bool number = !plain && nDigits && nDigits + nDots == (size_t)(end - begin);
    int64_t intValue = 0;
    if (number && nDots == 0 && std::from_chars(begin, end, intValue).ec == std::errc()) {
      return text[0] == '-' ? -intValue : intValue;
    } else if (number && nDots <= 1) {
      return std::stod(text); // Also integers out of the int64_t range
    }
    return text;
  };
  std::vector<consoleArg> args;
  for (size_t i = 0; i < nArgs; i++) {
    const auto& [text, plain] = tokens[i + 1];
    const char type = i < nFixed ? types[i] : '+';
    consoleArg value = type == 's' ? consoleArg(text) : autoType(text, plain);
    if (type == 'i' && !std::holds_alternative<int64_t>(value)) {
      throw std::runtime_error(name + ": argument " + std::to_string(i + 1) + " (" + text + ") must be an integer");
    } else if (type == 'd') {
      if (std::holds_alternative<int64_t>(value)) {
        value = (double)std::get<int64_t>(value);
      } else if (!std::holds_alternative<double>(value)) {
        throw std::runtime_error(name + ": argument " + std::to_string(i + 1) + " (" + text + ") must be a number");
      }
    } else if (type != 's' && type != 'i' && type != '+') {
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
  } catch (const std::runtime_error& e) {
    consolePrint("Invalid command: %s", e.what());
    return;
  }
  command->func(args);
}

void GPUDisplayConsole::runScript(std::string_view script)
{
  if (mScriptDepth >= 16) {
    consolePrint("Scripts nested too deeply");
    return;
  }
  mScriptDepth++;
  for (size_t pos = 0; pos < script.size();) {
    const size_t end = std::min(script.find_first_of("\r\n", pos), script.size());
    handleConsoleCommand(script.substr(pos, end - pos)); // Empty lines (consecutive \r / \n) are ignored
    pos = end + 1;
  }
  mScriptDepth--;
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

void GPUDisplayConsole::printCommands(const std::function<void(const std::string&)>& print)
{
  std::vector<std::string> names;
  for (const auto& [name, command] : mConsoleCommands) {
    names.emplace_back(name);
  }
  std::sort(names.begin(), names.end());
  print("Console commands (argument types: s string, i integer, d number, + further arguments):");
  for (const auto& name : names) {
    const auto& command = mConsoleCommands[name];
    print("  " + name + (command.args.empty() ? "" : " ") + command.args + ": " + command.description);
  }
}
