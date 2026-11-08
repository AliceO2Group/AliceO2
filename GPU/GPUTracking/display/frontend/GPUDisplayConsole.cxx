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
std::vector<GPUDisplayConsole::consoleArg> GPUDisplayConsole::parseConsoleCommand(const std::string& cmd)
{
  std::vector<consoleArg> retVal;
  std::string token;
  bool inToken = false, plainText = false, escape = false;
  char quote = 0;
  auto finishToken = [&]() {
    if (!inToken) {
      return;
    }
    const char* begin = token.c_str() + (token[0] == '+' || token[0] == '-');
    const size_t nDigits = std::count_if(begin, token.c_str() + token.size(), [](char c) { return c >= '0' && c <= '9'; });
    const size_t nDots = std::count(begin, token.c_str() + token.size(), '.');
    const bool number = !plainText && nDigits && nDigits + nDots == (size_t)(token.c_str() + token.size() - begin);
    int64_t intValue = 0;
    if (number && nDots == 0 && std::from_chars(begin, token.c_str() + token.size(), intValue).ec == std::errc()) {
      retVal.emplace_back(token[0] == '-' ? -intValue : intValue);
    } else if (number && nDots <= 1) {
      retVal.emplace_back(std::stod(token)); // Also integers out of the int64_t range
    } else {
      retVal.emplace_back(token);
    }
    token.clear();
    inToken = plainText = false;
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
  return retVal;
}

void GPUDisplayConsole::handleConsoleCommand(const std::string& cmd)
{
  std::vector<consoleArg> args;
  try {
    args = parseConsoleCommand(cmd);
  } catch (const std::runtime_error& e) {
    consolePrint("Invalid command: %s", e.what());
    return;
  }
  if (args.empty()) {
    return;
  }
  const std::string* name = std::get_if<std::string>(&args[0]);
  const auto it = name ? mConsoleCommands.find(*name) : mConsoleCommands.end();
  if (it == mConsoleCommands.end()) {
    consolePrint("Unknown command: %s", cmd.c_str());
    return;
  }
  it->second(std::span(args).subspan(1));
}
