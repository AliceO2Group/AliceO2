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

#include "CommonUtils/StringUtils.h"
#include <cstdlib>
#include <filesystem>
#ifndef GPUCA_STANDALONE
#include <TGrid.h>
#include <fmt/format.h>
#endif
#include <unistd.h>

using namespace o2::utils;

std::vector<std::string> Str::tokenize(const std::string& src, char delim, bool trimToken, bool skipEmpty)
{
  std::stringstream ss(src);
  std::string token;
  std::vector<std::string> tokens;

  while (std::getline(ss, token, delim)) {
    if (trimToken) {
      trim(token);
    }
    if (!token.empty() || !skipEmpty) {
      tokens.push_back(std::move(token));
    }
  }
  return tokens;
}

std::vector<std::string> Str::tokenize(const std::string& src, const std::string& delim, bool trimToken, bool skipEmpty)
{
  std::string inptStr{src};
  char* input = inptStr.data();
  auto mystrtok = [&]() -> char* {
    input += std::strspn(input, delim.c_str());
    if (*input == '\0') {
      return nullptr;
    }
    char* const token = input;
    input += std::strcspn(input, delim.c_str());
    if (*input != '\0') {
      *input++ = '\0';
    }
    return token;
  };
  std::vector<std::string> tokens;
  while (*input != '\0') {
    std::string token = mystrtok();
    if (trimToken) {
      trim(token);
    }
    if (!token.empty() || !skipEmpty) {
      tokens.push_back(std::move(token));
    }
  }
  return tokens;
}

// replace all occurencies of from by to, return count
int Str::replaceAll(std::string& s, const std::string& from, const std::string& to)
{
  int count = 0;
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.length(), to);
    pos += to.length(); // Handles case where 'to' is a substring of 'from'
    count++;
  }
  return count;
}

// generate random string of given lenght, suitable for file names
std::string Str::getRandomString(int lenght)
{
  int pid = (int)getpid();
  auto nextAllowed = [pid]() {
    constexpr char chars[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    constexpr size_t L = sizeof(chars) - 1;
    int rn = std::rand() | pid;
    return chars[rn % L];
  };
  std::string str(lenght, 0);
  std::generate_n(str.begin(), lenght, nextAllowed);
  return str;
}

bool Str::pathExists(const std::string_view p)
{
  return p.compare(0, 8, "alien://") ? std::filesystem::exists(std::string{p}) : true; // we don't validate alien paths
}

bool Str::pathIsDirectory(const std::string_view p)
{
  return std::filesystem::is_directory(std::string{p});
}

std::string Str::getFullPath(const std::string_view p)
{
  return std::filesystem::canonical(std::string{p}).string();
}

#ifndef GPUCA_STANDALONE
std::string Str::rectifyDirectory(const std::string_view p)
{
  std::string dir(p);
  if (dir.empty() || dir == "none") {
    dir = "";
  } else {
    if (p.compare(0, 8, "alien://") == 0) {
      if (!gGrid && !TGrid::Connect("alien://")) {
        throw std::runtime_error(fmt::format("failed to initialize alien for {}", dir));
      }
      // for root or raw files do not treat as directory
      if (dir.back() != '/' && !(endsWith(dir, ".root") || endsWith(dir, ".raw") || endsWith(dir, ".tf"))) {
        dir += '/';
      }
    } else {
      dir = getFullPath(dir);
      if (!pathIsDirectory(dir)) {
        throw std::runtime_error(fmt::format("{} is not an accessible directory", dir));
      } else {
        if (dir.back() != '/') {
          dir += '/';
        }
      }
    }
  }
  return dir;
}
#endif

// Create unique non-existing path name starting with prefix. Loose equivalent of boost::filesystem::unique_path()
// The prefix can be either existing directory or just a string to add in front of the random part
// in absence of such a function in std::filesystem
std::string Str::create_unique_path(const std::string_view prefix, int length)
{
  std::string path;
  bool needSlash = pathIsDirectory(prefix) && !prefix.empty() && prefix.back() != '/';
  do {
    path = concat_string(prefix, needSlash ? "/" : "", getRandomString(length));
  } while (pathExists(path));

  return path;
}
