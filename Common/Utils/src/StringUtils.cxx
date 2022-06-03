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

// generate random string of given lenght, suitable for file names
std::string Str::getRandomString(int lenght)
{
  auto nextAllowed = []() {
    constexpr char chars[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    constexpr size_t L = sizeof(chars) - 1;
    return chars[std::rand() % L];
  };
  std::string str(lenght, 0);
  std::generate_n(str.begin(), lenght, nextAllowed);
  return str;
}

bool Str::pathExists(const std::string_view p)
{
  return std::filesystem::exists(std::string{p});
}

bool Str::pathIsDirectory(const std::string_view p)
{
  return std::filesystem::is_directory(std::string{p});
}

std::string Str::getFullPath(const std::string_view p)
{
  return std::filesystem::canonical(std::string{p}).string();
}

std::string Str::rectifyDirectory(const std::string_view p)
{
  std::string dir(p);
  if (dir.empty() || dir == "none") {
    dir = "";
  } else {
    dir = getFullPath(dir);
    if (!pathIsDirectory(dir)) {
      throw std::runtime_error(fmt::format("{:s} is not an accessible directory", dir));
    } else {
      dir += '/';
    }
  }
  return dir;
}

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
