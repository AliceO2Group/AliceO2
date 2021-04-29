// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CommonUtils/StringUtils.h"
//#include <sys/stat.h>
//#include <cstdlib>
#include <boost/filesystem.hpp>

using namespace o2::utils;

std::vector<std::string> Str::tokenize(const std::string& src, char delim, bool trimToken)
{
  std::stringstream ss(src);
  std::string token;
  std::vector<std::string> tokens;

  while (std::getline(ss, token, delim)) {
    if (trimToken) {
      trim(token);
    }
    if (!token.empty()) {
      tokens.push_back(std::move(token));
    }
  }
  return tokens;
}

bool Str::pathExists(const std::string_view p)
{
  return boost::filesystem::exists(std::string{p});
}

bool Str::pathIsDirectory(const std::string_view p)
{
  return boost::filesystem::is_directory(std::string{p});
}

std::string Str::getFullPath(const std::string_view p)
{
  return boost::filesystem::canonical(std::string{p}).generic_string();
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
