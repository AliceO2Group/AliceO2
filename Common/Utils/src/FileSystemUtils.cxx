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

//
// Created by Sandro Wenzel on 04.06.21.
//

#include <CommonUtils/FileSystemUtils.h>
#include <filesystem>
#include <vector>
#include <regex>
#include <iostream>
#include <unistd.h>
#include <fmt/format.h>

namespace o2::utils
{

// Return a vector of file names in the current dir matching the search
// pattern. If searchpattern is empty, all files will be returned. Otherwise
// searchpattern will be treated/parsed as proper regular expression.
std::vector<std::string> listFiles(std::string const& dir, std::string const& searchpattern)
{
  std::vector<std::string> filenames;
  std::string rs = searchpattern.empty() ? ".*" : searchpattern;
  std::regex str_expr(rs);

  for (auto& p : std::filesystem::directory_iterator(dir)) {
    try {
      if (!p.is_directory()) {
        auto fn = p.path().filename().string();
        if (regex_match(fn, str_expr)) {
          filenames.push_back(p.path().string());
        }
      }
    } catch (...) {
      // problem listing some file, just ignore continue
      // with next one
      continue;
    }
  }
  return filenames;
}

std::vector<std::string> listFiles(std::string const& searchpattern)
{
  return listFiles("./", searchpattern);
}

void createDirectoriesIfAbsent(std::string const& path)
{
  if (!path.empty() && !std::filesystem::create_directories(path) && !std::filesystem::is_directory(path)) {
    throw std::runtime_error(fmt::format("Failed to create {} directory", path));
  }
}
// A function to expand string containing shell variables
// to a string in which these vars have been substituted.
// Motivation:: filesystem::exists() does not do this by default
// and I couldn't find information on this. Potentially there is an
// existing solution.
std::string expandShellVarsInFileName(std::string const& input)
{
  std::regex e(R"(\$\{?[a-zA-Z0-9_]*\}?)");
  std::regex e3("[a-zA-Z0-9_]+");
  std::string finalstr;
  std::sregex_iterator iter;
  auto words_end = std::sregex_iterator(); // the end iterator (default)
  auto words_begin = std::sregex_iterator(input.begin(), input.end(), e);

  // check first of all if there is shell variable inside
  if (words_end == words_begin) {
    return input;
  }

  std::string tail;
  for (auto i = words_begin; i != words_end; ++i) {
    std::smatch match = *i;
    // remove ${ and }
    std::smatch m;
    std::string s(match.str());

    if (std::regex_search(s, m, e3)) {
      auto envlookup = getenv(m[0].str().c_str());
      if (envlookup) {
        finalstr += match.prefix().str() + std::string(envlookup);
      } else {
        // in case of non existance we keep the env part unreplaced
        finalstr += match.prefix().str() + "${" + m[0].str().c_str() + "}";
      }
      tail = match.suffix().str();
    }
  }
  finalstr += tail;
  return finalstr;
}

} // namespace o2::utils
