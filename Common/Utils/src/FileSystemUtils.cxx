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

} // namespace o2::utils
