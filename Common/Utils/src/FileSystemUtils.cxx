// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
    if (!p.is_directory()) {
      auto fn = p.path().filename().string();
      if (regex_match(fn, str_expr)) {
        filenames.push_back(p.path().string());
      }
    }
  }
  return filenames;
}

std::vector<std::string> listFiles(std::string const& searchpattern)
{
  return listFiles("./", searchpattern);
}

} // namespace o2::utils
