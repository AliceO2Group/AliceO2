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

/// \file DirectoryLoader.h
/// \brief Loading content of the Folder and returning sorted
/// \author julian.myrcha@cern.ch

#include "EventVisualisationBase/DirectoryLoader.h"
#include <filesystem>
#include <algorithm>
#include <climits>
#include <map>
#include <iostream>
#include <forward_list>
#include <regex>
#include <fairlogger/Logger.h>

using namespace std;
using namespace o2::event_visualisation;

deque<string> DirectoryLoader::load(const std::string& path, const std::string& marker, const std::vector<std::string>& ext)
{
  deque<string> result;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (std::find(ext.begin(), ext.end(), entry.path().extension()) != ext.end()) {
      result.push_back(entry.path().filename());
    }
  }
  // comparison with safety if marker not in the filename (-1+1 gives 0)
  std::sort(result.begin(), result.end(),
            [marker](const std::string& a, const std::string& b) {
              return a.substr(a.find_first_of(marker) + 1) < b.substr(b.find_first_of(marker) + 1);
            });

  return result;
}

bool DirectoryLoader::canCreateNextFile(const std::vector<std::string>& paths, const std::string& marker, const std::vector<std::string>& ext, long long millisec, long capacityAllowed)
{
  deque<string> result;
  std::map<std::string, std::string> fullPath;
  for (const auto& path : paths) {
    try {
      for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (std::find(ext.begin(), ext.end(), entry.path().extension()) != ext.end()) {
          result.push_back(entry.path().filename());
          fullPath[entry.path().filename()] = entry.path();
        }
      }
    } catch (std::filesystem::filesystem_error const& ex) {
      LOGF(info, "filesystem problem: %s", ex.what());
    }
  }

  // comparison with safety if marker not in the filename (-1+1 gives 0)
  std::ranges::sort(result.begin(), result.end(),
                    [marker](const std::string& a, const std::string& b) {
                      return a.substr(a.find_first_of(marker) + 1) > b.substr(b.find_first_of(marker) + 1);
                    });
  unsigned long accumulatedSize = 0L;
  const std::regex delimiter{"_"};
  for (auto const& file : result) {
    std::vector<std::string> c(std::sregex_token_iterator(file.begin(), file.end(), delimiter, -1), {});
    if (std::stoll(c[1]) < millisec) {
      break;
    }
    try {
      accumulatedSize += filesystem::file_size(fullPath[file]);
    } catch (std::filesystem::filesystem_error const& ex) {
      LOGF(info, "problem scanning folder: %s", ex.what());
    }
    if (accumulatedSize > capacityAllowed) {
      return false;
    }
  }
  return true;
}

deque<string> DirectoryLoader::load(const std::vector<std::string>& paths, const std::string& marker, const std::vector<std::string>& ext)
{
  deque<string> result;
  for (const auto& path : paths) {
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
      if (std::find(ext.begin(), ext.end(), entry.path().extension()) != ext.end()) {
        result.push_back(entry.path().filename());
      }
    }
  }
  // comparison with safety if marker not in the filename (-1+1 gives 0)
  std::sort(result.begin(), result.end(),
            [marker](const std::string& a, const std::string& b) {
              return a.substr(a.find_first_of(marker) + 1) < b.substr(b.find_first_of(marker) + 1);
            });

  return result;
}

std::vector<std::string> DirectoryLoader::allFolders(const std::string& location)
{
  auto const pos = location.find_last_of('_');
  std::vector<std::string> folders;
  folders.push_back(location.substr(0, pos) + "_PHYSICS");
  folders.push_back(location.substr(0, pos) + "_COSMICS");
  folders.push_back(location.substr(0, pos) + "_SYNTHETIC");
  return folders;
}

void DirectoryLoader::reduceNumberOfFiles(const std::string& path, const std::deque<std::string>& files, std::size_t filesInFolder)
{
  if (filesInFolder == -1) {
    return; // do not reduce
  }
  const auto items = files.size() - std::min(files.size(), filesInFolder);
  for (int i = 0; i < items; i++) {
    std::remove((path + "/" + files[i]).c_str()); // delete file
  }
}

template <typename TP>
std::time_t to_time_t(TP tp)
{
  using namespace std::chrono;
  auto sctp = time_point_cast<system_clock::duration>(tp - TP::clock::now() + system_clock::now());
  return system_clock::to_time_t(sctp);
}

int DirectoryLoader::getNumberOfFiles(const std::string& path, std::vector<std::string>& ext)
{
  int res = 0;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (std::find(ext.begin(), ext.end(), entry.path().extension()) != ext.end()) {
      res++;
    }
  }
  return res;
}
std::string DirectoryLoader::getLatestFile(const std::string& path, std::vector<std::string>& ext)
{
  std::string oldest_file_name = "";
  std::time_t oldest_file_time = LONG_MAX;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (std::find(ext.begin(), ext.end(), entry.path().extension()) != ext.end()) {
      const auto file_time = entry.last_write_time();
      if (const std::time_t tt = to_time_t(file_time); tt < oldest_file_time) {
        oldest_file_time = tt;
        oldest_file_name = entry.path().filename().string();
      }
    }
  }
  return oldest_file_name;
}

void DirectoryLoader::removeOldestFiles(const std::string& path, std::vector<std::string>& ext, const int remaining)
{
  while (getNumberOfFiles(path, ext) > remaining) {
    LOGF(info, "removing oldest file in folder: %s : %s", path, getLatestFile(path, ext));
    filesystem::remove(path + "/" + getLatestFile(path, ext));
  }
}
