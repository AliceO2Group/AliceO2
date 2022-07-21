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
#include <FairLogger.h>

using namespace std;
using namespace o2::event_visualisation;

deque<string> DirectoryLoader::load(const std::string& path, const std::string& marker, const std::string& ext)
{
  deque<string> result;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.path().extension() == ext) {
      result.push_back(entry.path().filename());
    }
  }
  // comparison with safety if marker not in the filename (-1+1 gives 0)
  std::sort(result.begin(), result.end(),
            [marker](std::string a, std::string b) {
              return a.substr(a.find_first_of(marker) + 1) < b.substr(b.find_first_of(marker) + 1);
            });

  return result;
}

void DirectoryLoader::reduceNumberOfFiles(const std::string& path, const std::deque<std::string>& files, std::size_t filesInFolder)
{
  if (filesInFolder == -1) {
    return; // do not reduce
  }
  int items = files.size() - std::min(files.size(), filesInFolder);
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

int DirectoryLoader::getNumberOfFiles(std::string& path, std::string& ext)
{
  int res = 0;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.path().extension() == ext) {
      res++;
    }
  }
  return res;
}
std::string DirectoryLoader::getLatestFile(std::string& path, std::string& ext)
{
  std::string oldest_file_name = "";
  std::time_t oldest_file_time = LONG_MAX;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.path().extension() == ext) {
      auto file_time = entry.last_write_time();
      std::time_t tt = to_time_t(file_time);
      if (tt < oldest_file_time) {
        oldest_file_time = tt;
        oldest_file_name = entry.path().filename();
      }
    }
  }
  return oldest_file_name;
}

void DirectoryLoader::removeOldestFiles(std::string& path, std::string ext, int remaining)
{
  while (getNumberOfFiles(path, ext) > remaining) {
    LOG(info) << "removing oldest file in folder: " << path << " : " << getLatestFile(path, ext);
    filesystem::remove(path + "/" + getLatestFile(path, ext));
  }
}