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

///
/// \file    FileProducer.cxx
/// \author julian.myrcha@cern.ch

#include "EveWorkflow/FileProducer.h"
#include "CommonUtils/FileSystemUtils.h"

#include <deque>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <climits>
#include <fmt/core.h>

using namespace o2::event_visualisation;

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;

std::deque<std::string> FileProducer::load(const std::string& path)
{
  std::deque<std::string> result;

  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.path().extension() == ".json") {
      result.push_back(entry.path().filename());
    }
  }
  return result;
}

FileProducer::FileProducer(const std::string& path, int filesInFolder, const std::string& name)
{
  this->mFilesInFolder = filesInFolder;
  this->mPath = path;
  this->mName = name;
  o2::utils::createDirectoriesIfAbsent(path); // create folder if not exists (fails if no rights)
}

std::string FileProducer::newFileName() const
{
  auto millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

  char hostname[_POSIX_HOST_NAME_MAX];
  gethostname(hostname, _POSIX_HOST_NAME_MAX);

  auto pid = getpid();

  auto result = fmt::format(this->mName, fmt::arg("hostname", hostname), fmt::arg("pid", pid), fmt::arg("timestamp", millisec_since_epoch));

  auto files = this->load(this->mPath);
  std::sort(files.begin(), files.end());

  while (files.size() >= this->mFilesInFolder) {
    auto front = files.front();
    files.pop_front();
    std::remove((this->mPath + "/" + front).c_str()); // delete file
  }
  return this->mPath + "/" + result;
}
