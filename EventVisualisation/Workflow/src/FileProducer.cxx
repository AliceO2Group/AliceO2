// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    FileProducer.cxx
/// \author julian.myrcha@cern.ch

#include "EveWorkflow/FileProducer.h"
#include <deque>
#include <iostream>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <algorithm>

using namespace std;
using std::cout;
using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

std::deque<std::string> FileProducer::load(const std::string& path)
{
  deque<string> result;

  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.path().extension() == ".json") {
      result.push_back(entry.path().filename());
    }
  }
  return result;
}

FileProducer::FileProducer(const std::string& path, const std::string& name, int filesInFolder)
{
  this->mFilesInFolder = filesInFolder;
  this->mPath = path;
  this->mName = name;
}

std::string FileProducer::newFileName() const
{
  string pholder = "{}";
  string result = this->mName;
  auto millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  string stamp = std::to_string(millisec_since_epoch);
  result.replace(result.find(pholder), pholder.length(), stamp);
  auto files = this->load(this->mPath);
  std::sort(files.begin(), files.end());
  while (files.size() > this->mFilesInFolder) {
    string front = files.front();
    files.pop_front();
    std::remove((this->mPath + "/" + front).c_str()); // delete file
  }
  return this->mPath + "/" + result;
}
