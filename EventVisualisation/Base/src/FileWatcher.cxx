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

/// \file FileWatcher.h
/// \brief Observing folder for created and removed files - preserving current
/// \author julian.myrcha@cern.ch

#include "EventVisualisationBase/FileWatcher.h"
#include "FairLogger.h"

#include <list>
#include <filesystem>
#include <algorithm>
#include <sys/stat.h>
using namespace std;

namespace o2
{
namespace event_visualisation
{

const char* FileWatcher::mLowGuard = " 0"; /// start guard
const char* FileWatcher::mEndGuard = "~0"; /// stop guard

deque<string> FileWatcher::load(string path)
{
  //LOG(info) << "FileWatcher::load(" << path << ")";
  deque<string> result;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.path().extension() == ".json") {
      result.push_back(entry.path().filename());
    }
  }
  //LOG(info) << result.size();
  return result;
}

FileWatcher::FileWatcher(const string& path)
{
  //LOG(info) << "FileWatcher::FileWatcher(" << path << ")";
  this->mDataFolder = path;
  this->mCurrentFile = mEndGuard;
  this->mFiles.clear();
  this->mFiles.push_front(mLowGuard);
  this->mFiles.push_back(mEndGuard);
  //LOG(info) << "FileWatcher" << this->getSize();
}

void FileWatcher::changeFolder(const string& path)
{
  if (this->mDataFolder == path) {
    return; // the same folder - no action
  }
  this->mDataFolder = path;
  this->mCurrentFile = mEndGuard;
  this->mFiles.clear();
  this->mFiles.push_front(mLowGuard);
  this->mFiles.push_back(mEndGuard);
  this->refresh();
  //LOG(info) << "FileWatcher" << this->getSize();
}

string FileWatcher::nextItem(const string& item) const
{
  if (item == mEndGuard) {
    return mEndGuard;
  }
  return *(std::find(this->mFiles.begin(), this->mFiles.end(), item) + 1);
}

string FileWatcher::prevItem(const string& item) const
{
  if (item == mLowGuard) {
    return mLowGuard;
  }
  return *(std::find(mFiles.begin(), mFiles.end(), item) - 1);
}

string FileWatcher::currentItem() const
{
  if (this->mFiles.size() == 2) { // only guards on the list
    return "";
  }
  if (this->mCurrentFile == mLowGuard) {
    return *(this->mFiles.begin() + 1);
  }
  if (this->mCurrentFile == mEndGuard) {
    return *(this->mFiles.end() - 2);
  }
  return this->mCurrentFile;
}

void FileWatcher::setFirst()
{
  this->mCurrentFile = mLowGuard;
}

void FileWatcher::setLast()
{
  this->mCurrentFile = mEndGuard;
}

void FileWatcher::setNext()
{
  this->mCurrentFile = nextItem(this->mCurrentFile);
}

void FileWatcher::setPrev()
{
  this->mCurrentFile = prevItem(this->mCurrentFile);
}

int FileWatcher::getSize() const
{
  return this->mFiles.size(); // guards
}

int FileWatcher::getPos() const
{
  return std::distance(mFiles.begin(), std::find(mFiles.begin(), mFiles.end(), this->mCurrentFile));
}

bool FileWatcher::refresh()
{
  string previous = this->currentItem();
  LOG(info) << "previous:" << previous;
  LOG(info) << "currentFile:" << this->mCurrentFile;

  this->mFiles = load(this->mDataFolder);
  std::sort(this->mFiles.begin(), this->mFiles.end());
  if (this->mCurrentFile != mEndGuard) {
    if (this->mFiles.empty()) {
      this->mCurrentFile = mEndGuard; // list empty - stick to last element
    } else if (this->mCurrentFile < *(this->mFiles.begin())) {
      this->mCurrentFile = mLowGuard; // lower then first => go to first
    } else {
      auto it = std::find(mFiles.begin(), mFiles.end(), this->mCurrentFile);
      if (it == this->mFiles.end()) {
        this->mCurrentFile = mEndGuard; // not on the list -> go to last element
      }
    }
  }
  //for (auto it = this->mFiles.begin(); it != this->mFiles.end(); ++it) {
  //  LOG(info) << *it;
  //}
  this->mFiles.push_front(mLowGuard);
  this->mFiles.push_back(mEndGuard);

  LOG(info) << "this->mFiles.size() = " << this->mFiles.size();
  LOG(info) << "this->mCurrentFile = " << this->mCurrentFile;
  LOG(info) << "current:" << this->currentItem();
  return previous != this->currentItem();
}

void FileWatcher::setCurrentItem(int no)
{
  this->mCurrentFile = this->mFiles[no];
  LOG(info) << "this->setCurrentItem(" << no << ")";
  LOG(info) << "this->mCurrentFile = " << this->mCurrentFile;
}

std::string FileWatcher::currentFilePath() const
{
  return this->mDataFolder + "/" + this->currentItem();
}

bool FileWatcher::currentFileExist()
{
  struct stat buffer;
  return (stat(this->currentFilePath().c_str(), &buffer) == 0);
}

void FileWatcher::saveCurrentFileToFolder(const string& destinationFolder)
{
  if (!std::filesystem::exists(destinationFolder)) {
    return; // do not specified, where to save
  }
  if (this->mDataFolder == destinationFolder) {
    return; // could not save to yourself
  }
  if (this->currentFileExist()) {
    std::filesystem::path source = this->currentFilePath();
    std::filesystem::path destination = destinationFolder;
    destination /= source.filename();
    std::filesystem::copy_file(source, destination);
  }
}

} // namespace event_visualisation
} // namespace o2
