// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
const char* FileWatcher::mEndGard = "~0";  /// stop guard

deque<string> FileWatcher::load(string path)
{
  LOG(INFO) << "FileWatcher::load(" << path << ")";
  deque<string> result;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.path().extension() == ".json") {
      result.push_back(entry.path().filename());
    }
  }
  LOG(INFO) << result.size();
  return result;
}

FileWatcher::FileWatcher(const string& path)
{
  LOG(INFO) << "FileWatcher::FileWatcher(" << path << ")";
  this->mDataFolder = path;
  this->mCurrentFile = mLowGuard;
  this->mFiles.clear();
  this->mFiles.push_front(mLowGuard);
  this->mFiles.push_back(mEndGard);
  LOG(INFO) << "FileWatcher" << this->getSize();
}

string FileWatcher::nextItem(const string& item) const
{
  if (item == mEndGard) {
    return mEndGard;
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
  if (this->mCurrentFile == mEndGard) {
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
  this->mCurrentFile = mEndGard;
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
  LOG(INFO) << "previous:" << previous;
  LOG(INFO) << "currentFile:" << this->mCurrentFile;

  this->mFiles = load(this->mDataFolder);
  std::sort(this->mFiles.begin(), this->mFiles.end());
  if (this->mCurrentFile != mEndGard) {
    if (this->mFiles.empty()) {
      this->mCurrentFile = mEndGard; // list empty - stick to last element
    } else if (this->mCurrentFile < *(this->mFiles.begin())) {
      this->mCurrentFile = mLowGuard; // lower then first => go to first
    } else {
      auto it = std::find(mFiles.begin(), mFiles.end(), this->mCurrentFile);
      if (it == this->mFiles.end()) {
        this->mCurrentFile = mEndGard; // not on the list -> go to last element
      }
    }
  }
  for (auto it = this->mFiles.begin(); it != this->mFiles.end(); ++it) {
    LOG(INFO) << *it;
  }
  this->mFiles.push_front(mLowGuard);
  this->mFiles.push_back(mEndGard);

  LOG(INFO) << "this->mFiles.size() = " << this->mFiles.size();
  LOG(INFO) << "this->mCurrentFile = " << this->mCurrentFile;
  LOG(INFO) << "current:" << this->currentItem();
  return previous != this->currentItem();
}

void FileWatcher::setCurrentItem(int no)
{
  this->mCurrentFile = this->mFiles[no];
  LOG(INFO) << "this->setCurrentItem(" << no << ")";
  LOG(INFO) << "this->mCurrentFile = " << this->mCurrentFile;
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
} // namespace event_visualisation
} // namespace o2
