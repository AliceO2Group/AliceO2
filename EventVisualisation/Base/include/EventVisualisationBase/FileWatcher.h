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

#ifndef WATCHER_FILEWATCHER_H
#define WATCHER_FILEWATCHER_H

#include <string>
#include <deque>

namespace o2
{
namespace event_visualisation
{

class FileWatcher
{
  static const char* mLowGuard;   ///< "artificial" file name meaning "on the first" (guard)
  static const char* mEndGuard;   ///< "artificial" file name meaning "on the last"  (guard)
  std::deque<std::string> mFiles; ///< sorted file list with guards at the beginning and end
  std::string nextItem(const std::string& item) const;
  std::string prevItem(const std::string& item) const;
  std::string mDataFolder;  ///< folder being observed
  std::string mCurrentFile; ///< "current" file name
  bool currentFileExist();

 public:
  FileWatcher(const std::string& path);
  void changeFolder(const std::string& path);                         ///< switch to observe other folder
  void saveCurrentFileToFolder(const std::string& destinationFolder); ///< copies
  int getSize() const; ///< include guards (so >=2 )
  int getPos() const;  ///< include guards -> 0 points to mLowGuard
  void setFirst();
  void setLast();
  void setNext();
  void setPrev();
  void rollToNext();                   ///< round robin next item
  bool refresh();                      ///< reads folder content, updates current if points to not existing file
  std::string currentItem() const;     ///< name of the file (without path) but guards replaced with file names
  void setCurrentItem(int no);         ///< sets using index
  std::string currentFilePath() const; ///< name of the file (with path) but guards replaced with file names
};

} // namespace event_visualisation
} // namespace o2

#endif //WATCHER_FILEWATCHER_H
