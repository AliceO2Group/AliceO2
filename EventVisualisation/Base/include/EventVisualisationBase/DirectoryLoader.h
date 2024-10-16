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

#ifndef O2EVE_DIRECTORYLOADER_H
#define O2EVE_DIRECTORYLOADER_H

#include <string>
#include <vector>
#include <deque>

namespace o2
{
namespace event_visualisation
{

class DirectoryLoader
{
 private:
  static int getNumberOfFiles(const std::string& path, std::vector<std::string>& ext);
  static std::string getLatestFile(const std::string& path, std::vector<std::string>& ext);

 public:
  static std::deque<std::string> load(const std::string& path, const std::string& marker, const std::vector<std::string>& ext);
  static std::deque<std::string> load(const std::vector<std::string>& paths, const std::string& marker, const std::vector<std::string>& ext);
  static std::vector<std::string> allFolders(const std::string& location);
  static bool canCreateNextFile(const std::vector<std::string>& paths, const std::string& marker, const std::vector<std::string>& ext, long long millisec, long capacityAllowed);
  static void reduceNumberOfFiles(const std::string& path, const std::deque<std::string>& files, std::size_t filesInFolder);
  static void removeOldestFiles(const std::string& path, std::vector<std::string>& ext, int remaining);
};

} // namespace event_visualisation
} // namespace o2

#endif // O2EVE_DIRECTORYLOADER_H
