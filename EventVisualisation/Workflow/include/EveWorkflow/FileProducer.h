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
/// \file    FileProducer.h
/// \author julian.myrcha@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_WORKFLOW_FILEPRODUCER_H
#define ALICE_O2_EVENTVISUALISATION_WORKFLOW_FILEPRODUCER_H

#include <string>
#include <deque>

namespace o2
{
namespace event_visualisation
{
class FileProducer
{
 private:
  size_t mFilesInFolder;
  std::string mPath;
  std::string mName;
  std::string mExt;

 public:
  explicit FileProducer(const std::string& path, const std::string& ext, int filesInFolder = -1,
                        const std::string& name = "tracks_{timestamp}_{hostname}_{pid}{ext}");

  [[nodiscard]] std::string newFileName() const;
};

} // namespace event_visualisation
} // namespace o2

#endif // ALICE_O2_EVENTVISUALISATION_WORKFLOW_FILEPRODUCER_H
