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
/// \file    FileProducer.h
/// \author julian.myrcha@cern.ch

#ifndef FILEPRODUCER_FILEPRODUCER_H
#define FILEPRODUCER_FILEPRODUCER_H

#include <string>
#include <deque>

class FileProducer
{
 private:
  static std::deque<std::string> load(const std::string& path);
  size_t mFilesInFolder;
  std::string mPath;
  std::string mName;

 public:
  explicit FileProducer(const std::string& path, const std::string& name = "tracks{}.json", int filesInFolder = 10);
  [[nodiscard]] std::string newFileName() const;
};

#endif //FILEPRODUCER_FILEPRODUCER_H
