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

#ifndef O2_CCDB_CCDBOBJECTINFO_H_
#define O2_CCDB_CCDBOBJECTINFO_H_

#include <Rtypes.h>

#include <utility>
#include "Framework/Logger.h"

/// @brief information complementary to a CCDB object (path, metadata, startTimeValidity, endTimeValidity etc)

namespace o2::ccdb
{
class CcdbObjectInfo
{
 public:
  CcdbObjectInfo() = default;
  CcdbObjectInfo(std::string path, std::string objType, std::string flName,
                 std::map<std::string, std::string> metadata,
                 long startValidityTimestamp, long endValidityTimestamp)
    : mObjType(std::move(objType)), mFileName(std::move(flName)), mPath(std::move(path)), mMD(std::move(metadata)), mStart(startValidityTimestamp), mEnd(endValidityTimestamp) {}
  ~CcdbObjectInfo() = default;

  [[nodiscard]] const std::string& getObjectType() const { return mObjType; }
  void setObjectType(const std::string& tp) { mObjType = tp; }

  [[nodiscard]] const std::string& getFileName() const { return mFileName; }
  void setFileName(const std::string& nm) { mFileName = nm; }

  [[nodiscard]] const std::string& getPath() const { return mPath; }
  void setPath(const std::string& path) { mPath = path; }

  [[nodiscard]] const std::map<std::string, std::string>& getMetaData() const { return mMD; }
  void setMetaData(const std::map<std::string, std::string>& md) { mMD = md; }

  [[nodiscard]] long getStartValidityTimestamp() const { return mStart; }
  void setStartValidityTimestamp(long start) { mStart = start; }

  [[nodiscard]] long getEndValidityTimestamp() const { return mEnd; }
  void setEndValidityTimestamp(long end) { mEnd = end; }

 private:
  std::string mObjType{};                 // object type (e.g. class)
  std::string mFileName{};                // file name in the CCDB
  std::string mPath{};                    // path in the CCDB
  std::map<std::string, std::string> mMD; // metadata
  long mStart = 0;                        // start of the validity of the object
  long mEnd = 0;                          // end of the validity of the object

  ClassDefNV(CcdbObjectInfo, 1);
};

} // namespace o2::ccdb

#endif // O2_CCDB_CCDBOBJECTINFO_H_
