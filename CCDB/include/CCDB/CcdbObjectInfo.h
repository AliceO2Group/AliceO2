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
  // time intervals in milliseconds
  static constexpr long SECOND = 1000;
  static constexpr long MINUTE = 60 * SECOND;
  static constexpr long HOUR = 60 * MINUTE;
  static constexpr long DAY = 24 * HOUR;
  static constexpr long MONTH = 30 * DAY;
  static constexpr long YEAR = 364 * DAY;
  static constexpr long INFINITE_TIMESTAMP = 9999999999999;
  static constexpr long INFINITE_TIMESTAMP_SECONDS = 2000000000; // not really inifinity, but close to std::numeric_limits<int>::max() till 18.05.2033
  static constexpr const char* AdjustableEOV = "adjustableEOV";
  static constexpr const char* DefaultObj = "default";

  CcdbObjectInfo(bool adjustableEOV = true)
  {
    if (adjustableEOV) {
      setAdjustableEOV();
    }
  }
  CcdbObjectInfo(std::string path, std::string objType, std::string flName,
                 std::map<std::string, std::string> metadata,
                 long startValidityTimestamp, long endValidityTimestamp, bool adjustableEOV = true, bool validateUpload = false)
    : mObjType(std::move(objType)), mFileName(std::move(flName)), mPath(std::move(path)), mMD(std::move(metadata)), mStart(startValidityTimestamp), mEnd(endValidityTimestamp), mValidateUpload(validateUpload)
  {
    if (adjustableEOV) {
      setAdjustableEOV();
    }
  }
  ~CcdbObjectInfo() = default;

  [[nodiscard]] const std::string& getObjectType() const { return mObjType; }
  void setObjectType(const std::string& tp) { mObjType = tp; }

  [[nodiscard]] const std::string& getFileName() const { return mFileName; }
  void setFileName(const std::string& nm) { mFileName = nm; }

  [[nodiscard]] const std::string& getPath() const { return mPath; }
  void setPath(const std::string& path) { mPath = path; }

  [[nodiscard]] const std::map<std::string, std::string>& getMetaData() const { return mMD; }
  void setMetaData(const std::map<std::string, std::string>& md)
  {
    mMD = md;
    if (mAdjustableEOV) {
      setAdjustableEOV();
    }
  }

  void setAdjustableEOV()
  {
    mAdjustableEOV = true;
    if (mMD.find(DefaultObj) != mMD.end()) {
      LOGP(fatal, "default object cannot have adjustable EOV, {}", mPath);
    }
    if (mMD.find(AdjustableEOV) == mMD.end()) {
      mMD[AdjustableEOV] = "true";
    }
  }

  void setValidateUpload(bool v) { mValidateUpload = v; }

  bool isAdjustableEOV() const { return mAdjustableEOV; }
  bool getValidateUpload() const { return mValidateUpload; }

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
  bool mAdjustableEOV = false;            // each new object may override EOV of object it overrides to its own SOV
  bool mValidateUpload = false;           // request to validate the upload by querying its header
  ClassDefNV(CcdbObjectInfo, 3);
};

} // namespace o2::ccdb

#endif // O2_CCDB_CCDBOBJECTINFO_H_
