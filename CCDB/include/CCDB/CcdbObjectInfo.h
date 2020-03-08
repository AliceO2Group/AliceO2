// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTOR_CALIB_WRAPPER_H_
#define DETECTOR_CALIB_WRAPPER_H_

#include <Rtypes.h>
#include "Framework/Logger.h"

/// @brief information complementary to a CCDB object (path, metadata, startTimeValidity, endTimeValidity etc)

namespace o2
{
namespace ccdb
{
class CcdbObjectInfo
{
 public:
  CcdbObjectInfo() = default;
  CcdbObjectInfo(std::string const& path, std::string const& objType, std::string const& flName,
                 std::map<std::string, std::string> const& metadata,
                 long startValidityTimestamp, long endValidityTimestamp)
    : mPath(path), mObjType(objType), mFileName(flName), mMD(metadata), mStart(startValidityTimestamp), mEnd(endValidityTimestamp) {}
  ~CcdbObjectInfo() = default;

  const std::string& getObjectType() const { return mObjType; }
  void setObjectType(const std::string& tp) { mObjType = tp; }

  const std::string& getFileName() const { return mFileName; }
  void setFileName(const std::string& nm) { mFileName = nm; }

  const std::string& getPath() const { return mPath; }
  void setPath(const std::string& path) { mPath = path; }

  const std::map<std::string, std::string>& getMetaData() const { return mMD; }
  void setMetaData(const std::map<std::string, std::string>& md) { mMD = md; }

  long getStartValidityTimestamp() const { return mStart; }
  void setStartValidityTimestamp(long start) { mStart = start; }

  long getEndValidityTimestamp() const { return mEnd; }
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

} // namespace ccdb
} // namespace o2

#endif
