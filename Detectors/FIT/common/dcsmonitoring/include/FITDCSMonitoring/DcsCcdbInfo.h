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

#ifndef O2_FIT_DCS_VALIDITY_H
#define O2_FIT_DCS_VALIDITY_H

#include "CCDB/CcdbApi.h"

namespace o2::fit
{
class DcsCcdbInfo
{
 public:
  DcsCcdbInfo(long validityInDays) : mValidityDays(validityInDays) {}

  long getValidityTimestamp(long startTimestamp) const
  {
    return startTimestamp + mValidityDays * o2::ccdb::CcdbObjectInfo::DAY;
  }

  void setValidityPeriodInDays(long days)
  {
    mValidityDays = days;
  }

  long getValidityPeriodInDays() const
  {
    return mValidityDays;
  }

  void setValidateUpload(bool validate)
  {
    mValidateUpload = validate;
  }

  bool getValidateUpload() const
  {
    return mValidateUpload;
  }

  void setCcdbPath(const std::string path)
  {
    mCcdbPath = path;
  }

  const std::string& getCcdbPath() const
  {
    return mCcdbPath;
  }

  template <typename DcsConfigType>
  o2::ccdb::CcdbObjectInfo createObjectInfo(const DcsConfigType& configObject, long startValidityTimestamp, const std::map<std::string, std::string>& metadata)
  {
    o2::ccdb::CcdbObjectInfo objectInfo;
    o2::calibration::Utils::prepareCCDBobjectInfo(configObject, objectInfo, mCcdbPath, metadata, startValidityTimestamp, getValidityTimestamp(startValidityTimestamp));
    objectInfo.setValidateUpload(mValidateUpload);
    return objectInfo;
  }

 private:
  long mValidityDays;
  bool mValidateUpload;
  std::string mCcdbPath;
};
} // namespace o2::fit
#endif