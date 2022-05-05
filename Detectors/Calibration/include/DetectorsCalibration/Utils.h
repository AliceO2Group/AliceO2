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

/// @file   Utils.h
/// @brief  Utils and constants for calibration and related workflows

#ifndef O2_CALIBRATION_CONVENTIONS_H
#define O2_CALIBRATION_CONVENTIONS_H

#include <typeinfo>
#include <utility>
#include <fstream>
#include <TMemFile.h>
#include "Headers/DataHeader.h"
#include "CommonUtils/StringUtils.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"

namespace o2
{
namespace calibration
{

struct Utils {
  static constexpr o2::header::DataOrigin gDataOriginCDBPayload{"CLP"}; // generic DataOrigin for calibrations payload
  static constexpr o2::header::DataOrigin gDataOriginCDBWrapper{"CLW"}; // generic DataOrigin for calibrations wrapper
  template <typename T>
  static void prepareCCDBobjectInfo(T& obj, o2::ccdb::CcdbObjectInfo& info, const std::string& path,
                                    const std::map<std::string, std::string>& md, long start, long end = -1);
};

template <typename T>
void Utils::prepareCCDBobjectInfo(T& obj, o2::ccdb::CcdbObjectInfo& info, const std::string& path,
                                  const std::map<std::string, std::string>& md, long start, long end)
{

  // prepare all info to be sent to CCDB for object obj
  auto clName = o2::utils::MemFileHelper::getClassName(obj);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  info.setPath(path);
  info.setObjectType(clName);
  info.setFileName(flName);
  info.setStartValidityTimestamp(start);
  info.setEndValidityTimestamp(end);
  info.setMetaData(md);
}

} // namespace calibration
} // namespace o2

#endif
