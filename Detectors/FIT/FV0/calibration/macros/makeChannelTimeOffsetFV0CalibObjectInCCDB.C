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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <string>
#include "TFile.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/BasicCCDBManager.h"
#include <iostream>
#include <array>
#include "DataFormatsFV0/FV0ChannelTimeCalibrationObject.h"
#include "CCDB/CcdbApi.h"
#include "FV0Base/Constants.h"

int makeChannelTimeOffsetFV0CalibObjectInCCDB(const std::string url = "https://alice-ccdb.cern.ch:8080")
{
  using CalibObjWithInfoType = std::pair<o2::ccdb::CcdbObjectInfo, std::unique_ptr<std::vector<char>>>;
  std::array<int, o2::fv0::Constants::nFv0Channels> offsets;
  for (int i = 0; i < o2::fv0::Constants::nFv0Channels; i++) {
    offsets[i] = 0;
  }
  o2::ccdb::CcdbApi api;
  api.init(url);
  CalibObjWithInfoType result;
  o2::fv0::FV0ChannelTimeCalibrationObject calibrationObject;
  static std::map<std::string, std::string> metaData;
  auto clName = o2::utils::MemFileHelper::getClassName(calibrationObject);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  uint64_t starting = 1546300800; // 01.01.2019
  uint64_t stopping = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP;
  LOG(info) << " clName " << clName << " flName " << flName;
  result.first = o2::ccdb::CcdbObjectInfo("FV0/Calib/ChannelTimeOffset", clName, flName, metaData, starting, stopping);
  result.second = o2::ccdb::CcdbApi::createObjectImage(&offsets, &result.first);
  LOG(info) << " start " << starting << " end " << stopping;
  api.storeAsTFileAny(&calibrationObject, "FV0/Calib/ChannelTimeOffset", metaData, starting, stopping);
  return 0;
}
#endif
