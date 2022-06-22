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
#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"
#endif

int makeChannelOffsetCalibObjectInCCDB(const std::string url = "http://ccdb-test.cern.ch:8080")
{
  using CalibObjWithInfoType = std::pair<o2::ccdb::CcdbObjectInfo, std::unique_ptr<std::vector<char>>>;
  std::array<int, 208> offsets;
  for (int i = 0; i < 208; i++) {
    offsets[i] = 0;
  }
  o2::ccdb::CcdbApi api;
  api.init(url);
  //  std::map<std::string, std::string> md;
  CalibObjWithInfoType result;
  o2::ft0::FT0ChannelTimeCalibrationObject calibrationObject;
  static std::map<std::string, std::string> metaData;
  auto clName = o2::utils::MemFileHelper::getClassName(calibrationObject);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  uint64_t starting = 1546300800;                                   //01.01.2019
  uint64_t stopping = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP; //1633046400; // 01.10.2021
  LOG(info) << " clName " << clName << " flName " << flName;
  result.first = o2::ccdb::CcdbObjectInfo("FT0/Calib/ChannelTimeOffset", clName, flName, metaData, starting, stopping);
  result.second = o2::ccdb::CcdbApi::createObjectImage(&offsets, &result.first);
  LOG(info) << " FITCalibrationApi::doSerializationAndPrepareObjectInfo"
            << " start " << starting << " end " << stopping;
  api.storeAsTFileAny(&calibrationObject, "FT0/Calib/ChannelTimeOffset", metaData, starting, stopping);

  return 0;
}
