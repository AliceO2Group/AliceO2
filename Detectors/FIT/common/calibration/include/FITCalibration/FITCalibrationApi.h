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

#ifndef O2_FITCALIBRATIONAPI_H
#define O2_FITCALIBRATIONAPI_H

#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"
#include "FV0Calibration/FV0ChannelTimeCalibrationObject.h"
#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"
#include "FT0Calibration/FT0CalibTimeSlewing.h"
#include "DataFormatsFT0/GlobalOffsetsCalibrationObject.h"
#include "DataFormatsFT0/GlobalOffsetsContainer.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CCDB/BasicCCDBManager.h"
#include <vector>

namespace o2::fit
{
class FITCalibrationApi
{
 private:
  static constexpr const char* DEFAULT_CCDB_URL = "http://localhost:8080";
  using CalibObjWithInfoType = std::pair<o2::ccdb::CcdbObjectInfo, std::unique_ptr<std::vector<char>>>;
  using TFType = std::uint64_t;

 public:
  FITCalibrationApi() = delete;
  FITCalibrationApi(const FITCalibrationApi&) = delete;
  FITCalibrationApi(FITCalibrationApi&&) = delete;

  static void init();

  template <typename CalibrationObjectType>
  [[nodiscard]] static const char* getObjectPath();

  template <typename CalibrationObjectType>
  [[nodiscard]] static const CalibrationObjectType& getMostRecentCalibrationObject();

  template <typename CalibrationObjectType>
  [[nodiscard]] static const CalibrationObjectType& getCalibrationObjectForGivenTimestamp(long timestamp);

  template <typename CalibrationObjectType>
  [[nodiscard]] static std::vector<CalibObjWithInfoType> prepareCalibrationObjectToSend(const CalibrationObjectType& calibrationObject, TFType tfStart, TFType tfEnd);

 private:
  template <typename CalibrationObjectType>
  static void handleInvalidCalibrationObjectType();

  template <typename CalibrationObjectType>
  [[nodiscard]] static CalibObjWithInfoType doSerializationAndPrepareObjectInfo(const CalibrationObjectType& calibrationObject, TFType tfStart, TFType tfEnd);
};

inline void FITCalibrationApi::init()
{
  //caching in basicCCDBManager enabled by default
  o2::ccdb::BasicCCDBManager::instance().setURL(DEFAULT_CCDB_URL);
}

template <typename CalibrationObjectType>
void FITCalibrationApi::handleInvalidCalibrationObjectType()
{
  static_assert(sizeof(CalibrationObjectType) == 0, "[FITCalibrationApi] Cannot find specialization provided Calibration Object Type");
}

template <typename CalibrationObjectType>
FITCalibrationApi::CalibObjWithInfoType FITCalibrationApi::doSerializationAndPrepareObjectInfo(const CalibrationObjectType& calibrationObject, TFType starting, TFType stopping)
{
  static std::map<std::string, std::string> metaData;
  static std::string dummyStringVariableThatWillBeChangedAnyway;
  CalibObjWithInfoType result;
  auto clName = o2::utils::MemFileHelper::getClassName(calibrationObject);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  stopping = stopping + 86400000; // +1 day
  LOG(info) << " clName " << clName << " flName " << flName;
  result.first = o2::ccdb::CcdbObjectInfo(FITCalibrationApi::getObjectPath<CalibrationObjectType>(), clName, flName, metaData, starting, stopping);
  result.second = o2::ccdb::CcdbApi::createObjectImage(&calibrationObject, &result.first);
  LOG(info) << " FITCalibrationApi::doSerializationAndPrepareObjectInfo"
            << " start " << starting << " end " << stopping;
  return result;
}

template <typename CalibrationObjectType>
const CalibrationObjectType& FITCalibrationApi::getCalibrationObjectForGivenTimestamp(long timestamp)
{

  o2::ccdb::BasicCCDBManager::instance().setTimestamp(timestamp);
  auto calibObjectPtr = o2::ccdb::BasicCCDBManager::instance().get<CalibrationObjectType>(getObjectPath<CalibrationObjectType>());
  if (nullptr == calibObjectPtr) {
    throw std::runtime_error("Cannot read requested calibration object");
  }
  return *calibObjectPtr;
}

template <typename CalibrationObjectType>
const CalibrationObjectType& FITCalibrationApi::getMostRecentCalibrationObject()
{
  return getCalibrationObjectForGivenTimestamp<CalibrationObjectType>(o2::ccdb::getCurrentTimestamp());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CalibrationObjectType>
std::vector<FITCalibrationApi::CalibObjWithInfoType> FITCalibrationApi::prepareCalibrationObjectToSend(const CalibrationObjectType& calibrationObject, TFType, TFType)
{
  handleInvalidCalibrationObjectType<CalibrationObjectType>();
  return {};
}

template <typename CalibrationObjectType>
const char* FITCalibrationApi::getObjectPath()
{
  handleInvalidCalibrationObjectType<CalibrationObjectType>();
  return {};
}

//----FT0----//////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline const char* FITCalibrationApi::getObjectPath<o2::ft0::FT0ChannelTimeCalibrationObject>()
{
  return "FT0/Calib/ChannelTimeOffset";
}

template <>
inline const char* FITCalibrationApi::getObjectPath<o2::ft0::FT0CalibTimeSlewing>()
{
  return "FT0/Calib/SlewingCorrection";
}
template <>
inline const char* FITCalibrationApi::getObjectPath<o2::ft0::GlobalOffsetsCalibrationObject>()
{
  return "FT0/Calib/GlobalOffsets";
}

template <>
inline std::vector<FITCalibrationApi::CalibObjWithInfoType> FITCalibrationApi::prepareCalibrationObjectToSend<o2::ft0::FT0ChannelTimeCalibrationObject>(const o2::ft0::FT0ChannelTimeCalibrationObject& calibrationObject, TFType tfStart, TFType tfEnd)
{
  std::vector<CalibObjWithInfoType> result;
  result.emplace_back(doSerializationAndPrepareObjectInfo(calibrationObject, tfStart, tfEnd));
  return result;
}

template <>
inline std::vector<FITCalibrationApi::CalibObjWithInfoType> FITCalibrationApi::prepareCalibrationObjectToSend<o2::ft0::FT0CalibTimeSlewing>(const o2::ft0::FT0CalibTimeSlewing& calibrationObject, TFType tfStart, TFType tfEnd)
{
  std::vector<CalibObjWithInfoType> result;
  result.emplace_back(doSerializationAndPrepareObjectInfo(calibrationObject, tfStart, tfEnd));
  return result;
}

template <>
inline std::vector<FITCalibrationApi::CalibObjWithInfoType> FITCalibrationApi::prepareCalibrationObjectToSend<o2::ft0::GlobalOffsetsCalibrationObject>(const o2::ft0::GlobalOffsetsCalibrationObject& calibrationObject, TFType tfStart, TFType tfEnd)
{
  std::vector<CalibObjWithInfoType> result;
  result.emplace_back(doSerializationAndPrepareObjectInfo(calibrationObject, tfStart, tfEnd));
  return result;
}

//----FV0----//////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline const char* FITCalibrationApi::getObjectPath<o2::fv0::FV0ChannelTimeCalibrationObject>()
{
  return "FV0/Calib/ChannelTimeOffset";
}

template <>
inline std::vector<FITCalibrationApi::CalibObjWithInfoType> FITCalibrationApi::prepareCalibrationObjectToSend<o2::fv0::FV0ChannelTimeCalibrationObject>(const o2::fv0::FV0ChannelTimeCalibrationObject& calibrationObject, TFType tfStart, TFType tfEnd)
{
  std::vector<CalibObjWithInfoType> result;
  result.emplace_back(doSerializationAndPrepareObjectInfo(calibrationObject, tfStart, tfEnd));
  return result;
}


} // namespace o2::fit

#endif //O2_FITCALIBRATIONAPI_H
