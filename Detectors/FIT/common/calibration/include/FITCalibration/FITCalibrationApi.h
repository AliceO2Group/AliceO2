// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FITCALIBRATIONAPI_H
#define O2_FITCALIBRATIONAPI_H

#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "FT0Calibration/FT0DummyCalibrationObject.h" //delete this when example not needed anymore
#include "CCDB/CcdbObjectInfo.h"
#include <vector>

namespace o2::fit
{
class FITCalibrationApi
{
 private:
  static constexpr const char* DEFAULT_CCDB_URL = "http://localhost:8080";
  using CalibObjWithInfoType = std::pair<o2::ccdb::CcdbObjectInfo, std::unique_ptr<std::vector<char>>>;
  inline static unsigned long mProcessingTimestamp = 0;

 public:
  FITCalibrationApi() = delete;
  FITCalibrationApi(const FITCalibrationApi&) = delete;
  FITCalibrationApi(FITCalibrationApi&&) = delete;

  static void init();
  static void setProcessingTimestamp(unsigned long tf) { mProcessingTimestamp = tf; }
  [[nodiscard]] static unsigned long getProcessingTimestamp() { return mProcessingTimestamp; }

  template <typename CalibrationObjectType>
  [[nodiscard]] static const char* getObjectPath();

  template <typename CalibrationObjectType>
  [[nodiscard]] static const CalibrationObjectType& getMostRecentCalibrationObject();

  template <typename CalibrationObjectType>
  [[nodiscard]] static const CalibrationObjectType& getCalibrationObjectForGivenTimestamp(long timestamp);

  template <typename CalibrationObjectType>
  [[nodiscard]] static std::vector<CalibObjWithInfoType> prepareCalibrationObjectToSend(const CalibrationObjectType& calibrationObject);

 private:
  template <typename CalibrationObjectType>
  static void handleInvalidCalibrationObjectType();

  template <typename CalibrationObjectType>
  [[nodiscard]] static CalibObjWithInfoType doSerializationAndPrepareObjectInfo(const CalibrationObjectType& calibrationObject);
};

void FITCalibrationApi::init()
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
FITCalibrationApi::CalibObjWithInfoType FITCalibrationApi::doSerializationAndPrepareObjectInfo(const CalibrationObjectType& calibrationObject)
{
  static std::map<std::string, std::string> metaData;
  static std::string dummyStringVariableThatWillBeChangedAnyway;

  CalibObjWithInfoType result;
  result.first = o2::ccdb::CcdbObjectInfo(FITCalibrationApi::getObjectPath<CalibrationObjectType>(), dummyStringVariableThatWillBeChangedAnyway, dummyStringVariableThatWillBeChangedAnyway, metaData, o2::ccdb::getCurrentTimestamp(), -1);
  result.second = o2::ccdb::CcdbApi::createObjectImage(&calibrationObject, &result.first);

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
std::vector<FITCalibrationApi::CalibObjWithInfoType> FITCalibrationApi::prepareCalibrationObjectToSend(const CalibrationObjectType& calibrationObject)
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
const char* FITCalibrationApi::getObjectPath<o2::ft0::FT0ChannelTimeCalibrationObject>()
{
  return "FT0/Calibration/ChannelTimeOffset";
}

template <>
std::vector<FITCalibrationApi::CalibObjWithInfoType> FITCalibrationApi::prepareCalibrationObjectToSend<o2::ft0::FT0ChannelTimeCalibrationObject>(const o2::ft0::FT0ChannelTimeCalibrationObject& calibrationObject)
{
  std::vector<CalibObjWithInfoType> result;
  result.emplace_back(doSerializationAndPrepareObjectInfo(calibrationObject));
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

// DUMMY STUFF DELETE IT WHEN EXAMPLE NOT NEEDED ANYMORE
template <>
const char* FITCalibrationApi::getObjectPath<o2::ft0::FT0DummyCalibrationObjectTime>()
{
  return "FT0/Calibration/DummyTime";
}

template <>
const char* FITCalibrationApi::getObjectPath<o2::ft0::FT0DummyCalibrationObjectCharge>()
{
  return "FT0/Calibration/DummyCharge";
}

template <>
const char* FITCalibrationApi::getObjectPath<o2::ft0::FT0DummyNeededCalibrationObject>()
{
  return "FT0/Calibration/DummyNeeded";
}

template <>
std::vector<FITCalibrationApi::CalibObjWithInfoType> FITCalibrationApi::prepareCalibrationObjectToSend<o2::ft0::FT0DummyCalibrationObject>(const o2::ft0::FT0DummyCalibrationObject& calibrationObject)
{
  std::vector<CalibObjWithInfoType> result;
  result.emplace_back(doSerializationAndPrepareObjectInfo(calibrationObject.mChargeCalibrationObject));
  result.emplace_back(doSerializationAndPrepareObjectInfo(calibrationObject.mTimeCalibrationObject));
  return result;
}

// END OF DUMMY STUFF DELETE IT WHEN EXAMPLE NOT NEEDED ANYMORE

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace o2::fit

#endif //O2_FITCALIBRATIONAPI_H
