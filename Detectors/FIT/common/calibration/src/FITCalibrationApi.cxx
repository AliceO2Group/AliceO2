// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FITCalibration/FITCalibrationApi.h"
#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"

using namespace o2::fit;

void FITCalibrationApi::init()
{
  //caching in basicCCDBManager enabled by default
  o2::ccdb::BasicCCDBManager::instance().setURL(DEFAULT_CCDB_URL);
}

template <typename CalibrationObjectType>
const char* FITCalibrationApi::getObjectPath()
{
  static_assert(sizeof(CalibrationObjectType) == 0, "[FITCalibrationApi] Cannot find specialization provided Calibration Object Type");
  return {};
}

template <typename CalibrationObjectType>
const CalibrationObjectType& FITCalibrationApi::getMostRecentCalibrationObject()
{
  static_assert(sizeof(CalibrationObjectType) == 0,"[FITCalibrationApi] Cannot find specialization provided Calibration Object Type" );
  return {};
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
const char* FITCalibrationApi::getObjectPath<o2::ft0::FT0ChannelTimeCalibrationObject>()
{
  return "FT0/Calibration/ChannelTimeOffset";
}

template <>
const o2::ft0::FT0ChannelTimeCalibrationObject& FITCalibrationApi::getMostRecentCalibrationObject<o2::ft0::FT0ChannelTimeCalibrationObject>()
{
  return *(o2::ccdb::BasicCCDBManager::instance().get<o2::ft0::FT0ChannelTimeCalibrationObject>(getObjectPath<o2::ft0::FT0ChannelTimeCalibrationObject>()));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
