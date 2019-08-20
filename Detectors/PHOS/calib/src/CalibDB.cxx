// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSCalib/BadChannelMap.h"
#include "PHOSCalib/CalibDB.h"

using namespace o2::phos;

CalibDB::CalibDB(const std::string_view server) : CalibDB()
{
  mCCDBServer = server;
}

void CalibDB::init()
{
  mCCDBManager.init(mCCDBServer);
  mInit = true;
}

void CalibDB::storeBadChannelMap(BadChannelMap* bcm, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit)
    init();
  mCCDBManager.store(new o2::TObjectWrapper<o2::phos::BadChannelMap>(bcm), "BadChannelMap/PHS", metadata, rangestart, rangeend);
}

BadChannelMap* CalibDB::readBadChannelMap(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit)
    init();
  auto result = mCCDBManager.retrieve("BadChannelMap/PHS", metadata, timestamp);
  if (!result)
    throw ObjectNotFoundException(mCCDBServer, "BadChannelMap/PHS", metadata, timestamp);
  if (result->IsA() != TObjectWrapper<o2::phos::BadChannelMap>::Class())
    throw TypeMismatchException("TObjectWrapper<o2::phos::BadChannelMap>", result->IsA()->GetName());
  auto wrap = dynamic_cast<TObjectWrapper<o2::phos::BadChannelMap>*>(result);
  if (!wrap)
    throw TypeMismatchException("TObjectWrapper<o2::phos::BadChannelMap>", result->IsA()->GetName()); // type checked before - should not enter here
  return wrap->getObj();
}
void CalibDB::storeCalibParams(CalibParams* prms, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit)
    init();
  mCCDBManager.store(new o2::TObjectWrapper<o2::phos::CalibParams>(prms), "CalibParams/PHS", metadata, rangestart, rangeend);
}
CalibParams* CalibDB::readCalibParams(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit)
    init();
  auto result = mCCDBManager.retrieve("CalibParams/PHS", metadata, timestamp);
  if (!result)
    throw ObjectNotFoundException(mCCDBServer, "CalibParams/PHS", metadata, timestamp);
  if (result->IsA() != TObjectWrapper<o2::phos::CalibParams>::Class())
    throw TypeMismatchException("TObjectWrapper<o2::phos::CalibParams>", result->IsA()->GetName());
  auto wrap = dynamic_cast<TObjectWrapper<o2::phos::CalibParams>*>(result);
  if (!wrap)
    throw TypeMismatchException("TObjectWrapper<o2::phos::CalibParams>", result->IsA()->GetName()); // type checked before - should not enter here
  return wrap->getObj();
}
