// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALCalib/BadChannelMap.h"
#include "EMCALCalib/CalibDB.h"

using namespace o2::emcal;

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
  mCCDBManager.store(new o2::TObjectWrapper<o2::emcal::BadChannelMap>(bcm), "BadChannelMap/EMC", metadata, rangestart, rangeend);
}

BadChannelMap* CalibDB::readBadChannelMap(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit)
    init();
  auto result = mCCDBManager.retrieve("BadChannelMap/EMC", metadata, timestamp);
  if (!result)
    throw ObjectNotFoundException(mCCDBServer, "BadChannelMap/EMC", metadata, timestamp);
  if (result->IsA() != TObjectWrapper<o2::emcal::BadChannelMap>::Class())
    throw TypeMismatchException("TObjectWrapper<o2::emcal::BadChannelMap>", result->IsA()->GetName());
  auto wrap = dynamic_cast<TObjectWrapper<o2::emcal::BadChannelMap>*>(result);
  if (!wrap)
    throw TypeMismatchException("TObjectWrapper<o2::emcal::BadChannelMap>", result->IsA()->GetName()); // type checked before - should not enter here
  return wrap->getObj();
}