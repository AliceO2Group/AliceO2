// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "RegionInfoManager.h"
#include "Framework/CallbackService.h"

namespace o2::framework
{

RegionInfoManager::RegionInfoManager(CallbackService& callbackService)
  : mCallbackService{callbackService},
    mLastNotified{0},
    mNextInsert{0}
{
}

void RegionInfoManager::post(FairMQRegionInfo info)
{
  // Notice how this is a circular buffer.
  mPendingRegionInfos[(mNextInsert % MAX_PENDING_REGION_INFOS)] = info;
  mNextInsert++;
}

void RegionInfoManager::notify()
{
  while (mLastNotified < mNextInsert) {
    mCallbackService(CallbackService::Id::RegionInfoCallback, mPendingRegionInfos[(mLastNotified % MAX_PENDING_REGION_INFOS)]);
    mLastNotified++;
  }
}

} // namespace o2::framework
