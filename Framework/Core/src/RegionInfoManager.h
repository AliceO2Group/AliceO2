// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_REGIONINFOMANAGER_H_
#define O2_FRAMEWORK_REGIONINFOMANAGER_H_

#include "Framework/ServiceHandle.h"
#include <FairMQUnmanagedRegion.h>
#include <atomic>

namespace o2::framework
{

struct CallbackService;

/// Helper class to keep track of notified FairMQRegionInfos
class RegionInfoManager
{
 public:
  constexpr static ServiceKind service_kind = ServiceKind::Global;

  RegionInfoManager(CallbackService& callbackService);
  void post(FairMQRegionInfo);
  void notify();

 private:
  CallbackService& mCallbackService;
  std::atomic<size_t> mLastNotified;
  std::atomic<size_t> mNextInsert;

  /// Maximum number of pending notifications. If
  /// we have more, we are in trouble in any case.
  constexpr static size_t MAX_PENDING_REGION_INFOS = 256;
  /// A list of the region infos not yet notified.
  FairMQRegionInfo mPendingRegionInfos[MAX_PENDING_REGION_INFOS];
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_REGIONINFOMANAGER_H_
