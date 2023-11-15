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

/// Context associated to a device. In principle
/// multiple DataProcessors can run on a Device (even if we
/// do not do it for now).
#ifndef O2_FRAMEWORK_DEVICECONTEXT_H_
#define O2_FRAMEWORK_DEVICECONTEXT_H_

typedef struct uv_timer_s uv_timer_t;
typedef struct uv_signal_s uv_signal_t;

namespace o2::framework
{
struct ComputingQuotaStats;

/// Stucture which holds the whole runtime context
/// of a running device which is not stored as
/// a standalone object in the service registry.
struct DeviceContext {
  ComputingQuotaStats* quotaStats = nullptr;
  uv_timer_t* gracePeriodTimer = nullptr;
  uv_signal_t* sigusr1Handle = nullptr;
  int expectedRegionCallbacks = 0;
  int exitTransitionTimeout = 0;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_DEVICECONTEXT_H_
