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

#ifndef O2_FRAMEWORK_DECONGESTIONSERVICE_H_
#define O2_FRAMEWORK_DECONGESTIONSERVICE_H_
#include "Framework/AsyncQueue.h"

namespace o2::framework
{
struct DecongestionService {
  /// Wether we are a source in the processing chain
  bool isFirstInTopology = true;
  /// Last timeslice we communicated. Notice this should never go backwards.
  int64_t lastTimeslice = 0;
  /// The next timeslice we should consume, when running in order,
  /// using an ordered completion policy.
  int64_t nextTimeslice = 0;
  /// Ordered completion policy is active.
  bool orderedCompletionPolicyActive = false;
  // Task to enqueue the oldest possible timeslice propagation
  // at the end of any processing chain.
  o2::framework::AsyncTaskId oldestPossibleTimesliceTask = {0};
};
} // namespace o2::framework
#endif
