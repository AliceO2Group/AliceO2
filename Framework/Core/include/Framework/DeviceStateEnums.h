// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_DEVICESTATEENUMS_H_
#define O2_FRAMEWORK_DEVICESTATEENUMS_H_

namespace o2::framework
{
/// enumeration representing the current state of a given
/// device.
enum struct StreamingState {
  /// Data is being processed
  Streaming = 0,
  /// End of streaming requested, but not notified
  EndOfStreaming = 1,
  /// End of streaming notified
  Idle = 2,
};

enum struct TransitionHandlingState {
  /// No pending transitions
  NoTransition,
  /// A transition was notified to be requested
  Requested,
  /// A transition needs to be fullfilled ASAP
  Expired
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_DEVICESTATEENUMS_H_
