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
#ifndef O2_FRAMEWORK_DATAPROCESSINGHELPERS_H_
#define O2_FRAMEWORK_DATAPROCESSINGHELPERS_H_

#include "Framework/TimesliceIndex.h"
#include <fairmq/FwdDecls.h>

namespace o2::framework
{

struct OutputChannelSpec;
class FairMQDeviceProxy;

/// Generic helpers for DataProcessing releated functions.
struct DataProcessingHelpers {
  /// Send EndOfStream message to a given channel
  /// @param device the fair::mq::Device which needs to send the EndOfStream message
  /// @param channel the OutputChannelSpec of the channel which needs to be signaled
  ///        for EndOfStream
  static void sendEndOfStream(fair::mq::Device& device, OutputChannelSpec const& channel);
  static void sendOldestPossibleTimeframe(fair::mq::Channel& channel, size_t timeslice);
  static void broadcastOldestPossibleTimeslice(FairMQDeviceProxy& proxy, size_t timeslice);
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_DATAPROCESSINGHELPERS_H_
