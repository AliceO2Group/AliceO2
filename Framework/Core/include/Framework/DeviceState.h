// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DEVICESTATE_H_
#define O2_FRAMEWORK_DEVICESTATE_H_

#include "Framework/ChannelInfo.h"
#include <vector>
#include <string>
#include <map>
#include <utility>

namespace o2::framework
{

/// Running state information of a given device
struct DeviceState {
  enum struct StreamingState {
    /// Data is being processed
    Streaming,
    /// End of streaming requested, but not notified
    EndOfStreaming,
    /// End of streaming notified
    Idle,
  };
  std::vector<InputChannelInfo> inputChannelInfos;
  StreamingState streaming = StreamingState::Streaming;
  bool quitRequested = false;
};

} // namespace o2::framework
#endif
