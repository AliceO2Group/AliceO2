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
#ifndef O2_FRAMEWORK_DISPATCHCONTROL_H_
#define O2_FRAMEWORK_DISPATCHCONTROL_H_

#include "Framework/DispatchPolicy.h"
#include "Framework/OutputRoute.h"
#include "Framework/RoutingIndices.h"
#include <functional>
#include <string>

#include <fairmq/FwdDecls.h>

namespace o2::header
{
struct DataHeader;
}

namespace o2::framework
{
/// @struct DispatchControl
/// @brief Control for the message dispatching within message context.
/// Depending on dispatching policy, objects which get ready to be sent during
/// computing can be scheduled to be sent immediately. The trigger callback
/// is used to decide when to sent the scheduled messages via the actual dispatch
/// callback.
struct DispatchControl {
  using DispatchCallback = std::function<void(fair::mq::Parts&& message, ChannelIndex index, int)>;
  using DispatchTrigger = std::function<bool(o2::header::DataHeader const&)>;
  // dispatcher callback
  DispatchCallback dispatch;
  // matcher to trigger sending of scheduled messages
  DispatchTrigger trigger;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_DISPATCHCONTROL_H_
