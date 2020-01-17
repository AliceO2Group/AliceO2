// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DISPATCHCONTROL_H
#define FRAMEWORK_DISPATCHCONTROL_H

#include "Framework/DispatchPolicy.h"
#include <functional>
#include <string>

class FairMQParts;

namespace o2
{
namespace header
{
struct DataHeader;
}

namespace framework
{
/// @struct DispatchControl
/// @brief Control for the message dispatching within message context.
/// Depending on dispatching policy, objects which get ready to be sent during
/// computing can be scheduled to be sent immediately. The trigger callback
/// is used to decide when to sent the scheduled messages via the actual dispatch
/// callback.
struct DispatchControl {
  using DispatchCallback = std::function<void(FairMQParts&& message, std::string const&, int)>;
  using DispatchTrigger = std::function<bool(o2::header::DataHeader const&)>;
  // dispatcher callback
  DispatchCallback dispatch;
  // matcher to trigger sending of scheduled messages
  DispatchTrigger trigger;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_DISPATCHCONTROL_H
