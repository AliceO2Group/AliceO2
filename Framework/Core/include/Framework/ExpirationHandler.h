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

#ifndef O2_FRAMEWORK_EXPIRATIONHANDLER_H_
#define O2_FRAMEWORK_EXPIRATIONHANDLER_H_

#include "Framework/Lifetime.h"
#include "Framework/RoutingIndices.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/InputSpan.h"
#include <cstdint>
#include <functional>

namespace o2::framework
{

struct PartRef;
struct ServiceRegistry;
struct TimesliceIndex;
struct TimesliceSlot;
struct InputRecord;

struct ExpirationHandler {
  using Creator = std::function<TimesliceSlot(ChannelIndex, TimesliceIndex&)>;
  /// Callback type to check if the record must be expired
  using Checker = std::function<bool(ServiceRegistry&, uint64_t timestamp, InputSpan const& record)>;
  /// Callback type to actually materialise a given record
  using Handler = std::function<void(ServiceRegistry&, PartRef& expiredInput, data_matcher::VariableContext& variables)>;

  std::string name = "unset";
  RouteIndex routeIndex;
  Lifetime lifetime;
  Creator creator;
  Checker checker;
  Handler handler;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_EXPIRATIONHANDLER_H_
