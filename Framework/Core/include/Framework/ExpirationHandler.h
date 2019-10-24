// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_EXPIRATIONHANDLER_H
#define FRAMEWORK_EXPIRATIONHANDLER_H

#include "Framework/Lifetime.h"
#include <cstdint>
#include <functional>

namespace o2
{
namespace framework
{

struct PartRef;
struct ServiceRegistry;
struct TimesliceIndex;
struct TimesliceSlot;

/// Typesafe index inside
struct RouteIndex {
  int value;
};

struct ExpirationHandler {
  using Creator = std::function<TimesliceSlot(TimesliceIndex&)>;
  using Checker = std::function<bool(uint64_t timestamp)>;
  using Handler = std::function<void(ServiceRegistry&, PartRef& expiredInput, uint64_t timestamp)>;

  RouteIndex routeIndex;
  Lifetime lifetime;
  Creator creator;
  Checker checker;
  Handler handler;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_EXPIRATIONHANDLER_H
