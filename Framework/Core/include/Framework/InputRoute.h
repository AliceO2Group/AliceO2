// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_INPUTROUTE_H
#define FRAMEWORK_INPUTROUTE_H

#include "Framework/ExpirationHandler.h"
#include "Framework/InputSpec.h"
#include <cstddef>
#include <string>
#include <functional>

namespace o2
{
namespace framework
{

struct PartRef;
struct ServiceRegistry;
class ConfigParamRegistry;

/// This uniquely identifies a route to from which data matching @a matcher
/// input spec gets to the device. In case of time pipelining @a timeslice
/// refers to the timeslice associated to this route. The The two callbacks
/// @danglingChecker and @expirationHandler are used to decide wether or not
/// the input should be created without having incoming data associated to it
/// and if yes, how.  By default inputs are never considered valid and they are
/// never created from nothing.
struct InputRoute {
  using DanglingConfigurator = std::function<ExpirationHandler::Checker(ConfigParamRegistry const&)>;
  using ExpirationConfigurator = std::function<ExpirationHandler::Handler(ConfigParamRegistry const&)>;

  InputSpec matcher;
  std::string sourceChannel;
  size_t timeslice;
  DanglingConfigurator danglingConfigurator = nullptr;
  ExpirationConfigurator expirationConfigurator = nullptr;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_INPUTROUTE_H
