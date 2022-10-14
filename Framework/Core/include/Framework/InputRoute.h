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
#ifndef FRAMEWORK_INPUTROUTE_H
#define FRAMEWORK_INPUTROUTE_H

#include "Framework/ExpirationHandler.h"
#include "Framework/ServiceRegistryRef.h"
#include "Framework/InputSpec.h"
#include <cstddef>
#include <string>
#include <functional>
#include <optional>

namespace o2::framework
{

struct PartRef;
struct ServiceRegistry;
struct DeviceState;
class ConfigParamRegistry;

struct RouteConfigurator {
  using CreationConfigurator = std::function<ExpirationHandler::Creator(DeviceState&, ServiceRegistryRef, ConfigParamRegistry const&)>;
  using DanglingConfigurator = std::function<ExpirationHandler::Checker(DeviceState&, ConfigParamRegistry const&)>;
  using ExpirationConfigurator = std::function<ExpirationHandler::Handler(DeviceState&, ConfigParamRegistry const&)>;
  std::string name = "unknown";

  CreationConfigurator creatorConfigurator = nullptr;
  DanglingConfigurator danglingConfigurator = nullptr;
  ExpirationConfigurator expirationConfigurator = nullptr;
};

/// This uniquely identifies a route to from which data matching @a matcher
/// input spec gets to the device. In case of time pipelining @a timeslice
/// refers to the timeslice associated to this route. The three callbacks @a
/// creatorConfigurator, @a danglingChecker and @a expirationHandler are used
/// to respectively create new empty timeslices, decide wether or not the input
/// should be created without having incoming data associated to it and if yes,
/// how.  By default inputs are never considered valid and they are never
/// created from nothing.
struct InputRoute {

  // FIXME: This should really go away and we should make sure that
  //        whenever we pass the input routes we also have the associated
  //        input specs available.
  InputSpec matcher;
  size_t inputSpecIndex;
  std::string sourceChannel;
  size_t timeslice;
  std::optional<RouteConfigurator> configurator;
};

} // namespace o2::framework

#endif // FRAMEWORK_INPUTROUTE_H
