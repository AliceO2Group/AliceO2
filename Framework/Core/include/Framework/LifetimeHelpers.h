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
#ifndef O2_FRAMEWORK_LIFETIMEHELPERS_H_
#define O2_FRAMEWORK_LIFETIMEHELPERS_H_

#include "Framework/ExpirationHandler.h"
#include "Framework/PartRef.h"
#include "InputRoute.h"

#include <chrono>
#include <functional>
#include <string>

namespace o2::framework
{

struct ConcreteDataMatcher;
struct DeviceState;

/// Lifetime handlers are used to manage the cases in which data is not coming
/// from the dataflow, but from some other source or trigger, e.g.,
/// in the case condition data or time based triggers.
struct LifetimeHelpers {
  /// Callback which does nothing, waiting for data to arrive.
  static ExpirationHandler::Creator dataDrivenCreation();
  /// Callback which creates a new timeslice as soon as one is available and
  /// uses an incremental number as timestamp.
  /// @a repetitions number of times we should repeat the same value of the
  /// enumeration.
  static ExpirationHandler::Creator enumDrivenCreation(size_t first, size_t last, size_t step, size_t inputTimeslice, size_t maxTimeSliceId, size_t repetitions);

  /// Callback which creates a new timeslice when timer
  /// expires and there is not a compatible datadriven callback
  /// available.
  static ExpirationHandler::Creator timeDrivenCreation(std::vector<std::chrono::microseconds> periods, std::vector<std::chrono::seconds> intervals, std::function<bool(void)> hasTimerFired, std::function<void(uint64_t, uint64_t)> updateTimerPeriod);

  /// Callback which  creates a new timeslice whenever some libuv event happens
  static ExpirationHandler::Creator uvDrivenCreation(int loopReason, DeviceState& state);

  /// Callback which never expires records. To be used with, e.g.
  /// Lifetime::Timeframe.
  static ExpirationHandler::Checker expireNever();
  /// Callback which always expires records. To be used with, e.g.
  /// Lifetime::Transient.
  static ExpirationHandler::Checker expireAlways();
  /// Callback which expires records based on the content of the record.
  /// To be used with, e.g. Lifetime::Optional.
  static ExpirationHandler::Checker expireIfPresent(std::vector<InputRoute> const& schema, ConcreteDataMatcher matcher);

  /// Callback which expires records with the rate given by @a period, in
  /// microseconds.
  static ExpirationHandler::Checker expireTimed(std::chrono::microseconds period);

  /// Does nothing. Use this for cases where you do not want to do anything
  /// when records expire. This is the default behavior for data (which never
  /// expires via this mechanism).
  static ExpirationHandler::Handler doNothing();

  /// Fetches CTP if not requested via @a waitForCTP and not available
  /// Sets DataTakingContext::orbitResetTime accordingly
  static ExpirationHandler::Checker expectCTP(std::string const& serverUrl, bool waitForCTP);
  /// Build a fetcher for an object from CCDB when the record is expired.
  /// @a spec is the associated InputSpec
  /// @a prefix is the lookup prefix in CCDB
  /// @a overrideTimestamp can be used to override the timestamp found in the data.
  static ExpirationHandler::Handler fetchFromCCDBCache(InputSpec const& spec,
                                                       std::string const& prefix,
                                                       std::string const& overrideTimestamp,
                                                       std::string const& sourceChannel);

  /// Build a fetcher for an object from an out of band FairMQ channel whenever the record is expired.
  /// @a spec is the associated InputSpec
  /// @a channelName the channel we should Receive data from
  static ExpirationHandler::Handler fetchFromFairMQ(InputSpec const& spec,
                                                    std::string const& channelName);

  /// Create an entry in the registry for histograms on the first
  /// FIXME: actually implement this
  /// FIXME: provide a way to customise the histogram from the configuration.
  static ExpirationHandler::Handler fetchFromQARegistry();

  /// Create an entry in the registry for histograms on the first
  /// FIXME: actually implement this
  /// FIXME: provide a way to customise the histogram from the configuration.
  static ExpirationHandler::Handler fetchFromObjectRegistry();

  /// Enumerate entries on every invokation.  @a matcher is the InputSpec which the
  /// given enumeration refers to. In particular messages created by the
  /// returned ExpirationHandler will have an header which matches the
  /// dataOrigin, dataDescrition and dataSpecification of the given @a route.
  /// The payload of each message will contain an incremental number for each
  /// message being created.
  static ExpirationHandler::Handler enumerate(ConcreteDataMatcher const& spec, std::string const& sourceChannel,
                                              int64_t orbitOffset, int64_t orbitMultiplier);

  /// Create a dummy (empty) message every time a record expires, suing @a spec
  /// as content of the payload.
  static ExpirationHandler::Handler dummy(ConcreteDataMatcher const& spec, std::string const& sourceChannel);
};

std::ostream& operator<<(std::ostream& oss, Lifetime const& val);

} // namespace o2::framework

#endif // O2_FRAMEWORK_LIFETIMEHELPERS_H_
