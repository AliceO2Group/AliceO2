// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_LIFETIMEHELPERS_H
#define FRAMEWORK_LIFETIMEHELPERS_H

#include "Framework/ExpirationHandler.h"
#include "Framework/PartRef.h"

#include <chrono>
#include <functional>
#include <string>

namespace o2
{
namespace framework
{

struct ConcreteDataMatcher;

/// Lifetime handlers are used to manage the cases in which data is not coming
/// from the dataflow, but from some other source or trigger, e.g.,
/// in the case condition data or time based triggers.
struct LifetimeHelpers {
  /// Callback which does nothing, waiting for data to arrive.
  static ExpirationHandler::Creator dataDrivenCreation();
  /// Callback which creates a new timeslice as soon as one is available and
  /// uses an incremental number as timestamp.
  static ExpirationHandler::Creator enumDrivenCreation(size_t first, size_t last, size_t step);
  /// Callback which creates a new timeslice when timer
  /// expires and there is not a compatible datadriven callback
  /// available.
  static ExpirationHandler::Creator timeDrivenCreation(std::chrono::microseconds period);
  /// Callback which never expires records. To be used with, e.g.
  /// Lifetime::Timeframe.
  static ExpirationHandler::Checker expireNever();
  /// Callback which always expires records. To be used with, e.g.
  /// Lifetime::Transient.
  static ExpirationHandler::Checker expireAlways();
  /// Callback which expires records with the rate given by @a period, in
  /// microseconds.
  static ExpirationHandler::Checker expireTimed(std::chrono::microseconds period);

  /// Does nothing. Use this for cases where you do not want to do anything
  /// when records expire. This is the default behavior for data (which never
  /// expires via this mechanism).
  static ExpirationHandler::Handler doNothing();

  /// Build a fetcher for an object from CCDB when the record is expired.
  /// @a prefix is the lookup prefix in CCDB.
  /// @a overrideTimestamp can be used to override the timestamp found in the data.
  /// FIXME: provide a way to customize the namespace from the ProcessingContext
  static ExpirationHandler::Handler fetchFromCCDBCache(ConcreteDataMatcher const& matcher,
                                                       std::string const& prefix,
                                                       std::string const& overrideTimestamp,
                                                       std::string const& sourceChannel);

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
  static ExpirationHandler::Handler enumerate(ConcreteDataMatcher const& spec, std::string const& sourceChannel);
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_LIFETIMEHELPERS_H
